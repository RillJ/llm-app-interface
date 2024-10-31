# Licensed to Osnabrück University under one or more
# contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from tools import app_functions
from prompts import system_prompt

import os
import jwt #pyjwt
import uvicorn
from dotenv import load_dotenv #python-dotenv
from passlib.context import CryptContext
from typing import Any, List, Union, Annotated, TypedDict
from jwt.exceptions import InvalidTokenError
from datetime import datetime, timedelta, timezone
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from langserve import APIHandler
from langgraph.graph import START, END, StateGraph, MessagesState, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from IPython.display import Image, display
#from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.pydantic_v1 import BaseModel, Field
#from pydantic.v1 import BaseModel
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages

# Get environment variables (API keys and such)
load_dotenv()

# Create LLM model
tools = [app_functions]
model = ChatOpenAI(model="gpt-4o", temperature=0, streaming=False)
model_with_tools = model.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

#region LangGraph
# State
class CustomState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    #app_feedback: str

# Agent
def assistant(state: CustomState):
   return {"messages": [model_with_tools.invoke([system_prompt] + state["messages"])]}

# Graph
builder = StateGraph(CustomState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
react_graph = builder.compile(checkpointer=memory)

# Save graph visualisaiton
graph_visualisation = react_graph.get_graph(xray=True).draw_mermaid_png()
with open("output_graph_visualisation.png", "wb") as file:
    file.write(graph_visualisation)
#endregion

# # Create prompt
# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     MessagesPlaceholder(variable_name="chat_history", optional=True),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad")])

# # Trim prompts
# def prompt_trimmer(messages: List[Union[HumanMessage, AIMessage, FunctionMessage]]):
#     """Trims the prompt to a reasonable length."""
#     return messages[-10:] # Keep last 10 messages
#     # Keep in mind that when trimming we may want to keep the system message!

# # Create agent
# agent = (
#     {
#         "input": lambda x: x["input"],
#         "agent_scratchpad": lambda x: format_to_openai_tool_messages(
#             x["intermediate_steps"]
#         ),
#         "chat_history": lambda x: x["chat_history"],
#     }
#     | prompt
#     # | prompt_trimmer
#     | model_with_tools
#     | OpenAIToolsAgentOutputParser()
# )
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# App definition and authentication
class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None

class UserInDB(User):
    hashed_password: str

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# JWT token variables
SECRET_KEY = os.environ.get("SECRET_KEY") #openssl rand -hex 32
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$7S1J9VKngWXJf8txEva6Z.XTy2eeX03pJF4RoV1dfK1Xj3/Cen7YS",
        "disabled": False,
    }
}

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db: dict, username: str) -> Union[UserInDB, None]:
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(fake_users_db, username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)]):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@app.post("/token")
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# @app.get("/users/me/", response_model=User)
# async def read_users_me(
#     current_user: Annotated[User, Depends(get_current_active_user)],
# ):
#     return current_user

# @app.get("/users/me/items/")
# async def read_own_items(
#     current_user: Annotated[User, Depends(get_current_active_user)],
# ):
#     return [{"item_id": "Foo", "owner": current_user.username}]

class Input(BaseModel):
    """We need to add these input/output schemas because the current AgentExecutor is lacking in schemas."""
    messages: str

# class Output(BaseModel):
#     output: Any

# Let's define the API Handler
api_handler = APIHandler(
        react_graph.with_types(input_type=Input).with_config(
        {"run_name": "agent"}
    ),
    # Namespace for the runnable.
    # Endpoints like batch / invoke should be under /my_runnable/invoke
    # and /my_runnable/batch etc.
    path="/llm-app-interface"
)

@app.post("/llm-app-interface/invoke")
async def invoke_with_auth(
    request: Request,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> Response:
    """Handle a request."""
    # The API Handler validates the parts of the request
    # that are used by the runnnable (e.g., input, config fields)
    config = {"configurable": {"user_id": current_user.username, "thread_id": hash(current_user.username)}}
    return await api_handler.invoke(request, server_config=config)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)