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

from tools import app_functions, get_document_content
from prompts import assistant_prompt, app_call_prompt

import os
import jwt #pyjwt
import uvicorn
from dotenv import load_dotenv #python-dotenv
from passlib.context import CryptContext
from typing import Any, List, Union, Annotated, Optional, Literal
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
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, RemoveMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

# Get environment variables (API keys and such)
load_dotenv()

# Create LLM model
tools = [app_functions]
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)
model_with_tools = model.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

class CustomState(MessagesState):
    """
    Custom state containing the list of messages (inherited)
    and also a summary of previous messages."""
    summary: Optional[str] = None

#region Node Definitions
def assistant(state: CustomState):
    """
    Main assistant responsible for operating with the system prompt.
    """
    function_name = assistant.__name__
    # Get messages summary if it exists
    summary = state.get("summary", "")

    # If there is a summary, then we add it
    if summary:
       system_message = f"Summary of conversation earlier: {summary}"
       # Append summary to any newer messages
       messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
       messages = state["messages"]

    response = model_with_tools.invoke([SystemMessage(content=assistant_prompt)] +
                                       [SystemMessage(content="Here is the history of the chat so far: ")] +
                                       messages)
    print(f"{function_name}: Response: {response.content}")
    return {"messages": response}

def summarize_conversation(state: CustomState, messages_to_keep = 5):
    """
    Summarize the conversation in 'messages' in the given state.
    Returns 'summary' and 'messages' keys usable in a state. 
    """
    function_name = summarize_conversation.__name__
    # Try to get a summary from the state
    summary = state.get("summary", "")

    # If a summary exists, prompt the model to extend it
    if summary:
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new mesages above:"
        )
    # If not, prompt the model to create a summary
    else:
        summary_message = "Create a summary of the conversation above:"
    
    print(f"{function_name}: Invoking summarization model...")
    
    messages = state["messages"][:] # Create a copy of the messages list
    messages_with_summary_message = messages + [HumanMessage(content=summary_message)] # Add prompt to our history
    response = model.invoke(messages_with_summary_message) # Summarize

    # Delete previous messages akin to "messages_to_keep"
    delete_messages = []

    # Remove tool-related messages to prevent OpenAI errors when first message is tool message
    while len(messages) > messages_to_keep and is_tool_related_message(messages[-messages_to_keep]):
        to_delete = messages.pop(-messages_to_keep)
        delete_messages.append(RemoveMessage(id=to_delete.id))
        print(f"{function_name}: Deleted tool-related message: {to_delete}")
        messages_to_keep -= 1

    # Remove all remaining messages except the last "messages_to_keep"
    for message in messages[:-messages_to_keep]:
        delete_messages.append(RemoveMessage(id=message.id))
        print(f"{function_name}: Deleted message: {message}")

    return {"summary": response.content, "messages": delete_messages}

def app_call(state: CustomState):
    """
    Invokes the model for processing the app's returned input.
    """
    function_name = app_call.__name__
    messages = state.get("messages", "")

    # Find the two last user messages (<App> or <User>) and last AI message
    last_user_messages = []
    last_ai_message = None
    
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            last_user_messages.append(message.content)
            if len(last_user_messages) == 2 and last_ai_message:
                break
        elif isinstance(message, AIMessage) and not last_ai_message:
            last_ai_message = message.content
    print(f"{function_name}: Last user messages: {last_user_messages}")
    print(f"{function_name}: Last AI message: {last_ai_message}")

    # Check if the last message was an app message
    try:
        if last_user_messages[0].startswith("<App>"):
            print(f"{function_name}: App message found. Instructing LLM.")

            additional_prompt = f"""
            The description of the function that was called and additional data was required for is outlined as follows:
            {get_document_content(last_ai_message)}
            
            The last two prompts instructed as follows:
            {last_user_messages[0]}
            {last_user_messages[1]}

            The app's additional data response was:
            {last_user_messages[0]}
            """

            full_prompt = app_call_prompt + additional_prompt
            print(f"{function_name}: Full prompt: {full_prompt}")
            response = model.invoke(full_prompt)
            print(f"{function_name}: Response: {response.content}")

            return {"messages": response}
        elif last_user_messages[0].startswith("<User>"):
            print(f"{function_name}: User message found. Not performing any action.")
        else:
            print(f"{function_name}: Could not identify user message type. Not performing any action.")
    except Exception as e:
        print(f"{function_name}: Could not identify message. Not performing any action. Error: {e}")

    return {"messages": messages}

#endregion

#region Node Helper Functions
def is_tool_related_message(message):
    """
    Check if a message is either an AIMessage with tool calls or a ToolMessage.
    """
    return (
        isinstance(message, AIMessage) and message.tool_calls
    ) or isinstance(message, ToolMessage)
#endregion

#region Edge Definitions
def should_summarize(state: CustomState, messages_key: str = "messages", summarize_at = 27):
    """
    Checks if the messages should be summarized, and if so, route accordingly.
    """
    function_name = should_summarize.__name__
    messages = state["messages"]

    # Code from tools_condition
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    print(f"{function_name}: Messages length: {len(messages)}")
    # Check if there's a tool call, and if so, go to the tools node
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        routing = "tools"
        print(f"{function_name}: Tool call detected. Routing to {routing}.")
    # If there are more than 'summarize_at' messages, then we summarize the conversation
    elif len(messages) > summarize_at:
        routing = "summarize_conversation"
        print(f"{function_name}: More than {summarize_at} messages detected. Routing to {routing}.")
    # If not, we can end the graph
    else:
        routing = END
        print(f"{function_name}: No tool call detected and no need to summarize. Routing to {routing}.")
    return routing
    
def should_execute_app_call(state: CustomState):
    """
    Checks if the message was an app call, and if so, route accordingly.
    """
    function_name = should_execute_app_call.__name__
    messages = state["messages"]
    
    if messages[-1].content.startswith("<App>"):
        routing = "app_call"
        print(f"{function_name}: App call detected. Routing to {routing}.")
    else:
        routing = "assistant"
        print(f"{function_name}: No app call detected. Routing to {routing}.")
    return routing

#endregion

# Graph
workflow = StateGraph(CustomState)

# Define nodes: these do the work
workflow.add_node("assistant", assistant)
workflow.add_node("app_call", app_call)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node(summarize_conversation)

# Define edges: these determine how the control flow moves
workflow.add_conditional_edges(
    START,
    should_execute_app_call,
    {
        "assistant": "assistant",
        "app_call": "app_call"
    }
)
workflow.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    should_summarize,
    {
        "tools": "tools",
        "summarize_conversation": "summarize_conversation",
        END: END
    }
)
workflow.add_edge("tools", "assistant")
workflow.add_edge("summarize_conversation", END)
workflow.add_edge("app_call", END)

memory = MemorySaver()
react_graph = workflow.compile(checkpointer=memory)

# Save graph visualisaiton
graph_visualisation = react_graph.get_graph(xray=True).draw_mermaid_png()
with open("output_graph_visualisation.png", "wb") as file:
    file.write(graph_visualisation)

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
    # Form data requires "python-multipart" to be installed
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
    #summary: Optional[str] = None # Not necessary to be input from the client

# class Output(BaseModel):
#     output: Any

# Let's define the API Handler
api_handler = APIHandler(
        # "sse_starlette" must be installed to implement the stream and stream_log endpoints
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
    uvicorn.run(app, host="192.168.50.21", port=8001, log_level="info")