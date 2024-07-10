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

from typing import Any, List, Union
from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages

load_dotenv()

# If we want, we can use other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [app_functions]

# Create OpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=False)
model_with_tools = model.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

# Create parser
parser = StrOutputParser()

# RAG
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Define memory
memory = SqliteSaver.from_conn_string(":memory:")
config = {"configurable": {"thread_id": "abc123"}}

# Define prompt
system = """
You are an assistant integrated with various accessibility apps designed to help visually impaired users.
Your task is to help the user activate the correct function within an app based on their command.

Follow these steps:
1. Match Function: Identify the app function that best matches the user's command.
2. Output Format: Return only the function label in the format: label=<app label>. Do not include any additional information or explanations.
3. Clarification Loop:
   - If you are not confident which function matches the command, ask the user for more details.
   - Continue this process until you can confidently identify the correct function.
4. Strict Format Requirement: Only return the function label in the specified format. Do not include any other text or explanation.

Example Workflows:
    User Command: "Can you help me look for the apples?"
        You found some relevant functions within the app.
        If confident, respond with: label=object-recognition
        If not confident, ask: "Do you want to find the apple near you or want to navigate to your favorite supermarket?"
    User Command: "I haven't spoken to my dad in a while, can you call him?"
        You did not find any relevant functions within the app, or the user is asking you something unrelated.
        Since you are not confident, ask: "At the moment, it seems that I cannot assist you with that. Can you clarify what you want?"
    User Command: "How are you?"
        You are here to assist with app functions, not to chat.
        Respond with: "While I'd love to talk, my job is to assist you with the accessibility app. How can I help you today?"
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")])

# Trim prompts
def prompt_trimmer(messages: List[Union[HumanMessage, AIMessage, FunctionMessage]]):
    """Trims the prompt to a reasonable length."""
    return messages[-10:] # Keep last 10 messages
    # Keep in mind that when trimming we may want to keep the system message!

# Create agent
# agent = create_tool_calling_agent(model, tools, prompt)
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    # | prompt_trimmer
    | model_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

class Input(BaseModel):
    """We need to add these input/output schemas because the current AgentExecutor is lacking in schemas."""
    input: str
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "input", "output": "output"}},
    )

class Output(BaseModel):
    output: Any

# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
# /stream_events
add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    )
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)