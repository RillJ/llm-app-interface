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

from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

load_dotenv()

# If we want, we can use other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [app_functions]

# Create model
model = ChatOpenAI()

# Create parser
parser = StrOutputParser()

# RAG test
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Define memory
memory = SqliteSaver.from_conn_string(":memory:")
config = {"configurable": {"thread_id": "abc123"}}

system = """
You are an assistant helping users find the right app functions.
You are given a command from the user and a limited set of the app's functionalities.
Do none of the functions seem related to the command? Then ask the user to clarify.
Return ONLY THE LABEL of the function if you are confident that the context matches the function.
This label will be used by the app.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")])

# Create agent
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# chain = {"command": RunnablePassthrough(), "functions": retriever} | prompt | model | parser

# App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# Adding chain route for API

add_routes(
    app,
    agent_executor,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)