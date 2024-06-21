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

from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Create knowledge of app functions
docs = [
    # Document(
    #     page_content="""
    #     An app that aims to enhance spatial navigation for visually impaired individuals by developing an intelligent AI-driven system that aids in grocery shopping.
    #     At the core of this is a bracelet equipped with vibration feedback. By varying the intensity of the vibration, the app guides users to products in a supermarket.
    #     The app is designed to interface with the bracelet. The goal is to empower blind users with greater independence and ease in their supermarket hopping experiences.
    #     """,
    #     metadata={"type": "app-description"},
    # ),
    Document(
        page_content="A list where a user can add products they want to aquire at the supermarket.",
        metadata={"label": "grocery-list"},
    ),
    Document(
        page_content="Enables the user to scan barcodes of products to gain detailed nutritional information about them.",
        metadata={"label": "barcode-scanner"},
    ),
    Document(
        page_content="Allows the user to save their favorite supermarkets and start GPS navigation to this supermarket.",
        metadata={"label": "navigation"},
    ),
    Document(
        page_content="Can identify products in a supermarket and hands of the user in real time to help finding the right products.",
        metadata={"label": "object-and-hand-recognition"},
    ),
]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(docs, embeddings)

# metadata_field_info = [
#     AttributeInfo(
#         name="type",
#         description="Whether this document describes the app itself or functionality within the app.",
#         type="string",
#     ),
#     AttributeInfo(
#         name="label",
#         description="Description of certain functionality within the app.",
#         type="string",
#     ),
# ]

# ToDo: check similarity 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

#retriever.invoke("id like to buy apples and oranges at the supermarket later today")
#retriever.invoke("find right products")

# Create model
model = ChatOpenAI()

# Create parser
parser = StrOutputParser()

# RAG test
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

message = """
A visually impared user wants to use a certain app function.
You need to determine what function the user wants to use based on the context the user provides.
Only return the label obtained from the context.

Question:
{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

chain = {"question": RunnablePassthrough(), "context": retriever} | prompt | model | parser

#response = chain.invoke("i want to know how many calories this cereal has")
response = chain.invoke("i want to go to my favorite supermarket at 16:00")
print(response)

# # App definition
# app = FastAPI(
#   title="LangChain Server",
#   version="1.0",
#   description="A simple API server using LangChain's Runnable interfaces",
# )

# # Adding chain route for API

# add_routes(
#     app,
#     chain,
#     path="/chain",
# )

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="localhost", port=8000)