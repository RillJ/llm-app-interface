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

import json
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.chains.query_constructor.schema import AttributeInfo

load_dotenv()

# Create knowledge of app functions
docs = [
    Document(
        page_content="""
        An app that aims to enhance spatial navigation for visually impaired individuals by utilizing an intelligent AI-driven system that aids in grocery shopping.
        At the core of this is a bracelet equipped with vibration feedback. By varying the intensity of the vibration, the app guides users to products in a supermarket.
        The app is designed to interface with the bracelet. The goal is to empower blind users with greater independence and ease in their supermarket shopping experiences.
        """,
        metadata={"app_id": 1, "type": "app", "name": "Spatial Navigator"},
    ),
    Document(
        page_content="""
        A list where a user can add and remove products from that they want to aquire at a supermarket.
        The app will return the current grocery list as additional data.
        Return the product(s) that need to be added or removed, prefixed with either + (to add) or - (to remove), including the amount.
        Or if the user wants to know what's on the grocery list, then name the items that are on the grocery list in natural language.
        Interpret what to add or remove based on the user's input and current grocery list sent as additional data. 
        
        Example:
        The app might send the current grocery list as: [5]Elstar Apple; [4]Banana; [1]Quaker Oats
        The user might say: "I need 6 more apples and 2 packs of milk, but don't need the oats anymore."
        You will respond with: [+4]Elstar Apple; [2]Milk; [-1]Quaker Oats
        """,
        metadata={"app_id": 1, "type": "function", "label": "grocery-list", "additional-data-required": True},
    ),
    Document(
        page_content="Enables the user to scan barcodes of products to gain detailed nutritional information about them.",
        metadata={"app_id": 1, "type": "function", "label": "barcode-scanner", "additional-data-required": False},
    ),
    Document(
        page_content="""
        This group of functions manages a list of saved supermarket locations and supports navigation to them.
        The app will return the saved supermarkets and their metadata in JSON as additional data.

        The user may ask to:
        - remove a supermarket from their saved place list;
        - view the current place list;
        - get directions to a supermarket;
        - check opening hours;
        - find out whether a supermarket is currently open or closed.

        When the user wants to remove a supermarket from the list, respond only with the relevant place ID(s),
        using the exact format shown below, without adding any extra words, introductions, or explanations before or after the response:
        Format (use exactly as shown): <LLM>[-1]ChIJgUbEo8cfqokR5lP9_Wh_DaM; [-1]EicxMyBNYXJrZXQgU3QsIFdpbG1pbmd0b24sIE5DIDI4NDAxLCBVU0E

        Otherwise, respond to the user in clear, natural language paragraphs, using only the alphabet (A-Z), numbers (0-9), and spaces.
        Avoid using symbols, markdown, or bullet points, and focus on providing information in a straightforward, conversational way.
        """,
        metadata={"app_id": 1, "type": "function", "label": "navigation", "additional-data-required": True},
    ),
    Document(
        page_content="Can identify products in a supermarket and hands of the user in real time to help find the right products.",
        metadata={"app_id": 1, "type": "function", "label": "object-and-hand-recognition", "additional-data-required": False},
    ),
]

# additional_data_descriptions = {
#     "app_id": 1,
#     "label": "grocery-list",
#     "instruction": """
#         Extra Feature: Return the product(s) that need to be added, appended with either + (to add) or - (to remove), including the amount.
#         Example: [+4]Jonagold Apple; [-1]Quaker Oats"""
# }

# metadata_field_info = [
#     AttributeInfo(
#         name="type",
#         description="Whether this document describes the app itself or a functionality within the app.",
#         type="string",
#     ),
#     AttributeInfo(
#         name="name",
#         description="The name of the app.",
#         type="string",
#     ),
#     AttributeInfo(
#         name="app_id",
#         description="The unique identifyer of the app. Connects app descriptions to app functionality.",
#         type="integer",
#     ),
#     AttributeInfo(
#         name="label",
#         description="The unique label of a function within an app.",
#         type="string",
#     )
# ]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(docs, embeddings)

def metadata_retriever(k = 3, type = "function"):
    """
    Retrieves metadata using a vector store retriever.
    K can be as low as 1 and as high as infinity - 1.
    Type can be either "function" or "app".
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "filter": {"type": type}},
    )
    return retriever

def get_document_content(query_string: str) -> str:
    """
    Returns page_content of document matching label.
    Input format: "label=value" or "label=value;other-filter=value"
    Only uses label for matching, ignores other filters.
    """
    # Parse the query string
    params = dict(param.split("=") for param in query_string.split(';'))
    label = params.get("label", "").strip()
    
    # Search through docs
    for doc in docs:
        if doc.metadata.get("label") == label:
            return doc.page_content
            
    return "Document not found"

@tool
def app_functions(query: str) -> str:
    """
    Gives the most relevant app functions,
    including descriptions of it's purpose and accompanying metadata,
    based on the input string.
    """
    retriever = metadata_retriever(3, "function") # Get function metadata
    docsList = retriever.invoke(query)
    strList = []
    # Export the document content as well as its metadata
    for doc in docsList:
        strList.append({"document" : doc.page_content ,"metadata" : doc.metadata})
    return json.dumps(strList)

@tool
def get_current_datetime() -> str:
    """
    Returns the current date and time, including the day of the week,
    in a human-readable format. 
    """
    now = datetime.now()
    formatted_time = now.strftime("%A, %Y-%m-%d %H:%M:%S")
    return f"The current date and time is (YY-MM-DD, 24h): {formatted_time}"
# @tool
# def determine_input_type(query: str) -> str:
#     """
#     Determines whether the command received from the accessibility app is a user or app message.
#     System messages: A natural language command from the user, asking to perform a specific task or function.
#     App messages: Data returned by the app to complete a specific instruction sent by you.
#     """
#     if str.startswith("<User>"):
#         return "user"
#     elif str.startswith("<App>"):
#         return "app"
#     return "undeterminable"