import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import PromptTemplate, format_document
from langchain_core.documents import Document
from typing import Sequence

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
    search_kwargs={"k": 3},
)

@tool
def app_functions(query: str) -> str:
    """Gives the most relevant app functions and accompanying labels based on the input command"""
    docsList = retriever.invoke(query)
    strList =[]
    # Export the document content as well as its metadata
    for doc in docsList:
        strList.append({'document' : doc.page_content ,'metadata' : doc.metadata})
    return json.dumps(strList)

# print(app_functions.name)
# print(app_functions.description)
# print(app_functions.args)