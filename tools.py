import json
from dotenv import load_dotenv
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
        metadata={"app_id": 1, "name": "Spatial Navigator"},
    ),
    Document(
        page_content="""
        A list where a user can add and remove products from that they want to aquire at a supermarket.
        The app will return the current grocery list as additional data.
        Return the product(s) that need to be added, appended with either + (to add) or - (to remove), including the amount.
        Interpret based on the user's input and current grocery list 
        Example: [+4]Jonagold Apple; [-1]Quaker Oats""",
        metadata={"app_id": 1, "type": "function", "label": "grocery-list", "additional-data-required": True},
    ),
    Document(
        page_content="Enables the user to scan barcodes of products to gain detailed nutritional information about them.",
        metadata={"app_id": 1, "label": "barcode-scanner", "additional-data-required": False},
    ),
    Document(
        page_content="""
        Allows the user to save their favorite supermarkets and start GPS navigation to this supermarket.
        Extra Feature: If the user mentions a name of a supermarket, return the name of the supermarket.""",
        metadata={"app_id": 1, "label": "navigation", "additional-data-required": False},
    ),
    Document(
        page_content="Can identify products in a supermarket and hands of the user in real time to help find the right products.",
        metadata={"app_id": 1, "label": "object-and-hand-recognition", "additional-data-required": False},
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

def app_descriptions_retriever(k = 3):
    app_descriptions_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "filter": {"type": "app"}},
    )
    return app_descriptions_retriever

def app_functions_retriever(k = 3):
    app_functions_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "filter": {"type": "function"}},
    )
    return app_functions_retriever

@tool
def app_functions(query: str) -> str:
    """
    Gives the most relevant app functions, including descriptions of it's purpose and accompanying metadata, based on the input string.
    """
    retriever = app_functions_retriever(3)
    docsList = retriever.invoke(query)
    strList = []
    # Export the document content as well as its metadata
    for doc in docsList:
        strList.append({"document" : doc.page_content ,"metadata" : doc.metadata})
    return json.dumps(strList)

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