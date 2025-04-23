from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

import os

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here" 