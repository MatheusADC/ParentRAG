from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

import os

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here" 

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=500)

pdf_link = "your_pdf_link_here" 

loader = PyPDFLoader(pdf_link, extract_images=False)

pages = loader.load_and_split()

len(pages)

child_splitter = RecursiveCharacterTextSplitter(chunk_size = 200)

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 4000,
    chunk_overlap = 200,
    length_function = len,
    add_start_index = True
)

store = InMemoryStore()
vectorstore = Chroma(embedding_function=embeddings, persist_directory='childVectorDB')

parent_document_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

parent_document_retriever.add_documents(pages, ids=None)

parent_document_retriever.vectorstore.get()

TEMPLATE = """
    Você é um especialista em legislação e tecnologia. Responda a pergunta abaixo utilizando o contexto informado.
    Query: 
    {question}

    Context:
    {context}
"""

rag_prompt = ChatPromptTemplate.from_template(TEMPLATE)

setup_retrieval = RunnableParallel({"question": RunnablePassthrough(), "context": parent_document_retriever})

output_parser = StrOutputParser()

parent_chain_retrieval = setup_retrieval | rag_prompt | llm | output_parser

parent_chain_retrieval.invoke("Quais os principais riscos do marco legal de ia")
