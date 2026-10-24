from . import config
from .agent import create_hr_agent
from .document_loader import load_document
from .llm import get_llm
from .splitter import split_into_chunks
from .tools import create_search_tool

from .vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store,
    get_retriever,
    vector_store_exist
)
def build_vector_store_for_document(file_path:str=config.DATA_FILE_PATH):
    """load + split + embed the document, reusing a saved index if we have one"""
    if vector_store_exist():
        print(" Founda saved vector store on disk, loadingit (fast, no rembedding)")
        return load_vector_store()
    print("no saved Vector store found, building  one from scratch....")
    documents=load_document(file_path)
    chunks=split_into_chunks(documents)
    print(f"loaded '{file_path}' and split it into {len(chunks)} chunks.")

    vector_store=build_vector_store(chunks)
    save_vector_store(vector_store)
    print("vector store built and saved to disk for next time.")
    return vector_store


def build_hr_assistant(file_path: str=config.DATA_FILE_PATH):

    """build the full RAG agent, ready to answer questions."""
    config.check_api_keys()

    vector_store=build_vector_store_for_document(file_path)
    retriever= get_retriever(vector_store)
    search_tool=create_search_tool(retriever)

    llm=get_llm() 
    agent=create_hr_agent(llm,[search_tool])
    return agent


def ask(agent,question:str)->str:
    """Ask the agent a question and return its final answer as plain text."""
    response=agent.invoke({"messages":[{"role":"user","content":question}]})
    return response["messages"][-1].content