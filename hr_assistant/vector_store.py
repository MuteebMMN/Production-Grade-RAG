"""Step 4: store chunk embeddings in Qdrant Cloud so we can search them later."""

import os 
from langchain_community.vectorstores import FAISS
from . import config

from .embeddings import get_embeddings_model

def build_vector_store(chunks):
    """Embed every chunks and build a searchable FAISS index in memory"""
    embeddings_model=get_embeddings_model()
    return FAISS.from_documents(chunks,embeddings_model)

def save_vector_store(vector_store, path:str=config.VECTOR_STORE_PATH)->None:
    """
    Save the FAISS index to disk so we dont have to rebuild it every time.
    """
    vector_store.save_local(path)

def load_vector_store(path:str=config.VECTOR_STORE_PATH):
    """ load a previously saved FAISS index from disk"""
    embeddings_model=get_embeddings_model()
    return FAISS.load_local(path,embeddings_model,allow_dangerous_deserialization=True)

def vector_store_exist(path:str=config.VECTOR_STORE_PATH)->bool:
    """Check if a saved FAISS index already existon disk"""
    return os.path.exists(os.path.join(path,"index.faiss"))

def get_retriever(vector_store,k:int=config.TOP_K_RESULTS):
    """Turn a vector store into a retriever that return the top-K matching chunks."""
    return vector_store.as_retriever(search_kwargs={"k":k})