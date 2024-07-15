from langchain_community.embeddings.ollama import OllamaEmbeddings

from app.Constants import EMBEDDING_MODEL_NAME

def get_embedding_function():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url='http://127.0.0.1:11434')
    return embeddings
