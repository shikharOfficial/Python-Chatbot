import os
import shutil
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from html_loader import HTMLDirectoryLoader 

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    # Load HTML documents
    html_loader = HTMLDirectoryLoader(DATA_PATH)
    html_documents = html_loader.load()

    return html_documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    clear_database()
    db = Chroma(embedding_function=get_embedding_function())
    db.add_documents(chunks)
    print(f"Added {len(chunks)} chunks to Chroma.")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
