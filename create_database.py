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
    print(f"Current working directory: {os.getcwd()}")
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    print(f"Loading HTML documents from: {DATA_PATH}")
    html_loader = HTMLDirectoryLoader(DATA_PATH)
    html_documents = html_loader.load()

    return html_documents

def split_documents(documents: list[Document]):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    clear_database()
    db = Chroma.from_documents(chunks, get_embedding_function(), persist_directory=CHROMA_PATH)
    db.persist()
    print(f"Added {len(chunks)} chunks to Chroma at path: {CHROMA_PATH}")

    # Check if database file exists after adding documents
    print(f"Checking if Chroma database file exists at {CHROMA_PATH}...")
    if os.path.exists(CHROMA_PATH):
        print(f"Chroma database file found.")
    else:
        print(f"Chroma database file not found.")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        print(f"Clearing existing Chroma database at path: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    else:
        print(f"No existing Chroma database found at path: {CHROMA_PATH}")

if __name__ == "__main__":
    main()
