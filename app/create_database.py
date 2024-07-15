import os
import shutil
import logging
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from app.get_embedding_function import get_embedding_function
from app.html_loader import HTMLDirectoryLoader
from app.Constants import CHROMA_PATH

logging.basicConfig(level=logging.DEBUG)

class ProcessInputCreateDatabase:
    def __init__(self, input_directory):
        self.input_directory = input_directory
        
        
    def main(self):
        try: 
            logging.debug("Loading documents...")
            documents = self.load_documents(self.input_directory)
            
            logging.debug("Splitting documents...")
            chunks = self.split_documents(documents)
            
            logging.debug("Adding documents to Chroma...")
            print(chunks)
            self.add_to_chroma(chunks)
        except Exception as e:
            print(f"Exception during database creation: {e}")
            raise

    def load_documents(self, data_folder_path):
        html_loader = HTMLDirectoryLoader(data_folder_path)
        html_documents = html_loader.load()
        return html_documents

    def split_documents(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

    def add_to_chroma(self, chunks: list[Document]):
        self.clear_database()
        db = Chroma.from_documents(chunks, get_embedding_function(), persist_directory=CHROMA_PATH)
        logging.debug(f"Added {len(chunks)} chunks to Chroma at path: {CHROMA_PATH}")

    def clear_database(self):
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
