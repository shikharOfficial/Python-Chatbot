import os
import shutil
import logging
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from app.get_embedding_function import get_embedding_function
from app.html_loader import HTMLDirectoryLoader

logging.basicConfig(level=logging.INFO)

class ProcessInputCreateDatabase:
    def __init__(self, input_directory, database_directory):
        self.input_directory = input_directory
        self.database_directory = database_directory

    def main(self):
        try:
            logging.info("Clearing Database...")
            self.clear_database()
            
            logging.info("Loading documents...")
            documents = self.load_documents()

            logging.info("Splitting documents...")
            chunks = self.split_documents(documents)

            logging.info("Adding documents to Chroma...")
            self.add_to_chroma(chunks)
        except Exception as e:
            logging.error(f"Exception during database creation: {e}")
            raise

    def load_documents(self):
        html_loader = HTMLDirectoryLoader(self.input_directory)
        return html_loader.load()

    def split_documents(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

    def add_to_chroma(self, chunks: list[Document]):
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=self.database_directory, embedding_function=embedding_function)
        db.add_documents(chunks)
        db.persist()

    def clear_database(self):
        if os.path.exists(self.database_directory):
            shutil.rmtree(self.database_directory)
