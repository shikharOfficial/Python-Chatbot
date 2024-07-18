import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from app.get_embedding_function import get_embedding_function
from app.html_loader import HTMLDirectoryLoader
from app.Constants import PINECONE_API_KEY, PINECONE_INDEX_NAME
import uuid  # for generating unique IDs

logging.basicConfig(level=logging.INFO)

class ProcessInputCreateDatabase:
    def __init__(self, input_directory):
        self.input_directory = input_directory
        self.embedding_function = get_embedding_function()
        # Initialize Pinecone client
        self.pinecone = Pinecone(api_key=PINECONE_API_KEY)
        
        embedding_sample = self.embedding_function("sample text")
        embedding_dimension = len(embedding_sample)
        print(embedding_dimension)
        
        if PINECONE_INDEX_NAME not in self.pinecone.list_indexes():
            self.pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=embedding_dimension,  # Use the correct dimension
                metric='cosine'  # Use the appropriate metric
            )
        self.index = self.pinecone.Index(PINECONE_INDEX_NAME)

    def main(self):
        try:
            logging.info("Loading documents...")
            # documents = self.load_documents()

            # logging.info("Splitting documents...")
            # chunks = self.split_documents(documents)

            # logging.info("Adding documents to Pinecone...")
            # self.add_to_pinecone(chunks)
        except Exception as e:
            logging.error(f"Exception during database creation: {e}")
            raise

    def load_documents(self):
        html_loader = HTMLDirectoryLoader(self.input_directory)
        return html_loader.load()

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=300,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

    def add_to_pinecone(self, chunks):
        vectors = []

        for chunk in chunks:
            vector = self.embedding_function(chunk.page_content)
            # Generate a unique ID for each chunk
            unique_id = str(uuid.uuid4())
            vectors.append({"id": unique_id, "values": vector, "metadata": chunk.metadata})

        # Upsert items into Pinecone index
        self.index.upsert(vectors)