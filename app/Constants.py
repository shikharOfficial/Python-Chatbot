# Constants.py

import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
CHROMA_PATH = "chroma"
DATA_PATH = "data"
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')  # Add your Pinecone API key here
PINECONE_INDEX_NAME = "chatbot"  # Replace with your desired index name in Pinecone
