import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
CHROMA_PATH = "chroma"
DATA_PATH = "data"
