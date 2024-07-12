import os
from bs4 import BeautifulSoup
from langchain.schema.document import Document

class HTMLDirectoryLoader:
    def __init__(self, directory):
        self.directory = directory

    def load(self):
        documents = []
        for root, _, files in os.walk(self.directory):
            for filename in files:
                if filename.endswith(".html"):
                    filepath = os.path.join(root, filename)
                    with open(filepath, "r", encoding="utf-8") as file:
                        soup = BeautifulSoup(file, "html.parser")
                        text = soup.get_text()
                        metadata = {"source": filepath}
                        documents.append(Document(page_content=text, metadata=metadata))
        return documents
