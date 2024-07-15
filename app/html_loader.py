import os
from langchain_community.document_loaders import BSHTMLLoader

class HTMLDirectoryLoader:
    def __init__(self, directory):
        self.directory = directory

    def load(self):
        documents = []
        for root, _, files in os.walk(self.directory):
            for filename in files:
                if filename.endswith(".html"):
                    filepath = os.path.join(root, filename)
                    loader = BSHTMLLoader(filepath)
                    documents.extend(loader.load())
        return documents