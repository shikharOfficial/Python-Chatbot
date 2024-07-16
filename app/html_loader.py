import os
from langchain_community.document_loaders import BSHTMLLoader
from concurrent.futures import ThreadPoolExecutor, as_completed

class HTMLDirectoryLoader:
    def __init__(self, directory):
        self.directory = directory

    def load(self):
        documents = []
        with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers based on system capacity
            futures = []
            for root, _, files in os.walk(self.directory):
                for filename in files:
                    if filename.endswith(".html"):
                        filepath = os.path.join(root, filename)
                        futures.append(executor.submit(self.load_file, filepath))

            for future in as_completed(futures):
                documents.extend(future.result())

        return documents

    def load_file(self, filepath):
        loader = BSHTMLLoader(filepath)
        return loader.load()
