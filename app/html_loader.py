from langchain_community.document_loaders import BSHTMLLoader, DirectoryLoader

class HTMLDirectoryLoader:
    def __init__(self, directory):
        self.directory = directory

    def load(self):
        loader = DirectoryLoader(path=self.directory, glob="**/*.html", loader_cls=BSHTMLLoader)
        documents = loader.load()
        return documents
