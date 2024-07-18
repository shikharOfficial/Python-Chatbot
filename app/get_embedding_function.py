from sentence_transformers import SentenceTransformer

def get_embedding_function():
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    
    class CustomEmbeddingFunction:
        def __init__(self, model):
            self.model = model
        
        def __call__(self, text):
            # Convert text to embedding tensor and then to list
            embedding = self.model.encode(text, convert_to_tensor=True)  # Ensure this returns a tensor
            return embedding.cpu().numpy().tolist()  # Convert tensor to list

        def embed_documents(self, documents):
            # Embed a list of documents and convert each to a list
            return [self(document) for document in documents]

        def embed_query(self, query):
            # Embed a single query and convert it to a list
            return self(query)

    return CustomEmbeddingFunction(model)
