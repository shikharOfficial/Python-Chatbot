from transformers import BertTokenizer, BertModel
import torch

class BERTEmbeddings:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode

    def __call__(self, texts):
        return self.embed_texts(texts)

    def embed_texts(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings.tolist()  # Convert to list of lists

    def embed_documents(self, documents):
        texts = [doc for doc in documents]
        return self.embed_texts(texts)

    def embed_query(self, query):
        inputs = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings[0].tolist()  # Convert to list for single query

def get_embedding_function():
    return BERTEmbeddings()
