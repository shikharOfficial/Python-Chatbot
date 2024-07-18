import os
from flask import current_app
from langchain_community.vectorstores import Chroma
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

from app.Constants import CHROMA_PATH  # Make sure to update this with your actual path
from app.get_embedding_function import get_embedding_function

class QueryRetrieval:
    def __init__(self, database_directory):
        self.database_directory = database_directory
        self.embedding_function = get_embedding_function()
        self.db = Chroma(persist_directory=self.database_directory, embedding_function=self.embedding_function)

        # Initialize DistilBERT model and tokenizer for question answering
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
        self.model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

    def query(self, user_query, context_docs):
        try:
            answers = []
            for context_doc in context_docs:
                context = context_doc.page_content  # Retrieve the actual content of the document
                
                inputs = self.tokenizer.encode_plus(user_query, context, return_tensors='pt')
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                start_scores = outputs.start_logits
                end_scores = outputs.end_logits
                
                all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
                
                answers.append(answer)
            return answers
        except Exception as e:
            raise Exception(f"Exception during query retrieval: {e}")

def query_rag(query_text: str):
    current_dir = current_app.root_path
    full_path = os.path.abspath(os.path.join(current_dir, '..', CHROMA_PATH))
    query_retrieval = QueryRetrieval(full_path)
    
    # Retrieve the context from the database using Chroma
    context_docs = query_retrieval.db.similarity_search(query_text, k=5)
    
    result = query_retrieval.query(query_text, context_docs)
    
    print(f"Response: {result}")

    return result
