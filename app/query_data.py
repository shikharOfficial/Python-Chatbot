import os
from flask import current_app
from langchain_community.vectorstores import Chroma
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

from app.Constants import CHROMA_PATH
from app.get_embedding_function import get_embedding_function

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

def query_rag(query_text: str):
    # Initialize the embedding function and load the Chroma database
    embedding_function = get_embedding_function()
    current_dir = current_app.root_path
    full_path = os.path.abspath(os.path.join(current_dir, '..', CHROMA_PATH))
    db = Chroma(persist_directory=full_path, embedding_function=embedding_function)

    # Perform the similarity search
    results = db.similarity_search(query_text, k=5)
    print(f"Results: {results}")  # Print the structure of results

    # Check the type and content of the first result to determine the structure
    first_result = results[0]
    print(f"First result type: {type(first_result)}")
    print(f"First result content: {first_result}")

    # Adjust the unpacking based on the structure
    if hasattr(first_result, 'page_content'):
        # If the result is a Document object with a 'page_content' attribute
        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    else:
        # If the result is a tuple (Document, score)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    print(f"Context Text: {context_text[:1000]}")  # Print a portion of the context text for inspection

    # Tokenize the query and context
    inputs = tokenizer.encode_plus(
        query_text,
        context_text,
        add_special_tokens=True,
        max_length=512,  # BERT’s maximum input length
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']

    # Print tensor shapes for debugging
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Token Type IDs shape: {token_type_ids.shape}")
    print(f"Attention Mask shape: {attention_mask.shape}")

    # Verify BERT model input requirements
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

    # Print output shapes for verification
    print(f"Start logits shape: {outputs.start_logits.shape}")
    print(f"End logits shape: {outputs.end_logits.shape}")

    # Extract start and end positions for the answer
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Find the tokens with the highest `start` and `end` scores
    start_index = torch.argmax(start_logits, dim=1).item()
    end_index = torch.argmax(end_logits, dim=1).item() + 1

    # Ensure start_index is less than end_index
    if start_index > end_index:
        start_index, end_index = end_index, start_index

    # Convert token indices to text
    answer_tokens = input_ids[0, start_index:end_index]  # Keep batch dimension
    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # Print the answer tokens and text for debugging
    print(f"Answer tokens: {answer_tokens}")
    print(f"Answer text: {answer_text}")

    return answer_text
