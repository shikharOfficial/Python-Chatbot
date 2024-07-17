import os
from flask import current_app
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from app.get_embedding_function import get_embedding_function
from app.Constants import CHROMA_PATH, EMBEDDING_MODEL_NAME

PROMPT_TEMPLATE = """
Answer the question based on the context below. If you can't answer the question, reply "I don't know".
Ensure your responses are clear, concise, and helpful.

Context: {context}

Question: {question}

"""

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    current_dir = current_app.root_path
    full_path = os.path.abspath(os.path.join(current_dir, '..', CHROMA_PATH))
    print(full_path)
    db = Chroma(persist_directory=full_path, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model=EMBEDDING_MODEL_NAME)
    response_text = model.invoke(prompt)
    
    return response_text

    