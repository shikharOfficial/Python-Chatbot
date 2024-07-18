from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from app.get_embedding_function import get_embedding_function
from app.Constants import EMBEDDING_MODEL_NAME, PINECONE_API_KEY, PINECONE_INDEX_NAME
from pinecone import Pinecone

PROMPT_TEMPLATE = """
Answer the question based on the context below. If you can't answer the question, reply "I don't know".
Ensure your responses are clear, concise, and helpful.

Context: {context}

Question: {question}

"""

def query_pinecone(query_text):
    embedding_function = get_embedding_function()
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    vector_query = embedding_function(query_text)
    results = pinecone.query(index_name=PINECONE_INDEX_NAME, query_vector=vector_query.tolist(), top_k=5)

    context_text = "\n\n---\n\n".join([doc['metadata']['page_content'] for doc in results['matches']])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model=EMBEDDING_MODEL_NAME)
    response_text = model.invoke(prompt)
    
    return response_text
