import os
from openai import OpenAI
import tiktoken
from collections import deque
from typing import Dict
from pprint import pprint
import numpy as np
import faiss
import json
import os
import openai
import pickle

# --- Config --- #
MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-ada-002"
CONTEXT_MAX_LENGTH = 30000
FORMAT_TOKENS = 3
API_KEY = os.environ.get("OPENAI_API_KEY")

SYSTEM_MESSAGE = """You are a helpful assistant that answers questions about Seekr's products based on their documentation. 
If the information is in the provided context, answer based on that. 
If the information is not in the context, say "I don't have specific information about that in the documentation."
Always be truthful, clear, and concise."""

# --- Setup --- #
client = OpenAI(api_key=API_KEY)
encoding = tiktoken.encoding_for_model(MODEL)

# Store chat history as a collection of objects
msg_history = deque()
token_sum = 0

# --- Helper Functions --- #
def update_msg_history(message: Dict[str, str]):
    '''Add message to history. Make sure history length is under token limit.'''
    global token_sum
    msg_history.append(message)
    token_sum += len(encoding.encode(message["content"])) + FORMAT_TOKENS

    # Remove older messages if exceeds token limit
    while msg_history and token_sum > CONTEXT_MAX_LENGTH:
        token_sum -= (len(encoding.encode(msg_history.popleft()["content"])))

# --- Search Vector DB for RAG --- #
def search_vector_database(query, top_k=5, index_path="faiss_index", chunks_path="chunks.pkl"):
    '''Search the vector database for chunks similar to the query.'''
    
    # Load the index
    index = faiss.read_index(index_path)
    
    # Load the chunks
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    # Generate an embedding for the query
    response = openai.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    query_embedding = response.data[0].embedding
    
    # Convert to numpy array
    query_embedding = np.array([query_embedding]).astype('float32')
    
    # Search the index
    distances, indices = index.search(query_embedding, top_k)
    
    # Return the most similar chunks
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk": chunks[idx],
            "distance": distances[0][i]
        })
    
    return results

# --- Call OpenAI API --- #
def openai_with_rag(user_input: str) -> str:
    # Create context from retrieval
    context_chunks = [result["chunk"]["content"] for result in search_vector_database(user_input)]
    context = "\n\n".join(context_chunks)

    user_content = f"""Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer this question: {user_input}"""
    
    update_msg_history({
        "role": "user",
        "content": user_content
    })

    try:
        msg_history.appendleft({
            "role": "system",
            "content": SYSTEM_MESSAGE
        })

        completion = client.chat.completions.create(
            model=MODEL,
            messages=list(msg_history)
        )

    except Exception as e:
        return f"An error occurred: {e}"
    
    update_msg_history(completion.choices[0].message.model_dump())
    return completion.choices[0].message.content

if __name__ == "__main__":
    # Get input from the user
    while True:
        user_input = input("\nEnter your prompt:\n")
        openai_reply = openai_with_rag(user_input)
        print(f"\nOpenAI:\n{openai_reply}")