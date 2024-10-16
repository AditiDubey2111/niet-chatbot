import faiss
import numpy as np
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import requests
import gradio as gr

# Load DPR context encoder
ctx_encoder = DPRContextEncoder.from_pretrained(
    'facebook/dpr-ctx_encoder-single-nq-base')
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    'facebook/dpr-ctx_encoder-single-nq-base')

# Example knowledge base documents
docs = ["Paris is the capital of France.",
        "Berlin is the capital of Germany.", "Tokyo is the capital of Japan.", "Noida Institute of Technology is situated in Noida, Uttar Pradesh."]

# Tokenize and encode documents
inputs = ctx_tokenizer(docs, return_tensors="pt",
                       padding=True, truncation=True)
doc_embeddings = ctx_encoder(**inputs).pooler_output.detach().cpu().numpy()

# Create FAISS index
index = faiss.IndexFlatIP(doc_embeddings.shape[1])  # Use inner product index
index.add(doc_embeddings)

# Save document references
doc_map = {i: doc for i, doc in enumerate(docs)}


def retrieve(query):
    # Tokenize and encode the query
    query_inputs = ctx_tokenizer(
        query, return_tensors="pt", padding=True, truncation=True)
    query_embedding = ctx_encoder(
        **query_inputs).pooler_output.detach().cpu().numpy()

    # Retrieve top documents from FAISS
    D, I = index.search(query_embedding, k=1)  # Get top 1 result
    retrieved_doc = doc_map[I[0][0]]
    return retrieved_doc


def generate_with_ollama(query, context):
    # Prepare input for the model
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"

    print("======================")
    print(input_text)
    print("========================")

    # Send request to Ollama running locally
    url = "http://localhost:11434/api/generate"  # Ollama's local API endpoint
    payload = {
        "model": "orca-mini",
        "prompt": input_text,
        "stream": False
    }

    response = requests.post(url, json=payload)

    print("========================")
    print(response)
    print("========================")

    result = response.json()
    print("========================")
    print(result)
    print("========================")

    return result['response']


def rag_pipeline(query):
    # Retrieve relevant document
    context = retrieve(query)
    print(f"Retrieved context: {context}")

    # Generate answer based on the query and retrieved document
    answer = generate_with_ollama(query, context)
    return answer


def rag_gradio(query):
    return rag_pipeline(query)


# Define the Gradio interface
iface = gr.Interface(fn=rag_gradio, inputs="text",
                     outputs="text", title="RAG with Ollama and FAISS")

# Launch the interface
iface.launch()
