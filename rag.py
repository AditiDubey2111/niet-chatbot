import os
import faiss
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import gradio as gr

import scraper  # scraper.py file

# Scrape the website to generate 'knowledge'
scraper.crawl_website()


# Load DPR context encoder
ctx_encoder = DPRContextEncoder.from_pretrained(
    'facebook/dpr-ctx_encoder-single-nq-base')
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    'facebook/dpr-ctx_encoder-single-nq-base')


# Read documents from the scraped_pages folder
docs = []
scraped_pages_folder = os.path.join(os.path.dirname(__file__), 'scraped_pages')
for filename in os.listdir(scraped_pages_folder):
    if filename.endswith('.txt'):
        with open(os.path.join(scraped_pages_folder, filename), 'r', encoding='utf-8') as file:
            docs.append(file.read())


# Hard-coded for now because the scraped documents are not working well in the RAG pipeline
# Taken from the website of the college
docs = [
    "NIET is one of the premier Engineering and Management institutes of India's National Capital Region (NCR)."]


# Tokenize and encode documents
inputs = ctx_tokenizer(docs, return_tensors="pt",
                       padding=True, truncation=True)
doc_embeddings = ctx_encoder(**inputs).pooler_output.detach().cpu().numpy()


# Create FAISS index (a vector data structure that holds all our "knowledge" in a format understandable by the LLM)
index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(doc_embeddings)


# Save document references
doc_map = {i: doc for i, doc in enumerate(docs)}


def retrieve(query):
    # Tokenize the query
    query_inputs = ctx_tokenizer(
        query, return_tensors="pt", padding=True, truncation=True)

    # Encode the query
    query_embedding = ctx_encoder(
        **query_inputs).pooler_output.detach().cpu().numpy()

    # Search the FAISS index for the "knowledge" that most closely matches the query
    D, I = index.search(query_embedding, k=1)
    retrieved_doc = doc_map[I[0][0]]

    # Read the URL associated with the .txt document
    url = retrieved_doc.splitlines()[0][len("URL: "):]

    return (retrieved_doc, url)
