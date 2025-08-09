from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModel
import torch
from ecommbot.data_converter import dataconveter

load_dotenv()

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Load the Mistral model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

def mistral_generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling to get a single vector per input
    return embeddings.squeeze().numpy()  # Convert to a numpy array

def ingestdata(status):
    vstore = AstraDBVectorStore(
            embedding=mistral_generate_embedding,
            collection_name="chatbotecomm",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_KEYSPACE,
        )
    
    storage = status
    
    if storage == None:
        docs = dataconveter()
        inserted_ids = vstore.add_documents(docs)
    else:
        return vstore
    return vstore, inserted_ids

if __name__ == '__main__':
    vstore, inserted_ids = ingestdata(None)
    print(f"\nInserted {len(inserted_ids)} documents.")
    results = vstore.similarity_search("can you tell me the low budget sound basshead.")
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")
