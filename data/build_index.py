#build_index.py new
import json
import os
import shutil
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

# Clear existing Chroma DB to avoid dimension mismatch
persist_directory = "./chroma_db"
shutil.rmtree(persist_directory, ignore_errors=True)
os.makedirs(persist_directory, exist_ok=True)

# Load BioBERT model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

# Encode text with BioBERT
def encode_with_biobert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Custom embedding wrapper for LangChain
class BioBERTEmbedder(Embeddings):
    def embed_documents(self, texts):
        return [encode_with_biobert(text) for text in texts]

    def embed_query(self, text):
        return encode_with_biobert(text)

# Load medicine data
with open("data/medicines.json", "r") as f:
    data = json.load(f)

# Prepare texts and metadata
documents = [f"{item['name']} is described as {item['uses']}" for item in data]
metadatas = [{"name": item["name"]} for item in data]
ids = [item["name"] for item in data]

# Create and persist vector DB
embedding_function = BioBERTEmbedder()
db = Chroma.from_texts(
    texts=documents,
    embedding=embedding_function,
    metadatas=metadatas,
    ids=ids,
    persist_directory=persist_directory,
)
db.persist()
print("✅ Successfully built index with BioBERT embeddings.")
