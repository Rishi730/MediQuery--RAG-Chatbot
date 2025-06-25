#app.py new
import os
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import gradio as gr

# Load BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

def encode_with_biobert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

class BioBERTEmbedder(Embeddings):
    def embed_documents(self, texts):
        return [encode_with_biobert(text) for text in texts]

    def embed_query(self, text):
        return encode_with_biobert(text)

# Load vector DB
persist_directory = "./chroma_db"
embedding_function = BioBERTEmbedder()
db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

# Load QA model
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

# RAG chatbot logic
def rag_chatbot(query):
    if not query.strip():
        return "Please enter a medicine name or question."

    # Convert short name input to question
    if len(query.split()) <= 3:
        query = f"What is {query.strip()} used for?"

    # Search DB
    results = db.similarity_search(query, k=1)
    if not results:
        return "Sorry, I couldn't find relevant medical info."

    context = results[0].page_content
    prompt = f"Answer the question based on the context.\n\nQuestion: {query}\nContext: {context}\nAnswer:"
    response = qa_model(prompt, max_new_tokens=100)[0]["generated_text"]
    return response.strip()

# Gradio UI
iface = gr.Interface(
    fn=rag_chatbot,
    inputs="text",
    outputs="text",
    title="🧪 Medicine Info Chatbot",
    description="Ask a question or enter a medicine name to know what it's used for.",
)

if __name__ == "__main__":
    iface.launch()
