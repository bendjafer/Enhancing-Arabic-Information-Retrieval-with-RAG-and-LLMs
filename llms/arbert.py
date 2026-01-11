import os
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ArabicEmbedder:
    def __init__(self):
        self.encoder = SentenceTransformer('UBC-NLP/ARBERT')

    def get_embedding(self, text):
        return self.encoder.encode(text, convert_to_tensor=True)

def get_relevant_context(user_input, vault_embeddings, vault_content, top_k=1):
    if len(vault_embeddings) == 0:
        return []
    input_embedding = arabic_embedder.get_embedding(user_input)
    cos_scores = cosine_similarity(input_embedding.unsqueeze(0).cpu(), vault_embeddings.cpu())
    top_k = min(top_k, len(cos_scores[0]))
    top_indices = np.argsort(cos_scores[0])[-top_k:][::-1]
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def chat(user_input, vault_embeddings, vault_content, conversation_history):
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content, top_k=5)
    if relevant_context:
        context_str = "\n".join(relevant_context)
    else:
        print("No relevant context found.")
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input
    
    conversation_history.append({"role": "user", "content": user_input_with_context})
    
    assistant_response = f"\n {user_input_with_context}"
    
    conversation_history.append({"role": "assistant", "content": assistant_response})
    
    return assistant_response

arabic_embedder = ArabicEmbedder()

vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()
print("✅ Vault content found: ", len(vault_content), " lines found!")

embeddings_file = "vault_embeddings_arbert.pt"
if os.path.exists(embeddings_file):
    print("Loading embeddings from file...")
    vault_embeddings_tensor = torch.load(embeddings_file)
    print("✅ Embeddings file exists: ", len(vault_embeddings_tensor), " embeddings found!")
else:
    print("❌ Embeddings file doesn't exist!")
    print("Generating embeddings for the vault content...")
    vault_embeddings = []
    total_lines = len(vault_content)
    for idx, content in enumerate(vault_content):
        print(f"Generating embedding {idx + 1}/{total_lines}...")
        embedding = arabic_embedder.get_embedding(content)
        vault_embeddings.append(embedding)

    print("Converting embeddings to tensor...")
    vault_embeddings_tensor = torch.stack(vault_embeddings)
    print("Saving embeddings to file...")
    torch.save(vault_embeddings_tensor, embeddings_file)

print("✅", len(vault_embeddings_tensor), "embeddings made!")

def test_rag():
    conversation_history = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting ARBERT Rag testing...")
            break
        
        response = chat(user_input, vault_embeddings_tensor, vault_content, conversation_history)
        print(f"\nAssistant: {response}")

print("\nRAG Testing Mode: Type your query or type 'exit' to quit.")
test_rag()