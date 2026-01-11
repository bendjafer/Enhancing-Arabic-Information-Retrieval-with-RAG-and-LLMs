import torch
import ollama
import os
import argparse
import json

def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=1):
    if vault_embeddings.nelement() == 0:
        return []
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context
   
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    relevant_context = get_relevant_context(user_input, vault_embeddings_tensor, vault_content, top_k=5)
    if relevant_context:
        context_str = "\n".join(relevant_context)
    else:
        print("No relevant context found.")
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input
    
    conversation_history.append({"role": "user", "content": user_input_with_context})
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    response = ollama.chat(
        model=ollama_model,
        messages=messages
    )
    
    conversation_history.append({"role": "assistant", "content": response['message']['content']})
    
    return response['message']['content']

parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama3.2:1b", help="Ollama model to use (default: llama3.2:1b)")
args = parser.parse_args()

vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()
print("✅ Vault content found: ", len(vault_content), " lines found!")

embeddings_file = "vault_embeddings_ollama.pt"
if os.path.exists(embeddings_file): 
    print("Loading embeddings from file...")
    vault_embeddings_tensor = torch.load(embeddings_file)
    print("✅ Embeddings file exists: ", len(vault_embeddings_tensor), " embeddings found!")
else:
    print("❌ Embeddings file doesn't exist!")
    print("Generating embeddings for the first 10 elements in the vault content...")
    vault_embeddings = []
    total_lines = len(vault_content)
    for idx, content in enumerate(vault_content):
        print(f"Generating embedding {idx + 1}/{total_lines}")
        response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
        embedding_tensor = torch.tensor(response["embedding"])
        vault_embeddings.append(embedding_tensor)

    print("Converting embeddings to tensor...")
    vault_embeddings_tensor = torch.stack(vault_embeddings)
    print("Saving embeddings to file...")
    torch.save(vault_embeddings_tensor, embeddings_file)

print("✅", len(vault_embeddings_tensor), "embeddings made!")

def test_rag():
    conversation_history = []
    system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant infromation to the user query from outside the given context."

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting Ollama Rag testing...")
            break
        
        response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
        print(f"\nAssistant: {response}")

print("\nRAG Testing Mode: Type your query or type 'exit' to quit.")
test_rag()