import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import argparse
import json
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from arbert import ArabicEmbedder, get_relevant_context, chat

arabic_embedder = ArabicEmbedder()

with open("queries.json", "r", encoding="utf-8") as f:
    queries_data = json.load(f)

vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()
print("✅ Vault content found: ", len(vault_content), " lines found!")

if os.path.exists("vault_embeddings_arbert.pt"):
    print("Loading embeddings from file...")
    vault_embeddings_tensor = torch.load("vault_embeddings.pt")
    print("✅ Embeddings file exists: ", len(vault_embeddings_tensor), " embeddings found!")
else:
    print("❌ Embeddings file doesn't exist! Please generate embeddings first.")
    exit(1)

conversation_history = []

SIMILARITY_THRESHOLD = 0.6

results = []
for category, queries in queries_data.items():
    print(f"\n📂 Processing category: {category}")
    for query_data in queries:
        query = query_data["query"]
        ground_truth = query_data["ground_truth"]

        response = chat(query, vault_embeddings_tensor, vault_content, conversation_history)
        
        top_5_documents = response.split("\n")  

        top_5_documents = [doc.strip() for doc in top_5_documents if doc.strip() and doc.strip() != query]

        if len(top_5_documents) < 5:
            top_5_documents.extend([""] * (5 - len(top_5_documents)))

        document_similarities = []
        for doc in top_5_documents:
            if doc:  
                doc_embedding = arabic_embedder.get_embedding(doc)
                ground_truth_embedding = arabic_embedder.get_embedding(ground_truth)
                similarity = cosine_similarity(doc_embedding.unsqueeze(0).cpu(), ground_truth_embedding.unsqueeze(0).cpu())[0][0]
                document_similarities.append(float(similarity))
            else:
                document_similarities.append(0.0)  

        accuracy = 0.0
        recall = 0.0
        cumulative_correct = 0  
        total_pertinent = 1

        for i, sim in enumerate(document_similarities):
            if sim >= SIMILARITY_THRESHOLD:
                cumulative_correct += 1  
            else:
                cumulative_correct += 0  

            accuracy = cumulative_correct / (i + 1)

        best_match_index = np.argmax(document_similarities)
        best_match_similarity = float(document_similarities[best_match_index])  
        best_match_document = top_5_documents[best_match_index]
        recall = 1 if accuracy else 0

        results.append({
            "category": category,
            "query": query,
            "ground_truth": ground_truth,
            "top_5_documents": top_5_documents,
            "document_similarities": document_similarities,
            "best_match_document": best_match_document,
            "best_match_similarity": best_match_similarity,
            "accuracy": accuracy,
            "recall": recall
        })

        print(f"✅ Query processed: {query}")
        print(f"📊 Accuracy: {accuracy:.2f}, Recall: {recall:.2f}")

output_file = "evaluation_results_arbert.json"

if not os.path.exists(output_file):
    print(f"🆕 Creating new file: {output_file}")
else:
    print(f"📄 Updating existing file: {output_file}")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\n✅ Evaluation results saved to {output_file}")