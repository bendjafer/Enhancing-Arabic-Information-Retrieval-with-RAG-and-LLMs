# Enhancing Arabic Information Retrieval with RAG and LLMs

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) system specifically designed for Arabic Information Retrieval (IR). The project focuses on integrating advanced embedding models with Large Language Models (LLMs) to overcome the limitations of traditional keyword-based search systems.

## Overview

<img width="1769" height="938" alt="Execution order" src="https://github.com/user-attachments/assets/4870aa74-2857-44e9-bc28-8e97e104bddc" />


Arabic Information Retrieval often faces challenges with context and semantics. This project implements a RAG pipeline that combines retrieval-based methods with generative models. By leveraging external knowledge from a curated corpus, the system improves response accuracy and ensures factual relevance in Arabic natural language processing.

## System Architecture

The architecture is divided into three main modules. You can use the following sections to place your architectural diagrams.

### 1. Document Embedding and Indexing
This module is responsible for preparing the knowledge base.
* Text Preprocessing: Documents are cleaned, normalized, and split into 500-character chunks to ensure high retrieval precision.
* Vector Generation: Text chunks are converted into dense vector representations using pre-trained models such as ARBERT or Ollama (mxbai-embed-large).
* Storage: The generated embeddings are saved as persistent files (.pt) to allow for efficient similarity matching during the retrieval phase.


### 2. Retrieval-Augmented Query Processing
This component manages the interaction between the user's input and the indexed data.
* Query Embedding: The user's query is transformed into a vector using the same embedding model as the document vault.
* Semantic Retrieval: The system calculates the cosine similarity between the query vector and the document vectors.
* Context Selection: The top-k most relevant document chunks are retrieved and fused with the original query to provide a rich context for the LLM.


### 3. LLM Response Generation
The final module generates a coherent and contextually accurate response.
* GPT-4 Integration: Uses the OpenAI API to generate high-quality responses based on the retrieved Arabic context.
* Local LLM (Ollama): Utilizes Llama 3.2:1b as a local alternative for privacy-focused or offline environments.
* Post-processing: Refines the generated text to ensure it meets the required quality and relevance standards.


## Model Comparison

The project evaluates two primary configurations:

| Component | ARBERT-based RAG | Ollama-based RAG |
| :--- | :--- | :--- |
| Embedding Model | ARBERT (Specialized Arabic BERT) | mxbai-embed-large |
| LLM Generator | GPT-4 (Cloud-based) | Llama 3.2:1b (Local) |
| Environment | API-dependent | Fully Local |

## Key Findings and Performance

* Chunk Optimization: Reducing the chunk length from 1000 to 500 characters significantly enhanced retrieval performance and accuracy.
* Retrieval Accuracy: ARBERT demonstrated superior precision and Mean Average Precision (MAP), making it highly effective for targeted Arabic retrieval.
* Recall: Ollama showed higher recall, indicating it can capture a broader range of relevant documents, though sometimes at the cost of precision.
* Overall Synergy: The combination of ARBERT for retrieval and GPT-4 for generation provided the most robust and accurate results.

## Implementation Details

The repository includes the following core scripts:
* upload.py: Data cleaning, normalization, and chunking of the Arabic corpus.
* arbert.py: Implementation of the ARBERT embedding and retrieval loop.
* llama.py: Implementation of the local Ollama-based RAG pipeline.
* ilm.py: Integration script merging ARBERT retrieval with GPT-4 generation.
* evaluation scripts: Tools for calculating Precision, Recall, F1-score, and Mean Average Precision.
