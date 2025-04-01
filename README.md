# 🧠 Know-Your-PDF
[![Hugging Face Spaces](https://img.shields.io/badge/Live%20Demo-HuggingFace-blue)](https://rahulhshah-know-your-pdf.hf.space/)

## 📌 Overview

**Know Your PDF** is a Retrieval-Augmented Generation (RAG) based application that allows users to interact with any PDF using natural language queries. Upload your PDF and ask questions—the app returns accurate, context-aware answers using LLaMA 3B and LangChain.

🔗 **Live Demo**: [https://rahulhshah-know-your-pdf.hf.space/](https://rahulhshah-know-your-pdf.hf.space/)

---

## 🛠️ Features

- 📄 Upload and process any PDF file
- 🔍 Chunking and dense vector embeddings using `BAAI/bge-base-en-v1.5`
- ⚡ Fast semantic search via FAISS
- 🧠 LLM-powered answers using `meta-llama/Llama-3.2-3B-Instruct`
- 🔗 Seamless integration with LangChain’s RetrievalQA
- 🧪 Evaluation metrics: BLEU, ROUGE, BERTScore, token-level F1, cosine similarity, and latency
- 🌐 Interactive web interface built using Gradio

---

## 📂 Project Structure

- `app.py` – Gradio app for PDF Q&A powered by LangChain and LLaMA.
- `know_your_pdf.py` – Notebook/script for the entire RAG and LLM pipeline and evaluating the RAG pipeline on custom queries.
- `requirements.txt` – Python dependencies.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/know-your-pdf.git
cd know-your-pdf
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the app locally
```bash
python app.py
```

---

## 🧠 Models Used
- Embeddings: BAAI/bge-base-en-v1.5
- Language Model: meta-llama/Llama-3.2-3B-Instruct
  
---

## 📈 Evaluation Metrics
The evaluation script includes support for:
- BLEU
- ROUGE
- BERTScore
- Token-level F1 Score
- Cosine Similarity
- Retrieval & generation latency tracking
