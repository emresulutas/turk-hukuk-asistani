# ⚖️ Local Legal Assistant AI (RAG + Gemini 1.5)

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced **RAG (Retrieval-Augmented Generation)** application designed to perform semantic search and Q&A on local legal PDF documents (Laws, Regulations, Case Law). It utilizes **Google Gemini 2.5 Flash** for reasoning and **ChromaDB** for vector storage.

Unlike standard keyword search, this assistant understands the **legal context**, interprets articles, and provides cited answers.

---

## 🚀 Key Features

* **🧠 Hybrid Search (Fusion):** Combines Vector Search (Auto-Merging Retriever) and Keyword Search (BM25) for maximum accuracy.
* **📑 Smart Chunking:** Uses Hierarchical Node Parsing (Parent-Child) to preserve the context of legal articles.
* **⚡ Google Gemini 2.5 Flash:** High-speed response generation with a large context window.
* **💾 Local Database:** All data is stored locally in `ChromaDB`. No need to re-parse PDFs on every run.
* **🐳 Dockerized:** Ready to deploy anywhere with a single container.
* **🖥️ Streamlit UI:** User-friendly chat interface.

---

## 🛠️ Installation & Usage (Docker)

You can run this project easily using Docker without worrying about dependencies.

### 1. Build the Image
Open your terminal in the project directory:

```bash
docker build -t legal-assistant .