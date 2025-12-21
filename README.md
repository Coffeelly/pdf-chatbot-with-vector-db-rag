# Streamlit RAG Chatbot

A Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and engage in conversational Q&A with the content. This application utilizes a local vector database for data persistence, local embeddings for privacy and speed, and the Groq API for high-performance inference.

## Features

- **PDF Analysis:** Extracts text from PDF files using PyMuPDF (Fitz) for accurate handling of scientific layouts and spacing.
- **Conversational Memory:** Maintains context across multiple questions, allowing for follow-up inquiries.
- **Local Vector Storage:** Uses Qdrant to store vector embeddings locally on disk, ensuring data persists.
- **Hybrid AI Architecture:**
  - **Embeddings:** Runs locally using HuggingFace (`all-MiniLM-L6-v2`) on the CPU.
  - **Inference:** Uses Groq Cloud (Llama 3 model) for near-instant responses.

## Prerequisites

- Anaconda or Miniconda installed on your system.
- A Groq API Key (available from the Groq Console).

## Installation

Follow these steps to set up the project environment.

### 1. Clone the Repository

```
git clone https://github.com/Coffeelly/pdf-chatbot-with-vector-db-rag.git
cd pdf-chatbot-with-vector-db-rag
```

### 2. Install Dependencies

This project uses an environment.yml file to manage dependencies.

```
conda env create -f environment.yml
conda activate pdf-chatbot-with-vector-db-rag
```

### 3. Configure API Keys

Create a .env file in the root directory. You need API keys from Google AI Studio and Serper.dev.

```
GROQ_API_KEY="your_GROQ_api_key"
```

### 4. Run the Application

```
streamlit run app.py
```

## Project Structure

Ensure your project files are organized as follows:

```text
├── qdrant_db_local/       # Local vector database storage (generated automatically)
├── src/
│   ├── __init__.py
│   └── rag_engine.py      # Core RAG logic (PDF processing, embeddings, chain)
├── .env                   # Environment variables (Groq API Key)
├── environment.yml        # Conda environment configuration
└── app.py                 # Streamlit frontend application
```

## ⚠️ CRITICAL: Resetting the Database

The application uses a local file-based database (qdrant_db_local) to store PDF data. Due to file locking mechanisms, the "Reset" button in the UI cannot fully clear the database from your disk.

To clear the memory completely:

1. Stop the application in your terminal (press Ctrl + C).

2. Manually delete the qdrant_db_local folder located in the project root directory.

3. Restart the application

