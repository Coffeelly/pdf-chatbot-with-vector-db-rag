# PDF Chatbot using RAG and Vector Database

A Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and engage in conversational Q&A with the content. This application utilizes a local vector database for data persistence, local embeddings for privacy and speed, and the Groq API for high-performance inference.

## Key Features

- **Multi-Chat Sessions:** Manage multiple conversations simultaneously. Each chat session is isolated, meaning the AI knows context only from the PDF uploaded in that specific session.
- **Persistent Memory:** Chat history is saved in a local SQLite database (`chat_history.db`), so conversations are not lost when you refresh the browser.
- **Cloud Native Vector DB:** Uses **Qdrant Cloud** to store document embeddings securely and reliably (no more local file locking issues).
- **Transparent AI:** Features a "Debug Mode" that shows exactly which text chunks the AI read from the PDF before answering.
- **Anti-Hallucination:** Strict prompting ensures the AI answers _only_ based on the provided PDF context.
- **Data Management:** Easily delete specific chat sessions (clears both the vector data in Cloud and local chat history).

## Architecture

- **Frontend:** Streamlit
- **Embedding Model:** `all-MiniLM-L6-v2` (via HuggingFace)
- **LLM:** Llama 3 (via Groq API)
- **Vector Database:** Qdrant Cloud (Managed Cluster)
- **Chat Database:** SQLite (Local)

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
QDRANT="your_qdrant_cloud_api_key"
QDRANT_ENDPOINT="your_qdrant_cluster_url"
```

### 4. Run the Application

```
streamlit run app.py
```

## Project Structure

Ensure your project files are organized as follows:

```text
├── src/
│   ├── __init__.py
│   ├── rag_engine.py      # Core RAG logic (PDF processing, embeddings, chain)
│   ├── db.py              # SQLite database handling (Session & Message CRUD)
│   └── chat_history.db    # Local database file (auto-generated)
├── .env                   # Environment variables (Groq API Key)
├── environment.yml        # Conda environment configuration
└── app.py                 # Streamlit frontend application
```

## Demo Video

https://github.com/user-attachments/assets/f5081f0a-0aec-4ca8-b69a-c9f53ab7bfd0

Future Roadmap (To-Do)
[ ] Implement Streaming Response (Typewriter effect).

[ ] Upgrade to Cloud-based Embeddings (OpenAI/Voyage) for better performance on low-end hardware.

[ ] Add support for uploading multiple PDFs in a single session.

[ ] Add "Regenerate Answer" button.

