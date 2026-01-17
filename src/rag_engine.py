import streamlit as st
import fitz  
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
COLLECTION_NAME = "global_pdf_storage" 
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
CHAT_MODEL_NAME = "llama-3.1-8b-instant"

@st.cache_resource
def get_qdrant_client():
    url = os.getenv("QDRANT_ENDPOINT")
    api_key = os.getenv("QDRANT")
    
    if url and api_key:
        try:
            client = QdrantClient(url=url, api_key=api_key)
            return client
        except Exception as e:
            print(f"Fail Connect To Cloud: {e}")
    return QdrantClient(path="./qdrant_db_local")

def load_and_split_pdf(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    doc = fitz.open(stream=bytes_data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return text, chunks

def setup_vector_store(chunks, session_id):
    client = get_qdrant_client()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    session_id = int(session_id) 
    
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        print("⏳ Create Payload Index...")
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.session_id", 
            field_schema=models.PayloadSchemaType.INTEGER
        )
        
    metadatas = [{"session_id": session_id} for _ in chunks]

    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embedding=embeddings,
    )
    vector_store.add_texts(texts=chunks, metadatas=metadatas)
    return vector_store

def get_rag_chain(client, session_id):
    llm = ChatGroq(model=CHAT_MODEL_NAME, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    session_id = int(session_id)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    search_kwargs = {
        "k": 5,
        "filter": models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.session_id", 
                    match=models.MatchValue(value=session_id)
                )
            ]
        )
    }

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    
    # --- PROMPT (Tetap sama) ---
    system_prompt = (
        "You are a specialized assistant for analyzing PDF documents. "
        "You must answer the user's question based ONLY on the provided context below. "
        "Do not use your outside knowledge or general information. "
        "If the answer is not found in the context, say 'Sorry, there are no such information in the document.' "
        "Keep your answer concise and relevant.\n\n"
        "Context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

def generate_summary(text):
    llm = ChatGroq(model=CHAT_MODEL_NAME, temperature=0)
    prompt = f"Summarize this text concisely in 3 bullet points:\n\n{text[:3000]}"
    return llm.invoke(prompt).content

def delete_session_data(session_id):
    """Delete all vectors belonging to a specific session_id"""
    client = get_qdrant_client()
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.session_id",
                            match=models.MatchValue(value=int(session_id))
                        )
                    ]
                )
            )
        )
        print(f"✅ Vector {session_id} succesfully deleted from qdrant.")
    except Exception as e:
        print(f"⚠️ Failed to delete vector: {e}")