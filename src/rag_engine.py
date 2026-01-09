import streamlit as st
import fitz  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
VECTOR_DB_PATH = "./qdrant_db_local"
COLLECTION_NAME = "my_pdf_collection_local"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
CHAT_MODEL_NAME = "llama-3.1-8b-instant"

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(path=VECTOR_DB_PATH)

def load_and_split_pdf(uploaded_file):
    """
    Refactored to use PyMuPDF (fitz) which handles spacing 
    in scientific papers MUCH better than PyPDF2.
    """
    # 1. Read the file bytes
    bytes_data = uploaded_file.getvalue()
    
    # 2. Open with fitz
    doc = fitz.open(stream=bytes_data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    
    # 3. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return text, chunks

def setup_vector_store(chunks):
    client = get_qdrant_client()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    
    # Create fresh collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embedding=embeddings,
    )
    vector_store.add_texts(chunks)
    return vector_store

def get_rag_chain(vector_store):
    llm = ChatGroq(model=CHAT_MODEL_NAME, temperature=0)

    retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    
    # --- Contextualize Question ---
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
    
    # --- Answer Question ---
    system_prompt = (
        "You are a helpful assistant. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer, say that you don't know.\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

def generate_summary(text):
    llm = ChatGroq(model=CHAT_MODEL_NAME, temperature=0)
    prompt = f"Summarize this text. Format the output as a Markdown list with 3 concise bullet points.\n\n{text[:3000]}"
    return llm.invoke(prompt).content

def clear_database():
    # 1. Get the existing client (from the cache)
    client = get_qdrant_client()
    
    try:
        # 2. Delete the collection if it exists
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Error cleaning up: {e}")
        
    # 3. Explicitly close the connection to release the file lock
    client.close()
    

    return True
