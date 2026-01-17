import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from src.rag_engine import (
    load_and_split_pdf, 
    setup_vector_store, 
    get_rag_chain, 
    generate_summary,
    get_qdrant_client,
    delete_session_data
)
import src.db as db

st.set_page_config(page_title="PDF Multi-Chat", page_icon="📚", layout="wide")

# --- 1. SETUP DATABASE ---
db.init_db()

# --- 2. SESSION STATE MANAGEMENT ---
# Default state is None (no chat selected)
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# switch to old chat session
def switch_session(session_id):
    st.session_state.current_session_id = session_id

# Prepare the interface for a new chat (but don’t create the database yet)
def init_new_chat_view():
    st.session_state.current_session_id = None

# --- 3. SIDEBAR (History Chat) ---
with st.sidebar:
    st.title(" Chat History")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        init_new_chat_view()
        st.rerun()
    
    # --- DELETE ACTIVE CHAT ---
    if st.session_state.current_session_id is not None:
        if st.button("🗑️ Delete Current Chat", use_container_width=True):
            # 1. Delete Vector in Qdrant
            with st.spinner("Deleting memories..."):
                delete_session_data(st.session_state.current_session_id)
            
            # 2. Delete data in SQLite
            db.delete_session(st.session_state.current_session_id)
            
            # 3. Reset State
            st.session_state.current_session_id = None
            st.success("Chat deleted!")
            st.rerun()

    st.divider()
    
    # List Chat from database
    sessions = db.get_all_sessions()
    if not sessions:
        st.caption("No chat history.")
        
    for sess in sessions:
        # button to change chat
        if st.button(f" {sess['title']}", key=f"sess_{sess['id']}", use_container_width=True):
            switch_session(sess['id'])
            st.rerun()

# --- 4. MAIN CONTENT LOGIC ---
current_session_id = st.session_state.current_session_id

# === SCENARIO A: NO SESSION (Upload/Draft) ===
if current_session_id is None:
    st.header("✨ Start New Chat")
    st.info("Please upload PDF to start a new chat.")
    
    # Widget Upload
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    
    if uploaded_file:
        if st.button(" Process & Start Chat", type="primary"):
            with st.spinner("Analyzing Document..."):
                # 1. CREATE NEW SESSION ON DATABASE
                # Session only created when user process the pdf
                new_id = db.create_session(title=uploaded_file.name)
                
                # 2. process PDF
                raw_text, chunks = load_and_split_pdf(uploaded_file)
                
                # 3. store to Qdrant with the new created session id
                setup_vector_store(chunks, new_id)
                
                # 4. Generate Summary 
                summary = generate_summary(raw_text)
                db.add_message(new_id, "assistant", f"**Document Ready!**\n\nSummary:\n{summary}")
                
                # 5. Update State & Rerun
                st.session_state.current_session_id = new_id
                st.rerun() 

# === SKENARIO B: SESSION IS ACTIVE (Chatting Mode) ===
else:
    # FETCH SESSION DATA
    
    # 1. Fetch Title From Database
    session_title = db.get_session_title(current_session_id)
    
    # 2. Display Title
    st.header(f" {session_title}")
        
    # --- DISPLAY MESSAGE ---
    chat_container = st.container()
    stored_messages = db.get_messages(current_session_id)
    
    with chat_container:
        if not stored_messages:
            st.info("Empty chat.")
        
        for msg in stored_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # --- INPUT USER & RAG PROCESS ---
    if prompt := st.chat_input("Ask about your PDF..."):
        # 1. Display & Save User Chat
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        db.add_message(current_session_id, "user", prompt)

        # 2. Generate Answer (WITH DEBUGGING VISUAL)
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        client = get_qdrant_client()
                        
                        # Call Chain
                        rag_chain = get_rag_chain(client, current_session_id)
                        
                        # history
                        chat_history_lc = []
                        for msg in stored_messages:
                            if msg["role"] == "user":
                                chat_history_lc.append(HumanMessage(content=msg["content"]))
                            elif msg["role"] == "assistant":
                                chat_history_lc.append(AIMessage(content=msg["content"]))

                        # Execute
                        response = rag_chain.invoke({
                            "input": prompt,
                            "chat_history": chat_history_lc
                        })
                        
                        answer = response["answer"]
                        retrieved_docs = response["context"] # Dokumen yang ditemukan
                        
                        # --- DISPLAY MAIN ANSWER ---
                        st.markdown(answer)
                        
                        # display debugging
                        with st.expander(" Debug: What LLM reads?"):
                            if not retrieved_docs:
                                st.error(" The LLM did not find any relevant information in the PDF.")
                            else:
                                st.caption(f"Found {len(retrieved_docs)} relevant text excerpts:")
                                for i, doc in enumerate(retrieved_docs):
                                    st.markdown(f"** excerpts #{i+1}**")
                                    st.code(doc.page_content, language="text") 
                                    st.divider()
                        
                        # 3. Save LLM Answer To Database
                        db.add_message(current_session_id, "assistant", answer)

                    except Exception as e:
                        st.error(f"Error: {e}")
                        print(f" ERROR DETAIL: {e}")