import streamlit as st
import os
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from src.rag_engine import (
    load_and_split_pdf, 
    setup_vector_store, 
    get_rag_chain, 
    generate_summary,
    clear_database
)

st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
st.header("🤖 Understand Your PDF")

# Check API Key
if not os.getenv("GROQ_API_KEY"):
    st.error("Groq API Key not found. Please set it in the .env file.")
    st.stop()

# --- INIT SESSION STATE ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())

with st.sidebar:
    st.title("Settings")
    
    # --- UPDATED: RESET BUTTON ---
    if st.button("🗑️ Reset", type="primary"):
        try:
            # 1. Clear the actual database on disk
            clear_database()
        except Exception as e:
            st.warning(f"Note: Database file might be locked. {e}")
            
        # 2. Clear Python's cached connection
        st.cache_resource.clear()
        
        # 3. Clear the UI memory (Chat history, summary, etc)
        st.session_state.clear()
        
        # 4. Generate a NEW Key. This forces the file uploader to "re-render" as empty.
        st.session_state.uploader_key = str(uuid.uuid4())
        
        # 5. Rerun
        st.rerun()

    st.divider()

    st.header("Upload File")
    
    # ensure the key exists again because cleared the session above
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = str(uuid.uuid4())

    # --- FILE UPLOADER ---
    # added 'key=...'. When the key changes, this widget is destroyed and recreated.
    uploaded_file = st.file_uploader(
        "Choose a PDF", 
        type="pdf", 
        key=st.session_state.uploader_key
    )
    
    if uploaded_file:
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                raw_text, chunks = load_and_split_pdf(uploaded_file)
                vector_store = setup_vector_store(chunks)
                
                st.session_state.vector_store = vector_store
                st.session_state.summary = generate_summary(raw_text)
                st.session_state.last_uploaded = uploaded_file.name
                st.session_state.messages = [] 
                
            st.success("Done!")

if "summary" in st.session_state:
    with st.expander("Document Summary", expanded=True):
        st.write(st.session_state.summary)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the PDF..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if "vector_store" in st.session_state:
            rag_chain = get_rag_chain(st.session_state.vector_store)
            
            # 1. Convert session state to LangChain history format
            #    exclude the very last message (the new one) because the chain adds it automatically
            chat_history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

            with st.spinner("Thinking..."):
                # 2. Pass 'chat_history' to the chain
                response = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history 
                })
                answer = response["answer"]
                source_docs = response["context"]
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("Debug: What did AI read?"):
                    for i, doc in enumerate(source_docs):
                        st.caption(f"**Chunk {i+1}:** {doc.page_content[:200]}...")

            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Please upload a PDF to start.")
