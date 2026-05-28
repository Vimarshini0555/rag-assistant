import streamlit as st
import requests
import uuid
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Setup Streamlit page
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="✨",
    layout="wide"
)

# Premium Glassmorphism CSS
st.markdown("""
<style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #151522 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphic Container */
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Citations Block */
    .citation-box {
        background: rgba(100, 150, 255, 0.1);
        border-left: 4px solid #6495ED;
        padding: 10px 15px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.9em;
        color: #a0aec0;
    }
    
    .citation-title {
        font-weight: bold;
        color: #e2e8f0;
        margin-bottom: 5px;
    }
    
    /* Titles */
    h1, h2, h3 {
        color: #e2e8f0;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Document Upload
with st.sidebar:
    st.markdown("### 📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF to query", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document into vectors..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                try:
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Success! Processed {data['chunks_processed']} chunks.")
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

# Main Chat Interface
st.markdown("<h1>✨ Production RAG AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #a0aec0;'>Upload documents in the sidebar and start asking questions. Answers will include citations.</p>", unsafe_allow_html=True)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Display citations if any
        if msg.get("citations"):
            for idx, cit in enumerate(msg["citations"]):
                st.markdown(f"""
                <div class="citation-box">
                    <div class="citation-title">[{idx+1}] Source: {cit['source']} (Page {cit.get('page', '?')})</div>
                    <div>{cit['content']}</div>
                </div>
                """, unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask about your document..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
    # Call Backend API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "session_id": st.session_state.session_id,
                    "message": prompt
                }
                res = requests.post(f"{API_URL}/chat", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    citations = data.get("citations", [])
                    
                    st.write(answer)
                    if citations:
                        for idx, cit in enumerate(citations):
                            st.markdown(f"""
                            <div class="citation-box">
                                <div class="citation-title">[{idx+1}] Source: {cit['source']} (Page {cit.get('page', '?')})</div>
                                <div>{cit['content']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "citations": citations
                    })
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
