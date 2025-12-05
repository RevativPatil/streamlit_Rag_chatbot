import streamlit as st
import os
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import google.generativeai as genai

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="AI Document Chatbot ",
    page_icon="https://cdn-icons-png.flaticon.com/512/4712/4712102.png",
    layout="wide"
)

# Gemini API setup
genai.configure(api_key="jaosygjsvbmnxbna782ganiosdnm")  # replace with your Gemini API key

# Embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# CHROMADB SETUP
# -----------------------------
client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection = client_chroma.get_or_create_collection(name="docs_embeddings")

# -----------------------------
# CHAT HISTORY
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# STYLING (unchanged)
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #0f0f0f 0%, #000000 80%);
    color: white;
}
.block-container {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 3rem;
    margin-top: 2rem;
    box-shadow: 0 0 30px rgba(255,255,255,0.1);
}
h1, h2, h3 { color: #ffffff !important; font-weight: 600; }
p, label { color: #b0b0b0 !important; }
.stButton>button {
    background: linear-gradient(90deg, #00ffe0 0%, #0077ff 100%);
    color: #000; font-weight: 600; border: none; border-radius: 12px;
    padding: 0.8rem 2.2rem; font-size: 1.1rem; transition: all 0.3s ease;
}
.stButton>button:hover { transform: scale(1.07); box-shadow: 0 0 20px #00bfff; }
div[data-testid="stFileUploader"] button {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: 1px solid #444 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stFileUploader"] button:hover {
    background-color: #111111 !important;
    box-shadow: 0 0 20px rgba(255,255,255,0.3) !important;
}
[data-testid="stFileUploaderFileName"] { color: white !important; }
.chat-bubble-user {
    background-color: #ffffff;
    color: #000000;
    padding: 10px;
    border-radius: 10px;
    margin-top: 5px;
    border: 1px solid #888;
}
.chat-bubble-bot {
    background: linear-gradient(90deg, #ff4ec4 0%, #9d00ff 100%);
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin-top: 5px;
}
header {
    text-align: center; margin-bottom: 30px; background: rgba(20, 20, 20, 0.7);
    backdrop-filter: blur(10px); border-radius: 10px; padding: 15px;
}
header p { color: #c0c0c0; font-size: 18px; }
</style>
<header>
    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712102.png" width="100">
    <h1> AI Document Chatbot</h1>
    <p>Upload • Summarize • Chat — with a human touch</p>
</header>
""", unsafe_allow_html=True)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_files = st.file_uploader(" Upload one or more documents",
                                  type=["pdf", "docx", "txt"], accept_multiple_files=True)

def extract_text(file):
    """Extract text safely."""
    text = ""
    try:
        if file.name.endswith(".pdf"):
            try:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            except Exception:
                file.seek(0)
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
        elif file.name.endswith(".docx"):
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            text = file.read().decode("utf-8")
    except Exception as e:
        st.warning(f" Couldn't read {file.name}. Reason: {e}")
    return text

# -----------------------------
# PROCESS & SUMMARIZE
# -----------------------------
if uploaded_files:
    model = genai.GenerativeModel("gemini-2.5-flash")
    for uploaded_file in uploaded_files:
        with st.spinner(f" Processing {uploaded_file.name}..."):
            text_data = extract_text(uploaded_file)
            if not text_data.strip():
                st.warning(f" No readable text in {uploaded_file.name}.")
                continue

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_text(text_data)
            embeddings = embed_model.encode(chunks, convert_to_numpy=True).tolist()

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                collection.add(
                    ids=[f"{uploaded_file.name}_{i}"],
                    documents=[chunk],
                    embeddings=[emb]
                )

            prompt = f"Summarize this document briefly and naturally:\n\n{text_data[:4000]}"
            try:
                response = model.generate_content(prompt)
                summary = response.text.strip()
            except Exception as e:
                summary = f" Error generating summary: {e}"

            st.subheader(f" Summary for {uploaded_file.name}:")
            st.markdown(f"<div class='chat-bubble-bot'>{summary}</div>", unsafe_allow_html=True)

    st.success(" All files processed and stored!")

# -----------------------------
# CHATBOT SECTION
# -----------------------------
st.subheader(" Ask Anything About Your Documents")
query = st.text_input("Type your question...")

col1, col2 = st.columns([1, 0.3])
with col2:
    if st.button(" Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
    if st.button(" Clear Document Memory"):
        ids = collection.get()["ids"]
        if ids:
            collection.delete(ids=ids)
            st.success(" Document memory cleared!")

# -----------------------------
# HUMAN-LIKE ANSWERING
# -----------------------------
if query:
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    if results["documents"]:
        context = "\n\n".join(results["documents"][0])
    else:
        context = ""

    if not context.strip():
        answer = "Sorry, I couldn’t find that in your document."
    else:
        prompt = f"""
You are a helpful assistant. Use only the context below to answer **like a human** — short, direct, and natural.  
If the context doesn’t have the answer, say exactly: "Sorry, I couldn’t find that in your document."

Context:
{context}

Question: {query}
"""
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            answer = f" Error from Gemini: {str(e)}"

    # Save chat
    st.session_state.chat_history.append(("User", query))
    st.session_state.chat_history.append(("Bot", answer))

    # Display messages
    st.markdown(f"<div class='chat-bubble-user'><b> You:</b> {query}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble-bot'><b> Bot:</b> {answer}</div>", unsafe_allow_html=True)

# -----------------------------
# CHAT HISTORY
# -----------------------------
if st.session_state.chat_history:
    st.subheader(" Chat History")
    for role, msg in st.session_state.chat_history:
        bubble = 'chat-bubble-user' if role == "User" else 'chat-bubble-bot'
        icon = "" if role == "User" else ""
        st.markdown(f"<div class='{bubble}'><b>{icon} {role}:</b> {msg}</div>", unsafe_allow_html=True)
