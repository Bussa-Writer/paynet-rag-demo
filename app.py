import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
import tempfile
import os
import re
from openai import OpenAIError

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("📄 RAG Demo - Latest Version Only")

# -------------------------
# 1️⃣ Ask user for OpenAI API key
# -------------------------
api_key = st.text_input(
    "Enter your OpenAI API Key",
    type="password",
    placeholder="sk-...",
)
if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# -------------------------
# 2️⃣ Upload DOCX files
# -------------------------
uploaded_files = st.file_uploader(
    "Upload one or more DOCX files (v1, v2 etc.)",
    type=["docx"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload at least one DOCX file.")
    st.stop()

# -------------------------
# Helper function to extract version number from filename
# -------------------------
def extract_version(filename):
    match = re.search(r'v(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

# -------------------------
# 3️⃣ Pick the latest version file only
# -------------------------
latest_file = max(uploaded_files, key=lambda f: extract_version(f.name))
st.info(f"Processing latest document: **{latest_file.name}**")

# -------------------------
# 4️⃣ Load latest document
# -------------------------
docs = []
with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
    tmp_file.write(latest_file.read())
    loader = Docx2txtLoader(tmp_file.name)
    loaded_docs = loader.load()
    for doc in loaded_docs:
        doc.metadata["source"] = latest_file.name
    docs.extend(loaded_docs)
os.unlink(tmp_file.name)

# -------------------------
# 5️⃣ Split document into chunks
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
split_docs = text_splitter.split_documents(docs)

# -------------------------
# 6️⃣ Create embeddings and vectorstore
# -------------------------
try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(split_docs, embeddings)
except OpenAIError as e:
    st.error(f"OpenAI API error: {e}")
    st.stop()

st.success("✅ Latest document processed and vectorstore created successfully!")

# -------------------------
# 7️⃣ Ask a query and retrieve answer
# -------------------------
query = st.text_input("Ask a question about your latest document:")
if query:
    try:
        # Get the top 1 matching chunk
        results = vectordb.similarity_search(query, k=1)
        if results:
            doc = results[0]
            st.write(f"**Answer (from {doc.metadata['source']}):**")
            st.write(doc.page_content)
        else:
            st.warning("No matching content found in the latest document.")
    except Exception as e:
        st.error(f"Error retrieving results: {e}")
