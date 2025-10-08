import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
import tempfile
import os
from openai import OpenAIError

st.set_page_config(page_title="RAG Demo - Latest Docs Only", layout="wide")

st.title("📄 RAG Demo with DOCX Upload (Latest Version)")

# 1️⃣ Ask user for OpenAI API key
api_key = st.text_input(
    "Enter your OpenAI API Key",
    type="password",
    placeholder="sk-...",
)
if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# 2️⃣ Upload .docx files
uploaded_files = st.file_uploader(
    "Upload one or more DOCX files (latest version will be used, files must contain '_v2_' in the name)",
    type=["docx"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload at least one DOCX file.")
    st.stop()

# 2a️⃣ Filter only _v2_ files
latest_files = [f for f in uploaded_files if "_v2_" in f.name]
if not latest_files:
    st.error("No files with '_v2_' found. Please upload at least one latest version file.")
    st.stop()

# 3️⃣ Load latest documents
docs = []
for uploaded_file in latest_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        loader = Docx2txtLoader(tmp_file.name)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = uploaded_file.name
        docs.extend(loaded_docs)
    os.unlink(tmp_file.name)  # delete temp file

# 4️⃣ Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# 5️⃣ Create embeddings and vectorstore
try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(split_docs, embeddings)
except OpenAIError as e:
    st.error(f"OpenAI API error: {e}")
    st.stop()

st.success("✅ Latest documents processed and vectorstore created successfully!")

# 6️⃣ Ask a query and retrieve **single best answer**
query = st.text_input("Ask a question about your latest documents:")
if query:
    try:
        results = vectordb.similarity_search(query, k=1)  # only top 1 chunk
        if results:
            doc = results[0]
            st.write(f"**Answer (latest info from {doc.metadata.get('source','unknown')}):**")
            st.write(doc.page_content)
        else:
            st.info("No relevant content found in the latest documents.")
    except Exception as e:
        st.error(f"Error retrieving results: {e}")
