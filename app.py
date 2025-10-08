import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
import tempfile
import os
from openai import OpenAIError

st.set_page_config(page_title="RAG Demo (Latest Only)", layout="wide")
st.title("📄 RAG Demo with DOCX Upload (v2 only)")

# ---------------------------
# 1️⃣ Ask user for OpenAI API key
# ---------------------------
api_key = st.text_input(
    "Enter your OpenAI API Key",
    type="password",
    placeholder="sk-...",
)
if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# ---------------------------
# 2️⃣ Upload .docx files
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload one or more DOCX files (v1, v2, etc.)",
    type=["docx"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload at least one DOCX file.")
    st.stop()

# ---------------------------
# 3️⃣ Load documents
# ---------------------------
docs = []
for uploaded_file in uploaded_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        loader = Docx2txtLoader(tmp_file.name)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source_file"] = uploaded_file.name  # track filename
        docs.extend(loaded_docs)
    os.unlink(tmp_file.name)

# ---------------------------
# 4️⃣ Split documents into chunks
# ---------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = []
for doc in docs:
    chunks = text_splitter.split_documents([doc])
    for chunk in chunks:
        # ensure metadata is preserved
        chunk.metadata["source_file"] = doc.metadata.get("source_file", "unknown")
        split_docs.append(chunk)

# ---------------------------
# 5️⃣ Create embeddings and vectorstore
# ---------------------------
try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(split_docs, embeddings)
except OpenAIError as e:
    st.error(f"OpenAI API error: {e}")
    st.stop()

st.success("✅ Documents processed and vectorstore created successfully!")

# ---------------------------
# 6️⃣ Ask a query and retrieve latest info only
# ---------------------------
query = st.text_input("Ask a question about your latest documents (v2):")
if query:
    try:
        # Get top 5 most similar chunks
        results = vectordb.similarity_search(query, k=5)
        
        # Filter to v2 documents only
        results_v2 = [r for r in results if "v2" in r.metadata.get("source_file", "").lower()]
        
        if results_v2:
            top_chunk = results_v2[0]  # top chunk from v2
            st.write(f"Answer (latest info from {top_chunk.metadata['source_file']}):")
            st.write(top_chunk.page_content)
        else:
            st.warning("No relevant info found in the latest documents (v2).")
    except Exception as e:
        st.error(f"Error retrieving results: {e}")
