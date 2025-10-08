import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
import tempfile
import os
import re
from openai import OpenAIError

st.set_page_config(page_title="RAG Demo - Latest Version Only", layout="wide")
st.title("📄 RAG Demo - Latest Version Only")

# 1️⃣ Ask for OpenAI API key
api_key = st.text_input(
    "Enter your OpenAI API Key",
    type="password",
    placeholder="sk-...",
)
if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# 2️⃣ Upload DOCX files
uploaded_files = st.file_uploader(
    "Upload DOCX files (v1, v2, etc.)",
    type=["docx"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload at least one DOCX file.")
    st.stop()

# 3️⃣ Extract version from filename
def extract_version(filename: str) -> int:
    match = re.search(r"v(\d+)", filename.lower())
    return int(match.group(1)) if match else 0

# 4️⃣ Keep only latest version per base name
latest_files = {}
for file in uploaded_files:
    base_name = re.sub(r"v\d+", "", file.name.lower())
    version = extract_version(file.name)
    if base_name not in latest_files or version > latest_files[base_name][1]:
        latest_files[base_name] = (file, version)

latest_docs_files = [f for f, v in latest_files.values()]

# 5️⃣ Load latest documents only and attach metadata
docs = []
for uploaded_file in latest_docs_files:
    version = extract_version(uploaded_file.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        loader = Docx2txtLoader(tmp_file.name)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["version"] = version
        docs.extend(loaded_docs)
    os.unlink(tmp_file.name)

# 6️⃣ Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# 7️⃣ Create embeddings with OpenAI
try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(split_docs, embeddings)
except OpenAIError as e:
    st.error(f"OpenAI API error: {e}")
    st.stop()

st.success("✅ Latest version documents processed!")

# 8️⃣ Query latest chunks only
query = st.text_input("Ask a question about your latest documents:")
if query:
    try:
        # Search top 5 chunks
        results = vectordb.similarity_search(query, k=5)

        if results:
            # Only keep highest version chunks
            max_version = max(doc.metadata.get("version", 0) for doc in results)
            latest_results = [doc for doc in results if doc.metadata.get("version") == max_version]

            # Return **top 1 chunk** from latest version only
            doc = latest_results[0]
            st.write(f"**Answer (latest info from {doc.metadata.get('source')}):**")
            st.write(doc.page_content)
        else:
            st.info("No matching content found in latest documents.")
    except Exception as e:
        st.error(f"Error retrieving results: {e}")
