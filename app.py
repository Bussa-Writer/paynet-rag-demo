import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
import tempfile
import os
import re
import openai
from openai import OpenAIError

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("📄 RAG Demo - Auto Latest Version")

# 1️⃣ Ask user for OpenAI API key
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
    "Upload DOCX files (latest version will be automatically detected)",
    type=["docx"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload at least one DOCX file.")
    st.stop()

# 3️⃣ Detect latest version automatically
def extract_version(filename: str) -> int:
    """
    Extract version number from filename.
    Looks for patterns like v1, v2, v10, etc.
    Returns 0 if no version found.
    """
    match = re.search(r"v(\d+)", filename.lower())
    if match:
        return int(match.group(1))
    return 0

# Group files by base name (ignoring version) and pick highest version
latest_files = {}
for file in uploaded_files:
    base_name = re.sub(r"v\d+", "", file.name.lower())
    version = extract_version(file.name)
    if base_name not in latest_files or version > latest_files[base_name][1]:
        latest_files[base_name] = (file, version)

# Only use the latest version files
latest_docs_files = [f for f, v in latest_files.values()]
if not latest_docs_files:
    st.warning("No latest version documents found.")
    st.stop()

# 4️⃣ Load only latest documents
docs = []
for uploaded_file in latest_docs_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        loader = Docx2txtLoader(tmp_file.name)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = uploaded_file.name
        docs.extend(loaded_docs)
    os.unlink(tmp_file.name)

# 5️⃣ Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# 6️⃣ Create embeddings and vectorstore (in-memory)
try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory=None)
except OpenAIError as e:
    st.error(f"OpenAI API error: {e}")
    st.stop()

st.success("✅ Latest documents processed and vectorstore created!")

# 7️⃣ Ask query and return only top result
query = st.text_input("Ask a question about your latest documents:")
if query:
    try:
        results = vectordb.similarity_search(query, k=1)  # top 1 chunk
        if results:
            doc = results[0]
            st.write(f"**Answer (latest info from {doc.metadata.get('source')}):**")
            st.write(doc.page_content)
        else:
            st.info("No matching content found in latest documents.")
    except Exception as e:
        st.error(f"Error retrieving results: {e}")
