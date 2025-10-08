from docx import Document
import os

input_files = ["paynet_api_v1_old.docx", "paynet_api_v2_new.docx"]
output_folder = "rag_docs"
chunk_size = 500

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def docx_to_text(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip() != "":
            full_text.append(para.text.strip())
    return "\n".join(full_text)

def split_text(text, chunk_size):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks

for file in input_files:
    text = docx_to_text(file)
    chunks = split_text(text, chunk_size)
    base_name = os.path.splitext(os.path.basename(file))[0]
    for i, chunk in enumerate(chunks):
        chunk_file = os.path.join(output_folder, f"{base_name}_chunk_{i+1}.txt")
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk)

print("Conversion completed!")
