import os

folders = [
    "data",
    "notebooks",
    "src",
    "vector_store",
    "app",
]

files = {
    "README.md": "# CrediTrust Complaint-Answering Chatbot using RAG\n",
    "requirements.txt": "",
    ".gitignore": "*.pyc\n__pycache__/\n.env\nvector_store/\n"
}

sub_files = {
    "notebooks/eda_preprocessing.ipynb": "",
    "src/chunking.py": "",
    "src/embed_index.py": "",
    "src/rag_pipeline.py": "",
    "src/evaluation.py": "",
    "app/app.py": "" 
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

# Create root files
for filename, content in files.items():
    with open(filename, "w") as f:
        f.write(content)
    print(f"Created file: {filename}")

# Create stub files in subfolders
for filepath, content in sub_files.items():
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Created file: {filepath}")
