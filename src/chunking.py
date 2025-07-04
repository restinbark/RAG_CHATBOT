from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_texts(texts, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    all_chunks = []
    for text in texts:
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
    return all_chunks
