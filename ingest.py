import os
import pdfplumber
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings

pdf_file = "./documents"
batch_size = 25

def read_pdf(file_path):
    """Extract text from a PDF file."""
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def load_documents_from_directory(directory_path):
    """Load PDF documents from a specified directory."""
    files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".pdf")]
    documents = [Document(page_content=read_pdf(file)) for file in files]
    return documents

def load_all_documents():
    """Load all documents from the directory."""
    pdf_documents = load_documents_from_directory(pdf_file)
    return pdf_documents

def ingest_into_vector_store(documents, db):
    """Ingest processed text into the Chroma vector store."""
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=5000, chunk_overlap=500, separator=".")
    doc_splits = text_splitter.split_documents(documents)
    
    # Add documents to Chroma and persist the data
    db.add_documents(doc_splits)

    db.persist()

    print("Data has been ingested into vector database.")

def initialize_vector_store():
    """Initialize the Chroma vector store for retrieval."""
    db = Chroma(
        persist_directory="./TP_db",
        embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
        collection_name="rag-chroma"
    )
    return db

def main():
    all_documents = load_all_documents()
    if all_documents:
        db = initialize_vector_store()
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i+batch_size]
            ingest_into_vector_store(batch, db)
    else:
        print("No data to process.")

main()