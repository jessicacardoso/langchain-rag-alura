import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def load_pdf(file_path):
    """
    Load a PDF file and return its pages as documents.

    Args:
        file_path (str): The path to the PDF file.
    Returns:
        List[Document]: A list of documents representing the pages of the PDF.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def split_text(documents, chunk_size=1000, chunk_overlap=0):
    """
    Split documents into smaller chunks.

    Args:
        documents (List[Document]): The list of documents to split.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.
    Returns:
        List[Document]: A list of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents


def create_vector_store(
    documents,
    embedding_model_name="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder",
    pinecone_index_name="my-pinecone-index",
):
    """
    Create a Pinecone vector store from documents.

    Args:
        documents (List[Document]): The list of documents to index.
        embedding_model_name (str): The name of the HuggingFace embedding model.
        pinecone_index_name (str): The name of the Pinecone index.
    Returns:
        PineconeVectorStore: The created Pinecone vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    query_result = embeddings.embed_query("This is a test document.")
    print(f"Sample embedding vector (first 5 values): {query_result[:5]}")
    print(f"Embedding vector length: {len(query_result)}")
    vector_store = PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name=pinecone_index_name,
    )
    return vector_store


if __name__ == "__main__":
    pdf_path = "regras_futebol.pdf"
    docs = load_pdf(pdf_path)
    print(f"Loaded {len(docs)} pages from the PDF.")
    chunks = split_text(docs, chunk_size=500, chunk_overlap=100)
    print(f"Splitted into {len(chunks)} chunks.")
    print("Creating vector store...")
    vector_store = create_vector_store(
        chunks,
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder",
        ),
        pinecone_index_name="regras-futebol-index",
    )
    print("Vector store created successfully.")
