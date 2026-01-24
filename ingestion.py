from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()


def load_pdfs(file_path):
    """
    Load PDF documents from the specified file path.
    Args:
        file_path (str): The path to the PDF files.
    Returns:
        List[Document]: A list of loaded documents.
    """
    loader = GenericLoader(
        blob_loader=FileSystemBlobLoader(
            file_path,
            glob="*.pdf",
        ),
        blob_parser=PyPDFParser(),
    )
    docs = loader.load()
    return docs


def create_metadata(content):
    """
    Create metadata for a document chunk based on its content.

    Args:
        content (str): The content of the document chunk.
    Returns:
        dict: A dictionary containing the metadata.
    """
    metadata = {}
    lower_content = content.lower()
    if "férias" in lower_content:
        metadata["categoria"] = "ferias"
    elif "home office" in lower_content or "remoto" in lower_content:
        metadata["categoria"] = "home_office"
    elif "conduta" in lower_content or "ética" in lower_content:
        metadata["categoria"] = "conduta"
    else:
        metadata["categoria"] = "geral"
    return metadata


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

    for chunk in chunked_documents:
        content = chunk.page_content.lower()
        metadata = create_metadata(content)
        chunk.metadata.update(metadata)
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
    pdf_path = "./data/"
    docs = load_pdfs(pdf_path)
    print(f"Loaded {len(docs)} pages from the PDF.")
    chunks = split_text(docs, chunk_size=600, chunk_overlap=150)
    print(f"Splitted into {len(chunks)} chunks.")
    print("Creating vector store...")
    vector_store = create_vector_store(
        chunks,
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder",
        ),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "agente-rh-index"),
    )
    print("Vector store created successfully.")
