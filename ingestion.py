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
    for doc in docs:
        doc.metadata["medicine"] = os.path.splitext(
            os.path.basename(doc.metadata["source"])
        )[0]
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
    if "identificação do medicamento" in lower_content or "composição" in lower_content:
        metadata["category"] = "identificação"
    elif (
        "indicação" in lower_content
        or "para que este medicamento é indicado" in lower_content
    ):
        metadata["category"] = "indicação"
    elif "como este medicamento funciona" in lower_content or "ação" in lower_content:
        metadata["category"] = "ação"
    elif (
        "contraindicação" in lower_content
        or "quando não devo usar" in lower_content
        or "contra-indicações" in lower_content
    ):
        metadata["category"] = "contraindicação"
    elif (
        "advertência" in lower_content
        or "precaução" in lower_content
        or "o que devo saber antes de usar" in lower_content
    ):
        metadata["category"] = "advertência"
    elif "interação" in lower_content or "interações medicamentosas" in lower_content:
        metadata["category"] = "interação"
    elif (
        "dose" in lower_content
        or "posologia" in lower_content
        or "como devo usar" in lower_content
    ):
        metadata["category"] = "posologia"
    elif "reações adversas" in lower_content or "quais os males" in lower_content:
        metadata["category"] = "reações adversas"
    elif (
        "onde, como e por quanto tempo posso guardar" in lower_content
        or "armazenar" in lower_content
    ):
        metadata["category"] = "armazenamento"
    elif (
        "quantidade maior do que a indicada" in lower_content
        or "superdosagem" in lower_content
    ):
        metadata["category"] = "superdosagem"
    else:
        metadata["category"] = "outros"
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
        pinecone_index_name=os.getenv("INDEX_NAME", "agente-farmaceutico-index"),
    )
    print("Vector store created successfully.")
