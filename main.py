from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import ToolMessage
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv(
        "EMBEDDING_MODEL_NAME",
        "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder",
    )
)
vector_store = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME", "agente-farmaceutico-index"),
    embedding=embeddings,
)
llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")
system_prompt = (
    "Você é um assistente especializado em informações farmacêuticas. "
    "Use as ferramentas disponíveis para responder às perguntas dos usuários com precisão e clareza. "
    "Sempre que possível, forneça referências às fontes de informação utilizadas. "
    "Se a informação não estiver disponível, informe que não pode responder à pergunta."
)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str, medicine: str) -> str:
    """Retrieve information to help answer a query.

    Args:
        query (str): The input query to retrieve context for.
        medicine (str): The medicine to filter the context by. E.g., "dipirona" or "paracetamol".
    """
    retrieved_docs = vector_store.similarity_search(
        query, k=4, filter={"medicine": medicine}
    )
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def main():
    question = "Quais são as contraindicações da dipirona?"
    agent = create_agent(llm, tools=[retrieve_context], system_prompt=system_prompt)
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = response["messages"][-1].content
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")
    print("\nFontes:\n")
    for idx, doc in enumerate(context_docs):
        print(f"--- Trecho {idx + 1} ---")
        print(f"Medicamento: {doc.metadata.get('medicine', 'N/A')}")
        print(f"Categoria: {doc.metadata.get('category', 'N/A')}")
        print(f"Documento: {doc.metadata.get('source', 'Documento desconhecido')}")
        print(f"Página: {doc.metadata.get('page', 'N/A')}")
        print("\nConteúdo do chunk:")
        print(doc.page_content)
        print("\n")


if __name__ == "__main__":
    main()
