from operator import itemgetter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
    Runnable,
)
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest
from langchain.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()


def format_docs(documents) -> str:
    """Format a list of documents into a single string."""
    return "\n\n".join([doc.page_content for doc in documents])


def rerank_documents_logic(
    query: str, documents: list[Document], ranker: Ranker, top_k: int = 5
) -> list[Document]:
    if not documents:
        return []

    passages = [{"id": doc.id, "text": doc.page_content} for doc in documents]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    doc_map = {doc.id: doc for doc in documents}

    reranked_docs = []
    for result in results[:top_k]:
        doc_id = result["id"]
        doc = doc_map[doc_id]
        doc.metadata["_rerank_score"] = result["score"]
        reranked_docs.append(doc)

    return reranked_docs


def create_retrieval_chain_with_sources(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    ranker: Ranker,
    prompt_template: ChatPromptTemplate,
) -> Runnable:
    """
    Cria uma chain RAG que retorna tanto a resposta quanto as fontes reranqueadas.
    """

    # 1. Passo de Recuperação (Retrieval)
    # Adiciona a chave 'documents' ao estado atual
    retrieval_step = RunnablePassthrough.assign(
        documents=itemgetter("question") | retriever
    ).with_config({"run_name": "Retrieve Documents"})

    # 2. Passo de Reranking
    # Atualiza a chave 'documents' com a versão reordenada
    reranking_step = RunnablePassthrough.assign(
        documents=RunnableLambda(
            lambda x: rerank_documents_logic(
                query=x["question"], documents=x["documents"], ranker=ranker, top_k=5
            )
        )
    ).with_config({"run_name": "Rerank Documents"})

    # 3. Passo de Geração (Generation) com Fontes
    # Usa RunnableParallel para bifurcar: uma via gera a resposta, a outra preserva os docs
    generation_step = RunnableParallel(
        answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["documents"]))
            | prompt_template
            | llm
            | StrOutputParser()
        ),
        documents=itemgetter("documents"),
    ).with_config({"run_name": "Generate Answer with Sources"})

    # Composição Final da Chain
    final_chain = retrieval_step | reranking_step | generation_step

    return final_chain


def main():
    # Configurações iniciais
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder",
        )
    )
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./tmp")
    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME", "agente-rh-index"),
        embedding=embeddings,
    )
    llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "Você é um assistente útil e preciso de RH. Use os seguintes trechos de contexto para responder à pergunta no final. "
                    "Se você não souber a resposta baseada no contexto, diga apenas que não sabe. "
                    "Responda em português do Brasil de forma profissional."
                )
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Contexto:\n{context}\n\nPergunta: {question}\nResposta Útil:"),
        ]
    )

    # Exemplo de uso da chain
    question = "Quais comportamentos são considerados inadequados segundo o código de conduta da empresa?"
    llm_chain = create_retrieval_chain_with_sources(
        llm, retriever, ranker, prompt_template
    )
    result = llm_chain.invoke({"question": question, "chat_history": []})
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {result['answer']}")
    print("\nFontes:\n")
    for idx, doc in enumerate(result["documents"]):
        print(f"Trecho {idx + 1}:")
        print(f"Fonte: {doc.metadata.get('source', 'No source available')}")
        print(f"Página: {doc.metadata.get('page', 'No page available')}")
        print(f"Categoria: {doc.metadata.get('category', 'No category available')}")
        print(f"Conteúdo: {doc.page_content}")
        print("-----")
        print()


if __name__ == "__main__":
    main()
