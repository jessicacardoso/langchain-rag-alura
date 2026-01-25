import streamlit as st
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from flashrank import Ranker
from chains import create_retrieval_chain_with_sources

st.set_page_config(
    page_title="Agente de RH com RAG + Reranker",
    page_icon="🔎",
    layout="wide",
)
load_dotenv()


@st.cache_resource
def load_resources():
    """Inicializa embeddings, ranker e conexões de banco de dados."""

    if not os.getenv("GROQ_API_KEY"):
        st.error("ERRO: GROQ_API_KEY não encontrada no .env")
        st.stop()
    if not os.getenv("PINECONE_API_KEY"):
        st.error("ERRO: PINECONE_API_KEY não encontrada no .env")
        st.stop()

    with st.spinner("Carregando modelos de IA e conectando ao Pinecone..."):
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

        llm = ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.1,
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )

        return llm, retriever, ranker


def show_sources(docs):
    """Exibe as fontes consultadas na interface Streamlit."""
    container_sources = st.container()
    with container_sources:
        st.markdown("---")
        st.subheader("📚 Fontes Consultadas")
        for i, doc in enumerate(docs):
            score = doc.metadata.get("_rerank_score", 0)
            source_name = doc.metadata.get("source", "Desconhecida")
            page = doc.metadata.get("page", "N/A")

            with st.expander(
                f"#{i + 1} - {source_name} (Pág. {page}) - Score: {score:.4f}"
            ):
                st.markdown(f"**Conteúdo:**")
                st.info(doc.page_content)
                st.markdown("**Metadados completos:**")
                st.json(
                    {
                        "Categoria": doc.metadata.get("categoria", "N/A"),
                        "Documento": doc.metadata.get("source", "Desconhecida"),
                        "Data de Criação": doc.metadata.get("creationdate", "N/A"),
                    }
                )


def main():
    st.title("🔎 Buscador Inteligente RH — Políticas Internas")
    st.caption("Powered by LangChain, Pinecone, FlashRank & Groq")

    try:
        llm, retriever, ranker = load_resources()
    except Exception as e:
        st.error(f"Falha ao carregar recursos: {e}")
        return

    # Prompt Template
    prompt_template = ChatPromptTemplate.from_template(
        """Você é um assistente útil e preciso de RH. Use os seguintes trechos de contexto para responder à pergunta no final.
        Se você não souber a resposta baseada no contexto, diga apenas que não sabe, não tente inventar uma resposta.
        Responda em português do Brasil de forma clara e profissional.

        Contexto:
        {context}

        Pergunta: {question}
        Resposta Útil:"""
    )

    # Inicializar histórico de chat na sessão se não existir
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Olá! Sou seu assistente de RH. Como posso ajudar você hoje?",
                "sources": [],
            }
        ]

    # Exibir mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message["sources"]:
                show_sources(message["sources"])

    if question := st.chat_input("Digite sua dúvida sobre normas ou processos..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            container_resposta = st.empty()

            with st.spinner("Pesquisando documentos e analisando relevância..."):
                try:
                    chain = create_retrieval_chain_with_sources(
                        llm, retriever, ranker, prompt_template
                    )
                    result = chain.invoke({"question": question})
                    resposta_texto = result["answer"]
                    docs = result["documents"]

                    container_resposta.markdown(resposta_texto)

                    if docs:
                        show_sources(docs)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": resposta_texto,
                            "sources": docs,
                        }
                    )

                except Exception as e:
                    st.error(f"Ocorreu um erro ao processar sua solicitação: {str(e)}")


if __name__ == "__main__":
    main()
