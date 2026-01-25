import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_docs(documents) -> str:
    """Format a list of documents into a single string."""
    return "\n\n".join([doc.page_content for doc in documents])


def create_basic_retrieval_chain(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    prompt_template: ChatPromptTemplate,
):
    retrieval_step = RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    ).with_config({"run_name": "Retrieve and Format Documents"})

    generation_step = (prompt_template | llm | StrOutputParser()).with_config(
        {"run_name": "Generate Answer"}
    )

    return (retrieval_step | generation_step).with_config(
        {"run_name": "Basic Retrieval Chain"}
    )


def create_retrieval_chain_with_sources(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    prompt_template: ChatPromptTemplate,
):
    """Create a retrieval chain that returns sources."""
    retrieval_step = RunnablePassthrough.assign(
        documents=itemgetter("question") | retriever
    ).with_config({"run_name": "Retrieve Documents"})

    generation_step = RunnableParallel(
        answer=(
            RunnablePassthrough.assign(
                context=itemgetter("documents") | RunnableLambda(format_docs)
            )
            | prompt_template
            | llm
            | StrOutputParser()
        ),
        documents=itemgetter("documents"),
    ).with_config({"run_name": "Generate Answer with Sources"})

    return (retrieval_step | generation_step).with_config(
        {"run_name": "Retrieval Chain with Sources"}
    )


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder",
        )
    )
    vector_store = PineconeVectorStore(
        index_name="regras-futebol-index", embedding=embeddings
    )
    llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.2)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    prompt_template = ChatPromptTemplate.from_template(
        (
            "Utilize as seguintes informações para responder à pergunta no final. "
            "Se você não souber a resposta, diga que não sabe, não tente inventar uma resposta.\n\n"
            "{context}\n\n"
            "Pergunta: {question}\n"
            "Resposta Útil:"
        )
    )

    question = "Um jogador pode usar a mão para marcar um gol?"
    llm_chain = create_basic_retrieval_chain(
        llm=llm,
        retriever=retriever,
        prompt_template=prompt_template,
    )
    answer = llm_chain.invoke({"question": question})
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("\n" + "=" * 50 + "\n")
    llm_chain_with_sources = create_retrieval_chain_with_sources(
        llm=llm,
        retriever=retriever,
        prompt_template=prompt_template,
    )
    result = llm_chain_with_sources.invoke({"question": question})
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {result['answer']}")
    print("\nFontes:\n")
    for idx, doc in enumerate(result["documents"]):
        print(f"Trecho {idx + 1}:")
        print(f"Fonte: {doc.metadata.get('source', 'No source available')}")
        print(f"Página: {doc.metadata.get('page', 'No page available')}")
        print(f"Conteúdo: {doc.page_content}")
        print("-----")
        print()


if __name__ == "__main__":
    main()
