# 🦜📜 Agente de RH — Políticas de Empresa

Este repositório contém um agente de IA desenvolvido para responder perguntas relacionadas às políticas de recursos humanos da empresa. O agente é capaz de interpretar e fornecer respostas com base nas informações oficiais das políticas internas.

## 👀 Visão Geral
Esse projeto contém uma implementação simples e completa de uma aplicação RAG (Retrieval-Augmented Generation) usando LangChain, Pinecone e LangSmith. O agente de RH é projetado para auxiliar usuários na obtenção de informações sobre as políticas da empresa. Ele utiliza um modelo de linguagem para gerar respostas baseadas em documentos relevantes recuperados de um banco de dados vetorial.

## 🚀 Começando
Para começar a usar o agente de RH, siga as etapas abaixo:
1. Instale as dependências necessárias:
    ```bash
    uv sync
    ```
2. Configure suas credenciais de API no arquivo `.env`.
    ```bash
    GROQ_API_KEY=your_groq_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    LANGSMITH_API_KEY=your_langsmith_api_key
    LANGSMITH_PROJECT=agente-rh
    LANGSMITH_TRACING=true
    INDEX_NAME=agente-rh-index
    EMBEDDING_MODEL_NAME=PORTULAN/serafim-900m-portuguese-pt-sentence-encoder
    ```
3. Execute o script principal:
    ```bash
    streamlit run main.py
    ```

## 🛠️ Tecnologias Utilizadas
- [LangChain](https://www.langchain.com/): Biblioteca para construir aplicações de IA com foco em LLMs.
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank): Biblioteca leve para re-ranking em pipelines de busca e recuperação.
- [Pinecone](https://www.pinecone.io/): Serviço de banco de dados vetorial para armazenamento e recuperação eficiente de embeddings.
- [LangSmith](https://smith.langchain.com/): Plataforma para rastreamento e monitoramento de experimentos com LLMs.
- [Streamlit](https://streamlit.io/): Framework para construção de aplicações web interativas em Python.


