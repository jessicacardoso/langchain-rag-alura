# 🦜💊 RAG Agêntico Farmacêutico — Análise bulas Paracetamol e Dipirona

Este repositório contém um agente de IA desenvolvido para responder perguntas relacionadas às bulas de medicamentos, especificamente Paracetamol e Dipirona. O agente é capaz de interpretar e fornecer respostas com base nas informações oficiais dos medicamentos.

## 👀 Visão Geral
Esse projeto contém uma implementação simples e completa de uma aplicação RAG (Retrieval-Augmented Generation) usando LangChain, Pinecone e LangSmith. O agente farmacêutico é projetado para auxiliar usuários na obtenção de informações sobre os medicamentos mencionados. O fluxo do agente é dividido em três etapas principais:
1. **Ingestão de Documentos**: Carrega, divide e armazena documentos em uma base de dados vetorial.
2. **Recuperação**: Recupera documentos relevantes a partir da consulta do usuário e de metadados.
3. **Geração**: Gera respostas baseadas nos documentos recuperados.


## 🚀 Começando
Para começar a usar o agente farmacêutico, siga as etapas abaixo:
1. Instale as dependências necessárias:
    ```bash
    uv sync
    ```
2. Configure suas credenciais de API no arquivo `.env`.
    ```bash
    GROQ_API_KEY=your_groq_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    LANGSMITH_API_KEY=your_langsmith_api_key
    LANGSMITH_PROJECT=agente-farmaceutico
    LANGSMITH_TRACING=true
    INDEX_NAME=agente-farmaceutico-index
    EMBEDDING_MODEL_NAME=PORTULAN/serafim-900m-portuguese-pt-sentence-encoder
    ```
3. Execute o script principal:
    ```bash
    python main.py
    ```

## 🛠️ Tecnologias Utilizadas
- [LangChain](https://www.langchain.com/): Biblioteca para construir aplicações de IA com foco em LLMs.
- [Pinecone](https://www.pinecone.io/): Serviço de banco de dados vetorial para armazenamento e recuperação eficiente de embeddings.
- [LangSmith](https://smith.langchain.com/): Plataforma para rastreamento e monitoramento de experimentos com LLMs.


