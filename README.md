# 🦜⛓ Curso Langchain — Agentes com RAG 🗂️

Este repositório contém materiais e exemplos do curso "Arquiteturas RAG com LLMs: embeddings, busca semântica e criação de agentes com LangChain", oferecido pela Alura. O curso aborda a construção de agentes utilizando a biblioteca Langchain e a técnica de Recuperação-Baseada em Geração (RAG).


## 📚 Conteúdo do Curso

Neste repositório, construímos um conjunto de três exemplos práticos de agentes utilizando Langchain e RAG:
1. **[Agente de Futebol — Regras de Futebol (FIFA)](https://github.com/jessicacardoso/langchain-rag-alura/tree/project/agente_futebol)**: Um agente que responde perguntas sobre as regras do futebol com base no regulamento oficial da FIFA.
2. **[Agente Farmacêutico  — Bulas de Paracetamol e Dipirona](https://github.com/jessicacardoso/langchain-rag-alura/tree/project/agente_farmaceutico)**: Um agente que fornece informações sobre medicamentos com base em bulas reais de paracetamol e dipirona.
3. **[Agente de RH — Políticas de Empresa](https://github.com/jessicacardoso/langchain-rag-alura/tree/project/agente_rh)**: Um agente que responde dúvidas sobre políticas internas de uma empresa fictícia.

## 🛠️ Tecnologias Utilizadas
- [Langchain](https://langchain.com/): Biblioteca para construção de aplicações com modelos de linguagem.
- [Embeddings](https://sbert.net/): Representações vetoriais de texto para facilitar a busca semântica.
- [Busca Semântica](https://www.pinecone.io/): Métodos para recuperação de informações relevantes.
- [Recuperação-Baseada em Geração (RAG)](https://arxiv.org/abs/2005.11401): Técnica que combina recuperação de informações com geração de texto.
- [LLMs (Modelos de Linguagem de Grande Porte)](https://ollama.com/): Modelos como OpenAI GPT-4, Hugging Face, entre outros.


## ⚙️ Pré-requisitos
Antes de executar os exemplos, certifique-se de ter o seguinte instalado em sua máquina:
- Python 3.12 ou superior
- Git
- Qualquer gerenciador de ambientes virtuais (opcional, mas recomendado)
- Acesso LLMs (como OpenAI, Hugging Face, etc.)

## 🚀 Como Executar os Exemplos
1. Clone este repositório:
   ```bash
   git clone https://github.com/jessicacardoso/langchain-rag-alura
   cd langchain-rag-alura
   ```
2. Escolha a branch do exemplo que deseja executar (por exemplo, `agente_futebol`, `agente_farmaceutico`, `agente_rh`).
    ```bash
    git checkout project/agente_futebol
    uv sync
    uv run python main.py
    ```

## 🤝 Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests para melhorias ou novos exemplos.

## ✍️ Reconhecimento
Os exemplos usados aqui são parte do curso [Arquiteturas RAG com LLMs: embeddings, busca semântica e criação de agentes com LangChain](https://cursos.alura.com.br/course/langchain-chatbots-rag) da Alura. 

