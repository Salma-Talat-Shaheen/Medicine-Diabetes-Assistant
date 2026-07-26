<div align="center">

# 💊 Medicine & Diabetes Assistant

### An AI-powered clinical decision-support agent for medicine and dosage guidance

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-1C3C3C?style=for-the-badge)](https://www.langchain.com/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-FF6F00?style=for-the-badge)](https://www.trychroma.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-LLMs-8A2BE2?style=for-the-badge)](https://openrouter.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)

*Helping clinicians make faster, safer, evidence-backed medicine and dosage decisions.*

[Overview](#-overview) • [Live Demo](#-live-demo) • [Features](#-key-features) • [How It Works](#-how-it-works) • [Knowledge Base](#-knowledge-base) • [Getting Started](#-getting-started) • [Project Structure](#-project-structure) • [Team](#-team)

</div>

---

## **Live Demo**

🔗 **[https://medicine-assistant.onrender.com/](https://medicine-assistant.onrender.com/)**

<p align="center">
  <img src="src/web/static/images/webApp.png" alt="Web Application Screenshot" width="70%"/>
</p>

---

## **Overview**

**Medicine Assistant** is an intelligent agent that supports clinicians in selecting appropriate **medicines and dosages**, with a focus on **diabetes care**. It combines a **LangGraph-based reasoning agent** with a **Retrieval-Augmented Generation (RAG)** pipeline, using **ChromaDB** as a vector store, **PostgreSQL** for structured data persistence, and **OpenRouter** LLMs for reasoning — so recommendations are grounded in retrieved medical reference data rather than model guesswork alone.

Instead of relying purely on an LLM's internal knowledge (which can be outdated or hallucinated), the agent retrieves relevant, trusted context first, then reasons over it step by step to produce a transparent, traceable recommendation.

> ⚠️ **Disclaimer:** This project is a decision-support and educational tool. It is **not** a substitute for professional medical judgment, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

##  Key Features

| | |
|---|---|
|  **LangGraph Agent** | Structured, stateful, multi-step reasoning pipeline instead of a single black-box prompt |
|  **RAG-Powered Retrieval** | Grounded, citation-friendly answers using ChromaDB vector search over medical references |
|  **PostgreSQL Integration** | Structured data storage for query history, session management, and audit logging |
|  **Flexible LLM Access** | Swap between models via OpenRouter without changing application code |
|  **Web Interface** | Lightweight, accessible web app for interactive querying |
|  **Containerized** | Fully Dockerized for consistent, reproducible deployment anywhere |
|  **Configurable Pipeline** | Centralized settings for model choice, chunk size, and retrieval depth (top-k) |
|  **Test Coverage** | Dedicated test suite to validate agent and retrieval behavior |

---

##  How It Works ?

```mermaid
flowchart LR
    A[User Query] --> B[LangGraph Agent]
    B --> C[RAG Retriever]
    C --> D[(ChromaDB Vector Store)]
    D --> C
    C --> E[Context-Grounded Prompt]
    E --> F[OpenRouter LLM]
    F --> G[Recommendation + Reasoning]
    G --> H[Web Interface]
    B <--> I[(PostgreSQL)]
```

1. **Query intake**: the user submits a medicine or dosage question through the web interface.
2. **Agent orchestration**: the LangGraph agent manages state and decides what information is needed.
3. **Retrieval**: relevant chunks are pulled from ChromaDB based on semantic similarity.
4. **Structured persistence**: session data, query history, and audit logs are stored in PostgreSQL.
5. **Grounded generation**: the retrieved context is passed to the LLM via OpenRouter to produce a well-supported answer.
6. **Response**: the final recommendation and reasoning are returned to the user.

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Agent Framework | LangGraph |
| Vector Retrieval | ChromaDB (semantic search) |
| Structured Storage | PostgreSQL |
| LLM Access | OpenRouter |
| Frontend Tooling | Node.js, Tailwind CSS |
| Deployment | Docker, Docker Compose |
| Testing | Pytest |

---

## Knowledge Base

The assistant's knowledge is grounded in authoritative clinical references on diabetes management. Each source was processed through a rigorous pipeline: documents were converted to tamper-proof images, then **OCR** was applied to extract text and structured table data before indexing into ChromaDB.

<p align="center">
  <img src="src/web/static/images/MANAGING DIABETES.png" alt="Managing Diabetes" height="300"/>
  &nbsp;&nbsp;&nbsp;
  <img src="src/web/static/images/Diabetes2025.png" alt="Standards of Care in Diabetes 2025" height="300"/>
</p>

<p align="center">

**Managing Diabetes — Clinical Reference Guide**<br/>
A WHO/Bhutan clinical guide for health workers covering diabetes pharmacology, dosage protocols, and monitoring parameters. Converted to images and processed with OCR to extract tabular drug data.

**[Standards of Care in Diabetes — 2025](https://www.binasss.sa.cr/standards-of-care-2025.pdf)**<br/>
The ADA's 2025 clinical practice guidelines covering glycemic targets, medication algorithms, and special populations. OCR was used to capture structured protocol tables and graded recommendations.

**[WHO Diabetes Management Guidelines](https://extranet.who.int/ncdccs/Data/BTN_D1_Diabetes%20Management%20guidelines.pdf)**<br/>
The World Health Organization's evidence-based guidelines for diabetes management in primary care settings. Provides WHO-recommended first- and second-line treatment pathways and monitoring protocols.

</p>

---

## 🚀 Getting Started

### Prerequisites
- [Conda](https://docs.conda.io/) (recommended) or a Python 3.10 virtual environment
- An [OpenRouter](https://openrouter.ai/) API key
- A running [PostgreSQL](https://www.postgresql.org/) instance
- Docker (optional, for containerized setup)

### 1. Clone the repository
```bash
git clone https://github.com/Salma-Talat-Shaheen/Medicine-Diabetes-Assistant.git
cd Medicine-Diabetes-Assistant
```

### 2. Create and activate the environment
```bash
conda create -n medicine-assistant python=3.10 -y
conda activate medicine-assistant
```

### 3. Install dependencies
```bash
pip install -e .
pip install -e ".[dev]"
```

### 4. Configure environment variables
```bash
cp .env.example .env
```
Then open `.env` and set your `OPENROUTER_API_KEY` and `DATABASE_URL` (PostgreSQL connection string). Additional settings (model name, chunk size, top-k retrieval) can be tuned in `src/config.py`.

### 5. Run the app
```bash
python src/web/app.py
```

###  Or run with Docker
```bash
docker-compose up --build
```

---

##  Project Structure

```
Medicine-Diabetes-Assistant/
├── chroma_db/             # Vector database storage
├── medicine-assistant/    # Core application package
├── scripts/               # Utility & setup scripts
├── src/
│   ├── agent.py           # LangGraph agent definition
│   ├── config.py          # Central configuration (model, chunking, top-k)
│   ├── llm.py             # LLM provider integration (OpenRouter)
│   ├── main.py            # Application entry point
│   ├── rag.py             # RAG pipeline (retrieval + generation)
│   ├── state.py           # Agent state management
│   └── web/app.py         # Web application entry point
├── tests/                 # Test suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

##  Testing

```bash
pytest tests/
```

---

##  Contributing
 
Contributions, issues, and feature requests are welcome!
 
1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
   
---

##  Team

This project was built with dedication by:

| Contributor | Role |
|---|---|
| **Salma Shaheen** | Development | 220210654 |
| **Hebatallah AbuHarb** | Development | 220210448 |
| **Zahraa Alderawi** | Development | 220221428 |

---

##  License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

<div align="center">

⭐ If you find this project useful, consider giving it a star!

</div>
