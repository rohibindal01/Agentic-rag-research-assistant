# 🤖 Agentic RAG Research Assistant

An autonomous, multi-tool research assistant powered by LangGraph agent orchestration, Retrieval-Augmented Generation (RAG), and LangSmith observability. Designed to ingest academic PDFs, answer domain-specific questions with cited sources, and self-correct via a ReAct-style agent loop.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green)
![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-orange)

---

## 🧠 Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         LangGraph Agent Loop        │
│  ┌──────────┐    ┌───────────────┐  │
│  │  Router  │───▶│  Tool Caller  │  │
│  └──────────┘    └───────┬───────┘  │
│        ▲                 │          │
│        │    ┌────────────▼──────┐   │
│  ┌─────┴──┐ │   Tool Results    │   │
│  │Reflect │◀│ (RAG / Web / Cite)│   │
│  └────────┘ └───────────────────┘   │
└─────────────────────────────────────┘
    │
    ▼
Grounded Answer + Source Citations
    │
    ▼
LangSmith Trace + RAGAS Evaluation
```

**Key components:**
- **Ingestion Pipeline** — PDF chunking, embedding, and FAISS/Pinecone vector store indexing
- **Multi-Tool Agent** — LangGraph ReAct loop with 3 tools: RAG retriever, web search, citation fetcher
- **Reflexion Module** — Self-critique and re-routing when confidence score is below threshold
- **Evaluation Suite** — RAGAS metrics (faithfulness, answer relevance, context precision) + LangSmith tracing
- **Streamlit UI** — Interactive frontend with source cards, confidence scores, and agent trace viewer

---

## 📁 Project Structure

```
agentic-rag-research-assistant/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── graph.py              # LangGraph agent graph definition
│   │   ├── nodes.py              # Agent node functions (router, tool_caller, reflect)
│   │   └── state.py              # AgentState TypedDict
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── rag_tool.py           # FAISS/Pinecone retrieval tool
│   │   ├── web_search_tool.py    # Tavily web search tool
│   │   └── citation_tool.py      # Semantic Scholar / CrossRef citation fetcher
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── ingest.py             # PDF ingestion and chunking pipeline
│   │   ├── embeddings.py         # Embedding model wrapper
│   │   └── vectorstore.py        # Vector store init and management
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── ragas_eval.py         # RAGAS evaluation pipeline
│   │   └── metrics.py            # Custom metric definitions
│   └── ui/
│       ├── app.py                # Streamlit application entry point
│       └── components.py         # Reusable UI components
├── data/
│   ├── raw/                      # Raw PDF files (gitignored)
│   ├── processed/                # Chunked text files
│   └── vectorstore/              # Persisted FAISS index (gitignored)
├── notebooks/
│   ├── 01_ingestion_demo.ipynb
│   ├── 02_agent_walkthrough.ipynb
│   └── 03_ragas_evaluation.ipynb
├── tests/
│   ├── test_ingestion.py
│   ├── test_agent.py
│   ├── test_tools.py
│   └── test_evaluation.py
├── configs/
│   ├── config.yaml               # Main config (models, chunking params, thresholds)
│   └── prompts.yaml              # All system and tool prompts
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI pipeline
├── .env
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/rohibindal01/agentic-rag-research-assistant.git
cd agentic-rag-research-assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Ingest your PDFs
```bash
python -m src.retrieval.ingest --input_dir data/raw/ --output_dir data/processed/
```

### 5. Run the Streamlit app
```bash
streamlit run src/ui/app.py
```

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key — free tier at [console.groq.com](https://console.groq.com) |
| `LANGCHAIN_API_KEY` | LangSmith API key for tracing |
|LANGCHAIN_TRACING_V2=true |
| `LANGCHAIN_PROJECT` | LangSmith project name |
| `TAVILY_API_KEY` | Tavily web search API key |
| `PINECONE_API_KEY` | Pinecone API key (optional; defaults to local FAISS) |
| `PINECONE_ENV` | Pinecone environment (optional) |

---

## 🛠️ Agent Tools

| Tool | Description | Source |
|---|---|---|
| `rag_retriever` | Semantic search over ingested PDF corpus | FAISS / Pinecone |
| `web_search` | Real-time web search for recent information | Tavily API |
| `citation_fetcher` | Fetches paper metadata and abstracts by DOI/title | Semantic Scholar API |

---

## 📊 Evaluation Metrics

Evaluated using [RAGAS](https://github.com/explodinggradients/ragas) on a curated Q&A test set:

| Metric | Score |
|---|---|
| Faithfulness | — |
| Answer Relevance | — |
| Context Precision | — |
| Context Recall | — |

> Scores will be populated after running `python -m src.evaluation.ragas_eval`

---

## 🔭 LangSmith Observability

All agent runs are traced end-to-end via LangSmith. Set `LANGCHAIN_TRACING_V2=true` in your `.env` to enable.

Traces capture:
- Full agent loop iterations
- Tool inputs/outputs at each step
- Token usage and latency per node
- Reflexion re-routing events

---

## 🧪 Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## 🗺️ Roadmap

- [x] PDF ingestion pipeline with recursive text splitting
- [x] FAISS vector store with persistence
- [x] LangGraph ReAct agent with 3 tools
- [x] Reflexion self-critique loop
- [x] LangSmith tracing integration
- [x] RAGAS evaluation pipeline
- [x] Streamlit UI with source citations
- [ ] Pinecone cloud vector store support
- [ ] Multi-document cross-reference graph (knowledge graph)
- [ ] Docker containerization
- [ ] REST API endpoint (FastAPI)


