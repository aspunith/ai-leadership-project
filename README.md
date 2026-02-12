# AI Leadership Insight & Decision Agent

A **Retrieval-Augmented Generation (RAG)** application that ingests internal company documents and answers leadership questions grounded in real organisational data — covering performance, risks, strategy, and operations.

Built with **Python**, **LangChain**, **ChromaDB**, **sentence-transformers**, and **Google Gemini**, the agent demonstrates core Agentic AI concepts in a clean, modular architecture.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Agentic AI Concepts Used](#agentic-ai-concepts-used)
4. [Architecture & Data Flow](#architecture--data-flow)
5. [Project Structure](#project-structure)
6. [Module Descriptions](#module-descriptions)
7. [Setup & Installation](#setup--installation)
8. [Usage](#usage)
9. [Running Tests](#running-tests)
10. [Configuration Reference](#configuration-reference)
11. [Design Decisions](#design-decisions)

---

## Problem Statement

Senior leadership needs quick, accurate, data-backed answers from internal documents (annual reports, quarterly results, strategy notes, operational updates) without manually combing through hundreds of pages. An AI agent should retrieve the most relevant information and synthesise concise, cited answers.

---

## Solution Overview

The agent implements a full **RAG pipeline** with three stages:

| Stage | Module | What It Does |
|-------|--------|-------------|
| **Ingest** | `agent/ingest.py` | Loads `.txt`, `.pdf`, `.docx` files → splits into overlapping chunks → embeds with sentence-transformers or OpenAI → persists to ChromaDB |
| **Retrieve** | `agent/retriever.py` | Converts the user's question into a vector → performs cosine similarity search → returns the top-K most relevant chunks |
| **Generate** | `agent/generator.py` | Feeds retrieved context + question to Gemini / GPT-4o (or falls back to showing raw passages if no LLM is configured) |

**Tri-mode design:** Works fully offline with local `all-MiniLM-L6-v2` embeddings and raw-passage fallback. Also supports **Google Gemini 2.0 Flash** or **OpenAI GPT-4o** when API keys are provided. Priority: OpenAI → Gemini → Fallback.

---

## Agentic AI Concepts Used

| Concept | Where It's Applied |
|---------|-------------------|
| **RAG (Retrieval-Augmented Generation)** | Core architecture: retrieve real documents before generating answers, eliminating hallucination of company-specific data |
| **Document Chunking with Overlap** | `ingest.py` — `RecursiveCharacterTextSplitter` (1000-char chunks, 200-char overlap) ensures no information is lost at chunk boundaries |
| **Vector Embeddings** | Text is converted to dense numerical vectors capturing semantic meaning, via `all-MiniLM-L6-v2` (local) or `text-embedding-3-small` (OpenAI) |
| **Vector Store & Similarity Search** | ChromaDB stores embeddings persistently; at query time, cosine similarity identifies the top-5 most relevant passages |
| **Prompt Engineering / System Prompting** | `generator.py` uses a structured system prompt that constrains the LLM to answer only from provided context, cite data, and avoid hallucination |
| **Context Grounding (Faithfulness)** | The LLM receives retrieved passages as explicit context and is instructed to base answers strictly on them |
| **Graceful Degradation** | If no API key is present, the system still functions — returning raw retrieved passages instead of failing. Supports automatic fallback: OpenAI → Gemini → raw passages |
| **LangChain Orchestration** | LangChain abstractions (document loaders, text splitters, embeddings, vector stores, chat models) provide a modular, swappable framework |

---

## Architecture & Data Flow

```
                         ┌──────────────────────┐
                         │  company_documents/   │
                         │  (.txt, .pdf, .docx)  │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                 INGEST  │   1. Load Documents   │  (DirectoryLoader)
                         │   2. Chunk Text       │  (RecursiveCharacterTextSplitter)
                         │   3. Embed Chunks     │  (HuggingFace / OpenAI)
                         │   4. Store Vectors    │  (ChromaDB)
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │    vector_store/      │  (Persisted ChromaDB)
                         └──────────┬───────────┘
                                    │
               User Question        │
                    │                │
                    ▼                ▼
              ┌──────────────────────────────┐
     RETRIEVE │  Embed Question → Cosine     │
              │  Similarity → Top-K Chunks   │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
     GENERATE │  System Prompt + Context     │
              │  + Question → LLM Answer     │
              |  (Gemini / GPT-4o / fallback)  |
              └──────────────┬───────────────┘
                             │
                             ▼
                      Final Answer
```

---

## Project Structure

```
ai-leadership-agent/
├── config.py                # Centralised settings (paths, models, parameters)
├── main.py                  # CLI entry-point (--ingest, --query, --interactive)
├── requirements.txt         # Pinned dependencies
├── .env.example             # Template for API keys (OpenAI / Gemini)
├── README.md
│
├── agent/
│   ├── __init__.py
│   ├── ingest.py            # Stage 1: Document loading, chunking, embedding
│   ├── retriever.py         # Stage 2: Vector similarity search
    └── generator.py         # Stage 3: Answer synthesis (Gemini / GPT-4o / fallback)
│
├── company_documents/       # Source documents (add your own here)
│   ├── annual_report_2025.txt
│   ├── q3_quarterly_report.txt
│   ├── strategy_notes.txt
│   └── operational_update.txt
│
├── vector_store/            # Auto-created ChromaDB persistence directory
│
└── tests/
    └── test_agent.py        # Unit tests (loading, chunking, retrieval, generation)
```

---

## Module Descriptions

### `config.py` — Configuration
All tuneable settings in one place. Reads `OPENAI_API_KEY` and `GOOGLE_API_KEY` from a `.env` file and auto-detects which LLM backend to use (priority: OpenAI → Gemini → fallback). Key parameters:
- Chunk size (1000 chars) and overlap (200 chars)
- Top-K retrieval count (5)
- LLM temperature (0.2) and max tokens (1024)
- LLM selection flags: `USE_OPENAI`, `USE_GEMINI`

### `agent/ingest.py` — Document Ingestion Pipeline
- **`load_documents()`** — Uses LangChain `DirectoryLoader` with file-type-specific loaders (`TextLoader`, `PyPDFLoader`, `Docx2txtLoader`) to read all supported files.
- **`split_documents()`** — Splits loaded documents into overlapping chunks using `RecursiveCharacterTextSplitter` with smart separators (`\n\n`, `\n`, `. `, ` `).
- **`get_embedding_function()`** — Returns OpenAI or HuggingFace embeddings based on configuration.
- **`build_vector_store()`** — Embeds chunks and persists them to ChromaDB. Clears old data on re-ingest to prevent duplicates.
- **`ingest_pipeline()`** — Orchestrates the full load → chunk → embed → persist flow.

### `agent/retriever.py` — Semantic Retrieval
- **`get_retriever()`** — Loads the persisted ChromaDB store and returns a LangChain retriever configured for cosine similarity search.
- **`retrieve()`** — Takes a natural-language query, embeds it, and returns the top-K most relevant document chunks.

### `agent/generator.py` — Answer Generation
- **`SYSTEM_PROMPT`** — Instructs the LLM to act as a leadership insight agent, answer only from context, cite data, and avoid hallucination.
- **`generate_answer()`** — Routes to OpenAI, Gemini, or the fallback mode based on available configuration.
- **`_openai_answer()`** — Sends system prompt + retrieved context + question to GPT-4o via LangChain's `ChatOpenAI`.
- **`_gemini_answer()`** — Sends system prompt + retrieved context + question to Gemini 2.0 Flash via LangChain's `ChatGoogleGenerativeAI`.
- **`_fallback_answer()`** — Returns the retrieved passages directly when no LLM is configured.

### `main.py` — CLI Entry Point
Three modes of operation:
- `--ingest` — Run the document ingestion pipeline
- `--query "question"` — Ask a single question and get an answer
- `--interactive` — Start a Q&A loop for multiple questions

### `tests/test_agent.py` — Unit Tests
Covers five areas:
1. Document loading (valid directory and missing directory error handling)
2. Document chunking (correct split sizes)
3. Context formatting (passage numbering and source citation)
4. Fallback answer generation (output format without OpenAI key)
5. Full round-trip integration test (ingest → retrieve → validate relevance)

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- (Optional) A **Google Gemini API key** (free tier at [Google AI Studio]([https://aistudio.google.dev](https://aistudio.google.com/))) or an **OpenAI API key**

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd ai-leadership-agent

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your LLM backend (choose one)
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux

# Option A: Google Gemini (free tier, recommended)
#   Set in .env:   GOOGLE_API_KEY=AIza...

# Option B: OpenAI (paid)
#   Set in .env:   OPENAI_API_KEY=sk-...

# 5. Ingest the sample company documents
python main.py --ingest

# 6. Ask a question
python main.py --query "What is our current revenue trend?"
```

> **No API key?** The agent still works — it uses a local `all-MiniLM-L6-v2`
> model for embeddings and returns the top retrieved passages as-is (no LLM
> synthesis).
>
> **Priority order:** If both keys are configured, the agent picks:
> OpenAI → Gemini → raw-passage fallback.

---

## Usage

### Ingest Documents
```bash
python main.py --ingest
```
Loads all `.txt`, `.pdf`, and `.docx` files from `company_documents/`, chunks them, generates embeddings, and stores them in ChromaDB. Re-running clears old data automatically.

### Single Query
```bash
python main.py --query "What are our biggest risks?"
```

### Interactive Mode
```bash
python main.py --interactive
```
Opens a Q&A loop — type questions and get answers continuously. Type `quit` or `exit` to stop.

### Adding Your Own Documents
Place any `.txt`, `.pdf`, or `.docx` files in `company_documents/`, then re-run:
```bash
python main.py --ingest
```

### Sample Questions to Try
```
"What is our current revenue trend?"
"What are the key risks facing the company?"
"How is the Cloud Services division performing?"
"What is the engineering attrition rate?"
"What are the strategic priorities for FY 2026?"
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover document loading, chunking, context formatting, fallback answers, and a full ingest-to-retrieve integration test — all using temporary directories (no side effects on the real vector store).

---

## Configuration Reference

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `DOCUMENTS_DIR` | `company_documents/` | Directory containing source documents |
| `CHROMA_PERSIST_DIR` | `vector_store/` | ChromaDB persistence directory |
| `OPENAI_API_KEY` | `""` (from `.env`) | OpenAI API key — enables GPT-4o mode |
| `GOOGLE_API_KEY` | `""` (from `.env`) | Google API key — enables Gemini mode |
| `USE_OPENAI` | Auto-detected | `True` when a valid OpenAI key is present |
| `USE_GEMINI` | Auto-detected | `True` when a Google key is present and OpenAI is not |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | OpenAI chat model for answer generation |
| `GEMINI_CHAT_MODEL` | `gemini-2.0-flash` | Google Gemini chat model for answer generation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local HuggingFace embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks to retrieve per query |
| `MAX_ANSWER_TOKENS` | `1024` | Maximum tokens in LLM response |
| `TEMPERATURE` | `0.2` | LLM temperature (lower = more deterministic) |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **ChromaDB for vector storage** | Lightweight, embedded, no external server needed — ideal for a self-contained assessment project |
| **Tri-mode LLM support (OpenAI / Gemini / local)** | Supports OpenAI GPT-4o (paid), Google Gemini (free tier), and a no-LLM fallback — maximises accessibility and demonstrates multi-provider LLM integration |
| **LangChain orchestration** | Industry-standard framework; modular abstractions make each component independently swappable |
| **Overlapping chunks (200-char overlap)** | Prevents information loss at chunk boundaries — critical for accurate retrieval |
| **Clear old store on re-ingest** | Prevents duplicate passages from accumulating across multiple ingest runs |
| **System prompt with strict grounding rules** | Minimises hallucination by constraining the LLM to cited, context-based answers |
| **Separate config module** | Single source of truth for all parameters — easy to tune for different use cases |
