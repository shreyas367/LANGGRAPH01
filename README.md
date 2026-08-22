# LangGraph Learning

Examples for building LangGraph workflows, conversational agents, and retrieval-augmented generation (RAG) applications with Python.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Groq API key for the LLM examples

## Setup

From the project root:

```powershell
uv sync
```

Create a local `.env` file in the project root. Never commit this file.

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
MISTRAL_API_KEY=your_mistral_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
OPENWEATHER_API_KEY=your_openweather_api_key
```

## Project Examples

### Basic chat

```powershell
uv run python chat.py
```

### Conditional workflow

```powershell
uv run python conditionalagents/conditionalexercise.py
```

### Looping workflow

```powershell
uv run python loopingagent/loop1basics.py
```

### Conversational memory agent

```powershell
uv run python agents/memoryagent.py
```

### PDF RAG

The RAG examples load PDFs from the project root, split them into chunks, create embeddings, and search the chunks with FAISS.

```powershell
uv run python conditionalagents/conditional_RAG.py
```

### Hybrid RAG

The hybrid example combines:

- BM25 keyword retrieval
- FAISS semantic retrieval
- Ensemble ranking
- FlashRank reranking
- Groq answer generation

Run it from the `RAGmodified` directory:

```powershell
cd RAGmodified
uv run python hybridragsearch.py
```

Ask a question about the DBMS PDF, then type `exit` to stop.

## RAG Data and Indexes

The sample PDFs are stored in the project root:

- `DBMS-Unit-2.pdf`
- `Operating Systems Lecture Notes.pdf`

FAISS indexes are generated in the `faiss_indexes/` directory. They can be rebuilt by deleting the relevant index directory and running the RAG script again.

## Security

- Keep API keys in `.env` or environment variables.
- Do not commit `.env`, API keys, tokens, or private model files.
- Rotate any credential that has been exposed.
- Treat locally loaded FAISS pickle files as trusted files only.

## Common Commands

```powershell
# Install or update dependencies
uv sync

# Check repository changes
git status

# View code changes
git diff

# Run a Python file from the project root
uv run python path/to/script.py
```
