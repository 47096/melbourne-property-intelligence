# Melbourne Property Intelligence

LLM-powered insights into Melbourne's property market — auction data, suburb analysis, and natural language queries with source citations.

## Problem

Melbourne's property auction market generates thousands of results weekly across scattered sources. Getting actionable insights requires manually comparing suburbs, tracking price trends, and reading through listing after listing. This project automates all of it: scrape auction results, enrich with geolocation, build a searchable knowledge base, and query it conversationally using RAG.

## Architecture

```
Data Layer          →  Processing Layer  →  Model Layer     →  Serving Layer
Web scraping           Chunking              RAG pipeline        FastAPI
RSS feeds              Embeddings            Claude/Ollama       Streamlit
SQLite + Parquet       ChromaDB              Evaluation          Docker
```

## What's Inside

| Module | What It Does |
|--------|-------------|
| `src/ingestion/scraper.py` | Scrapes auction results, parses prices |
| `src/ingestion/geocoder.py` | Suburb coordinates, CBD distance |
| `src/ingestion/news_scraper.py` | Property news from RSS feeds |
| `src/ingestion/storage.py` | SQLite + Parquet storage layer |
| `src/index/chunker.py` | Text chunking strategies |
| `src/index/embedder.py` | Sentence-transformer embeddings |
| `src/index/vectorstore.py` | ChromaDB vector store |
| `src/query/rag.py` | RAG pipeline with citations |
| `src/query/llm_client.py` | Claude / Ollama / fallback client |
| `src/api/main.py` | FastAPI REST endpoints |
| `src/tracking/mlflow_utils.py` | Experiment tracking |
| `dashboard/app.py` | Streamlit interactive dashboard |

## Setup

### Quick Start

```bash
git clone https://github.com/47096/melbourne-property-intelligence.git
cd melbourne-property-intelligence
pip install -r requirements.txt
```

### Docker

```bash
docker-compose up
```

API available at `http://localhost:8000`, dashboard at `http://localhost:8501`.

### LLM Configuration

Set one of:

```bash
# Option 1: Claude API (recommended)
export ANTHROPIC_API_KEY=your_key_here

# Option 2: Local Ollama
ollama pull llama3
# No env var needed — auto-detected
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|------------|
| GET | `/health` | Health check |
| POST | `/query` | Ask a natural language question about Melbourne property |
| GET | `/suburbs/{suburb}` | Get auction statistics for a suburb |
| POST | `/ingest` | Trigger data ingestion from news feeds |
| GET | `/collections` | View vector store status |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the median house price in Richmond?"}'
```

Response:

```json
{
  "answer": "Based on recent auction data, the median house price in Richmond is approximately $1.8M...",
  "sources": [
    {"text": "Richmond auction results show...", "score": 0.87, "source": "news_rss"}
  ],
  "query": "What is the median house price in Richmond?"
}
```

## Evaluation

The project tracks RAG quality metrics via MLflow:

- **Relevance score** — cosine similarity between query and retrieved chunks
- **Source diversity** — number of unique sources in context
- **Query latency** — p50, p95 response times
- **Answer length** — proxy for detail level

```bash
# View experiments
mlflow ui
# Open http://localhost:5000
```

## Testing

```bash
pytest tests/ -v
```

## Tech Stack

- **Data**: BeautifulSoup, feedparser, SQLite, Parquet
- **ML**: sentence-transformers, ChromaDB
- **LLM**: Claude API / Ollama
- **API**: FastAPI, uvicorn
- **Dashboard**: Streamlit
- **Tracking**: MLflow
- **Deploy**: Docker, GitHub Actions

## License

MIT