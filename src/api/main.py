from __future__ import annotations
"""FastAPI application for Melbourne Property Intelligence.

Provides REST endpoints for querying the property knowledge base,
viewing suburb statistics, and managing data ingestion.
"""

import asyncio
import logging
import re
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.ingestion.storage import init_db, query_suburb_stats
from src.query.rag import RAGResponse, rag_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Melbourne Property Intelligence",
    description="LLM-powered insights into Melbourne's property market",
    version="0.1.0",
)
app.state.limiter = limiter

# Initialize database on startup
init_db()

# --- Input Sanitisation ---

MAX_QUERY_LENGTH = 500

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"system\s*prompt",
    r"you\s+are\s+now",
    r"forget\s+(everything|all)",
    r"disregard\s+(all\s+)?previous",
    r"new\s+instructions\s*:",
    r"override\s+(all\s+)?instructions",
    r"act\s+as\s+if",
    r"pretend\s+you\s+are",
    r"jailbreak",
    r"\[INST\]",
    r"<\|im_start\|>",
]

_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def sanitize_query(query: str) -> str | None:
    """Validate and sanitise user query.

    Returns the sanitised query string, or None if the query
    appears to be a prompt injection attempt.
    """
    if not query or not query.strip():
        return None

    if _INJECTION_RE.search(query):
        return None

    return query.strip()[:MAX_QUERY_LENGTH]


# --- Request/Response Models ---


class QueryRequest(BaseModel):
    query: str
    n_results: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str


class SuburbStatsResponse(BaseModel):
    suburb: str
    total_sales: int
    avg_price: Optional[float] = None
    median_price: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    avg_distance_km: Optional[float] = None


class IngestRequest(BaseModel):
    source: str  # "auctions", "news", "all"


class HealthResponse(BaseModel):
    status: str
    version: str


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query_property_knowledge(request: Request, body: QueryRequest):
    """Query the property knowledge base using natural language.

    Ask questions about Melbourne property market, suburbs, prices,
    trends, and auction results. Returns an AI-generated answer
    with source citations.
    """
    clean_query = sanitize_query(body.query)
    if clean_query is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid query. Please ask a question about Melbourne property.",
        )

    try:
        result: RAGResponse = await asyncio.to_thread(
            rag_query,
            query=clean_query,
            n_results=body.n_results,
        )
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            query=result.query,
        )
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/suburbs/{suburb}", response_model=SuburbStatsResponse)
async def get_suburb_stats(suburb: str):
    """Get auction statistics for a specific suburb."""
    stats = await asyncio.to_thread(query_suburb_stats, suburb.upper())
    if stats["total_sales"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No auction data found for suburb: {suburb}",
        )
    return SuburbStatsResponse(**stats)


@app.post("/ingest")
@limiter.limit("5/minute")
async def trigger_ingestion(request: Request, body: IngestRequest):
    """Trigger data ingestion from configured sources.

    Sources: 'auctions', 'news', or 'all'.
    """
    from src.ingestion.news_scraper import collect_property_news
    from src.ingestion.storage import store_news_articles

    results = {"auctions": 0, "news": 0}

    if body.source in ("news", "all"):
        articles = await asyncio.to_thread(collect_property_news)
        article_dicts = [
            {
                "title": a.title,
                "url": a.url,
                "source": a.source,
                "published_date": a.published_date,
                "content": a.content,
                "summary": a.summary,
                "date_scraped": date.today(),
            }
            for a in articles
        ]
        await asyncio.to_thread(store_news_articles, article_dicts)
        results["news"] = len(article_dicts)

    return {"status": "completed", "results": results}


@app.get("/collections")
async def list_collections():
    """List all vector store collections and their stats."""
    from src.index.vectorstore import get_collection_stats

    return {
        "collections": [
            await asyncio.to_thread(get_collection_stats, "property_knowledge"),
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
