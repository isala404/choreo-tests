import asyncio
import hashlib
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Optional
import numpy as np

import aiosqlite
import feedparser
import spacy
from sentence_transformers import SentenceTransformer
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import json

# Configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", "news.db")
FETCH_INTERVAL = int(os.getenv("FETCH_INTERVAL", "5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Structured JSON logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=getattr(logging, LOG_LEVEL), handlers=[handler])
logger = logging.getLogger(__name__)

# RSS Feed Sources
RSS_FEEDS = [
    {"url": "https://feeds.bbci.co.uk/news/rss.xml", "source": "BBC News", "category": "world"},
    {"url": "https://feeds.bbci.co.uk/news/technology/rss.xml", "source": "BBC Technology", "category": "tech"},
    {"url": "https://feeds.bbci.co.uk/news/business/rss.xml", "source": "BBC Business", "category": "business"},
    {"url": "https://feeds.npr.org/1001/rss.xml", "source": "NPR News", "category": "world"},
    {"url": "https://techcrunch.com/feed", "source": "TechCrunch", "category": "tech"},
    {"url": "https://www.wired.com/feed/rss", "source": "Wired", "category": "tech"},
    {"url": "https://www.engadget.com/rss.xml", "source": "Engadget", "category": "tech"},
    {"url": "https://www.aljazeera.com/xml/rss/all.xml", "source": "Al Jazeera", "category": "world"},
    {"url": "https://www.nasa.gov/rss/dyn/breaking_news.rss", "source": "NASA", "category": "science"},
    {"url": "https://finance.yahoo.com/news/rssindex", "source": "Yahoo Finance", "category": "business"},
]

# Track feed failures for exponential backoff
feed_failures: dict[str, int] = {}
last_fetch_time: Optional[datetime] = None

# Models
nlp = None
embedder = None

# Entity resolution using embeddings
class EntityResolver:
    """Resolves entity variations using semantic similarity."""

    def __init__(self, similarity_threshold: float = 0.75):
        self.threshold = similarity_threshold
        self.canonical_entities: dict[str, str] = {}  # normalized -> canonical
        self.embeddings: dict[str, np.ndarray] = {}   # canonical -> embedding

    def _basic_normalize(self, text: str) -> str:
        """Basic text normalization."""
        if not text:
            return ""
        # Strip possessives
        text = re.sub(r"['']s$", "", text)
        # Normalize whitespace
        text = " ".join(text.split()).strip()
        return text

    def resolve(self, entity_text: str) -> str:
        """Resolve entity to canonical form using embeddings."""
        if not embedder or not entity_text:
            return entity_text

        normalized = self._basic_normalize(entity_text)
        lower_key = normalized.lower()

        # Check cache first
        if lower_key in self.canonical_entities:
            return self.canonical_entities[lower_key]

        # No existing entities? This becomes the canonical form
        if not self.embeddings:
            self.canonical_entities[lower_key] = normalized
            self.embeddings[normalized] = embedder.encode(normalized, convert_to_numpy=True)
            return normalized

        # Compute embedding for new entity
        new_embedding = embedder.encode(normalized, convert_to_numpy=True)

        # Find most similar existing entity
        best_match = None
        best_similarity = 0.0

        for canonical, existing_embedding in self.embeddings.items():
            similarity = np.dot(new_embedding, existing_embedding) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = canonical

        # If similar enough, use existing canonical form
        if best_similarity >= self.threshold and best_match:
            self.canonical_entities[lower_key] = best_match
            return best_match

        # Otherwise, this is a new canonical entity
        self.canonical_entities[lower_key] = normalized
        self.embeddings[normalized] = new_embedding
        return normalized

    def load_from_db(self, entities: list[str]):
        """Pre-load known entities from database."""
        for entity in entities:
            if entity and entity not in self.embeddings:
                self.embeddings[entity] = embedder.encode(entity, convert_to_numpy=True)
                self.canonical_entities[entity.lower()] = entity
        logger.info(f"Loaded {len(self.embeddings)} known entities into resolver")

entity_resolver = EntityResolver()

def load_models():
    """Load spaCy and sentence transformer models."""
    global nlp, embedder

    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded")
    except OSError:
        logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        raise

    # Load sentence transformer for entity resolution (~80MB)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Sentence transformer model loaded")

# Database initialization
async def init_db():
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                url_hash TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                source_name TEXT NOT NULL,
                category TEXT NOT NULL,
                published_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                article_id INTEGER NOT NULL,
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS entity_trends (
                entity_name TEXT NOT NULL,
                mention_count INTEGER NOT NULL,
                window_start TIMESTAMP NOT NULL,
                window_hours INTEGER NOT NULL,
                PRIMARY KEY (entity_name, window_start, window_hours)
            );

            CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);
            CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_article ON entities(article_id);
            CREATE INDEX IF NOT EXISTS idx_trends_window ON entity_trends(window_start, window_hours);
        """)
        await db.commit()
    logger.info("Database initialized")

# Entity extraction
def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove tags
    text = re.sub(r'&\w+;', ' ', text)    # Remove HTML entities
    text = re.sub(r'https?://\S+', ' ', text)  # Remove URLs
    text = ' '.join(text.split())          # Normalize whitespace
    return text

def extract_entities(text: str) -> list[dict]:
    """Extract ORG, PERSON, GPE entities from text using spaCy + embedding resolution."""
    if not nlp or not text:
        return []

    # Clean HTML before processing
    text = strip_html(text)
    doc = nlp(text)
    entities = []
    seen = set()

    for ent in doc.ents:
        if ent.label_ not in ("ORG", "PERSON", "GPE"):
            continue

        # Skip very short or numeric entities
        if len(ent.text.strip()) < 2:
            continue

        # Resolve to canonical form using embeddings
        canonical_name = entity_resolver.resolve(ent.text)

        if not canonical_name or canonical_name.lower() in seen:
            continue

        seen.add(canonical_name.lower())
        entities.append({"name": canonical_name, "type": ent.label_})

    return entities

# RSS fetching
async def fetch_feed(feed_info: dict, timeout: float = 10.0) -> list[dict]:
    """Fetch and parse a single RSS feed."""
    url = feed_info["url"]

    # Check exponential backoff
    failures = feed_failures.get(url, 0)
    if failures > 0:
        backoff_minutes = min(2 ** failures, 60)
        # Skip this feed if in backoff period (simplified check)
        if failures > 3:
            logger.warning(f"Skipping {feed_info['source']} due to repeated failures")
            return []

    try:
        loop = asyncio.get_event_loop()
        parsed = await asyncio.wait_for(
            loop.run_in_executor(None, feedparser.parse, url),
            timeout=timeout
        )

        if parsed.bozo and not parsed.entries:
            raise Exception(f"Feed parse error: {parsed.bozo_exception}")

        articles = []
        for entry in parsed.entries:
            article = {
                "url": entry.get("link", ""),
                "title": entry.get("title", ""),
                "summary": entry.get("summary", entry.get("description", "")),
                "source": feed_info["source"],
                "category": feed_info["category"],
                "published": None
            }

            # Parse published date
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                article["published"] = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                article["published"] = datetime(*entry.updated_parsed[:6])

            if article["url"] and article["title"]:
                articles.append(article)

        # Reset failure count on success
        feed_failures[url] = 0
        logger.info(f"Fetched {len(articles)} articles from {feed_info['source']}")
        return articles

    except asyncio.TimeoutError:
        feed_failures[url] = failures + 1
        logger.error(f"Timeout fetching {feed_info['source']}")
        return []
    except Exception as e:
        feed_failures[url] = failures + 1
        logger.error(f"Error fetching {feed_info['source']}: {e}")
        return []

async def fetch_all_feeds() -> list[dict]:
    """Fetch all RSS feeds concurrently."""
    tasks = [fetch_feed(feed) for feed in RSS_FEEDS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles = []
    for result in results:
        if isinstance(result, list):
            articles.extend(result)

    return articles

# Database operations
async def article_exists(db: aiosqlite.Connection, url_hash: str) -> bool:
    """Check if article already exists."""
    cursor = await db.execute(
        "SELECT 1 FROM articles WHERE url_hash = ?", (url_hash,)
    )
    return await cursor.fetchone() is not None

async def insert_article_with_entities(
    db: aiosqlite.Connection,
    article: dict,
    entities: list[dict]
) -> Optional[int]:
    """Insert article and its entities in a transaction."""
    url_hash = hashlib.md5(article["url"].encode()).hexdigest()

    if await article_exists(db, url_hash):
        return None

    cursor = await db.execute("""
        INSERT INTO articles (url, url_hash, title, summary, source_name, category, published_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        article["url"],
        url_hash,
        article["title"],
        article["summary"][:1000] if article["summary"] else "",
        article["source"],
        article["category"],
        article["published"]
    ))

    article_id = cursor.lastrowid

    for entity in entities:
        await db.execute("""
            INSERT INTO entities (name, type, article_id)
            VALUES (?, ?, ?)
        """, (entity["name"], entity["type"], article_id))

    return article_id

async def update_entity_trends(db: aiosqlite.Connection):
    """Update entity trends for 1h and 24h windows."""
    now = datetime.utcnow()

    for hours in [1, 24]:
        window_start = now - timedelta(hours=hours)

        # Get current counts
        cursor = await db.execute("""
            SELECT e.name, COUNT(*) as count
            FROM entities e
            JOIN articles a ON e.article_id = a.id
            WHERE a.published_at >= ?
            GROUP BY e.name
            ORDER BY count DESC
        """, (window_start,))

        rows = await cursor.fetchall()

        # Clear old trends for this window
        await db.execute("""
            DELETE FROM entity_trends
            WHERE window_hours = ? AND window_start < ?
        """, (hours, now - timedelta(hours=hours*2)))

        # Insert new trends
        for name, count in rows:
            await db.execute("""
                INSERT OR REPLACE INTO entity_trends (entity_name, mention_count, window_start, window_hours)
                VALUES (?, ?, ?, ?)
            """, (name, count, now, hours))

# Background worker
async def fetch_and_process():
    """Main background worker task."""
    global last_fetch_time

    logger.info("Starting feed fetch cycle")
    start_time = datetime.utcnow()

    try:
        # Set timeout for entire cycle (4 minutes to ensure completion before next run)
        async with asyncio.timeout(240):
            articles = await fetch_all_feeds()
            logger.info(f"Fetched {len(articles)} total articles")

            async with aiosqlite.connect(DATABASE_PATH) as db:
                # Load known entities from database into resolver
                cursor = await db.execute("SELECT DISTINCT name FROM entities")
                rows = await cursor.fetchall()
                known_entities = [row[0] for row in rows]
                entity_resolver.load_from_db(known_entities)

                new_count = 0

                for article in articles:
                    text = f"{article['title']} {article['summary'] or ''}"
                    entities = extract_entities(text)

                    article_id = await insert_article_with_entities(db, article, entities)
                    if article_id:
                        new_count += 1

                await db.commit()
                logger.info(f"Inserted {new_count} new articles")

                # Update trends
                await update_entity_trends(db)
                await db.commit()
                logger.info("Updated entity trends")

            last_fetch_time = datetime.utcnow()
            elapsed = (last_fetch_time - start_time).total_seconds()
            logger.info(f"Fetch cycle completed in {elapsed:.2f}s")

    except asyncio.TimeoutError:
        logger.error("Fetch cycle timed out")
    except Exception as e:
        logger.error(f"Fetch cycle error: {e}")

# FastAPI app setup
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_models()
    await init_db()

    scheduler = AsyncIOScheduler()
    scheduler.add_job(fetch_and_process, 'interval', minutes=FETCH_INTERVAL)
    scheduler.start()
    logger.info(f"Scheduler started with {FETCH_INTERVAL} minute interval")

    # Run initial fetch
    asyncio.create_task(fetch_and_process())

    yield

    # Shutdown
    scheduler.shutdown()

app = FastAPI(
    title="News Aggregator API",
    description="Aggregates news from RSS feeds with entity extraction and trending analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/health")
@limiter.limit("100/minute")
async def health_check(request: Request):
    """Service health and last fetch timestamp."""
    return {
        "status": "healthy",
        "last_fetch": last_fetch_time.isoformat() if last_fetch_time else None,
        "feeds_configured": len(RSS_FEEDS)
    }

@app.get("/trending")
@limiter.limit("100/minute")
async def get_trending(
    request: Request,
    window: str = Query("24h", regex="^(1h|24h)$")
):
    """Get top 20 trending entities."""
    hours = 1 if window == "1h" else 24
    prev_hours = hours * 2

    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row

        now = datetime.utcnow()
        window_start = now - timedelta(hours=hours)
        prev_window_start = now - timedelta(hours=prev_hours)

        # Current window counts
        cursor = await db.execute("""
            SELECT e.name, e.type, COUNT(*) as count
            FROM entities e
            JOIN articles a ON e.article_id = a.id
            WHERE a.published_at >= ?
            GROUP BY e.name, e.type
            ORDER BY count DESC
            LIMIT 20
        """, (window_start,))
        current = {row["name"]: dict(row) for row in await cursor.fetchall()}

        # Previous window counts for comparison
        cursor = await db.execute("""
            SELECT e.name, COUNT(*) as count
            FROM entities e
            JOIN articles a ON e.article_id = a.id
            WHERE a.published_at >= ? AND a.published_at < ?
            GROUP BY e.name
        """, (prev_window_start, window_start))
        previous = {row["name"]: row["count"] for row in await cursor.fetchall()}

        # Calculate trends
        trending = []
        for name, data in current.items():
            prev_count = previous.get(name, 0)
            change = 0
            if prev_count > 0:
                change = ((data["count"] - prev_count) / prev_count) * 100
            elif data["count"] > 0:
                change = 100

            trending.append({
                "entity": name,
                "type": data["type"],
                "count": data["count"],
                "change_percent": round(change, 1),
                "is_breaking": change > 200
            })

        return {"window": window, "trending": trending}

@app.get("/trending/{category}")
@limiter.limit("100/minute")
async def get_trending_by_category(
    request: Request,
    category: str,
    window: str = Query("24h", regex="^(1h|24h)$")
):
    """Get trending entities filtered by category."""
    hours = 1 if window == "1h" else 24

    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row

        window_start = datetime.utcnow() - timedelta(hours=hours)

        cursor = await db.execute("""
            SELECT e.name, e.type, COUNT(*) as count
            FROM entities e
            JOIN articles a ON e.article_id = a.id
            WHERE a.published_at >= ? AND a.category = ?
            GROUP BY e.name, e.type
            ORDER BY count DESC
            LIMIT 20
        """, (window_start, category))

        results = [dict(row) for row in await cursor.fetchall()]

        return {"category": category, "window": window, "trending": results}

@app.get("/entity/{name}")
@limiter.limit("100/minute")
async def get_entity_articles(
    request: Request,
    name: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get articles mentioning an entity."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Check if entity exists
        cursor = await db.execute(
            "SELECT COUNT(*) as count FROM entities WHERE LOWER(name) = LOWER(?)",
            (name,)
        )
        row = await cursor.fetchone()
        if row["count"] == 0:
            raise HTTPException(status_code=404, detail="Entity not found")

        cursor = await db.execute("""
            SELECT DISTINCT a.id, a.title, a.url, a.summary, a.source_name,
                   a.category, a.published_at
            FROM articles a
            JOIN entities e ON a.id = e.article_id
            WHERE LOWER(e.name) = LOWER(?)
            ORDER BY a.published_at DESC
            LIMIT ? OFFSET ?
        """, (name, limit, offset))

        articles = [dict(row) for row in await cursor.fetchall()]

        return {
            "entity": name,
            "count": len(articles),
            "offset": offset,
            "articles": articles
        }

@app.get("/articles/breaking")
@limiter.limit("100/minute")
async def get_breaking_articles(
    request: Request,
    category: Optional[str] = None
):
    """Get articles from the last 2 hours."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row

        two_hours_ago = datetime.utcnow() - timedelta(hours=2)

        if category:
            cursor = await db.execute("""
                SELECT id, title, url, summary, source_name, category, published_at
                FROM articles
                WHERE published_at >= ? AND category = ?
                ORDER BY published_at DESC
            """, (two_hours_ago, category))
        else:
            cursor = await db.execute("""
                SELECT id, title, url, summary, source_name, category, published_at
                FROM articles
                WHERE published_at >= ?
                ORDER BY published_at DESC
            """, (two_hours_ago,))

        articles = [dict(row) for row in await cursor.fetchall()]

        return {"count": len(articles), "articles": articles}

@app.get("/articles/{article_id}/related")
@limiter.limit("100/minute")
async def get_related_articles(request: Request, article_id: int):
    """Get up to 5 articles sharing the most entities."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Check article exists
        cursor = await db.execute(
            "SELECT id FROM articles WHERE id = ?", (article_id,)
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Article not found")

        # Get entities for this article
        cursor = await db.execute(
            "SELECT name FROM entities WHERE article_id = ?", (article_id,)
        )
        entity_names = [row["name"] for row in await cursor.fetchall()]

        if not entity_names:
            return {"article_id": article_id, "related": []}

        # Find articles sharing these entities
        placeholders = ",".join("?" * len(entity_names))
        cursor = await db.execute(f"""
            SELECT a.id, a.title, a.url, a.source_name, a.category, a.published_at,
                   COUNT(DISTINCT e.name) as shared_entities
            FROM articles a
            JOIN entities e ON a.id = e.article_id
            WHERE e.name IN ({placeholders}) AND a.id != ?
            GROUP BY a.id
            ORDER BY shared_entities DESC, a.published_at DESC
            LIMIT 5
        """, (*entity_names, article_id))

        related = [dict(row) for row in await cursor.fetchall()]

        return {"article_id": article_id, "related": related}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
