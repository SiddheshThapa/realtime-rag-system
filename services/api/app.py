# services/api/app.py
import os, sys, json, logging, hashlib, time, re, uuid
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import redis
import jwt
from prometheus_client import Summary, make_asgi_app
from starlette_prometheus import PrometheusMiddleware
from sentence_transformers import SentenceTransformer

# Internal imports
sys.path.append('/app')
from services.utils.logging_config import configure_logging
from services.utils.settings import settings
from services.utils.otel_instrumentation import init_tracer
from services.utils.circuit_breaker import SimpleCircuitBreaker
from services.vector_store.adapter import get_vector_store

# -----------------------------------------------------------------------------
# Bootstrap / Observability
# -----------------------------------------------------------------------------
configure_logging()
logger = logging.getLogger("api")
tracer = init_tracer("multi-stage-rag-api")

app = FastAPI(title="Multi-Stage RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(PrometheusMiddleware)
app.mount("/metrics", make_asgi_app())

REQ_LATENCY = Summary('api_request_seconds', 'Request latency')
ASK_LATENCY = Summary('api_ask_seconds', 'Ask request latency')

# -----------------------------------------------------------------------------
# Globals / Settings
# -----------------------------------------------------------------------------
redis_client: Optional[redis.Redis] = None
faiss_adapter = None
llm_adapter = None
embedding_model: Optional[SentenceTransformer] = None

cb = SimpleCircuitBreaker()
JWT_SECRET = settings.jwt_secret
FAISS_DIR = "/data/faiss"

# Guardrails / version
MAX_QUESTION_CHARS = 1000
MAX_K = 10
MIN_K = 1
MAX_TOKENS = 512
DEFAULT_TOKENS = 256
SERVICE_VERSION = "1.0.0"

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class Query(BaseModel):
    query: str
    k: int = 5

# Production-grade ask models
class AskRequest(BaseModel):
    question: str = Field(..., description="User question")
    k: int = Field(3, description="Top-K documents to retrieve")
    max_tokens: int = Field(DEFAULT_TOKENS, description="LLM generation limit")

    @field_validator("question")
    @classmethod
    def _v_question(cls, v: str) -> str:
        v = sanitize_question(v)
        if not v:
            raise ValueError("question cannot be empty")
        if len(v) > MAX_QUESTION_CHARS:
            raise ValueError(f"question too long (>{MAX_QUESTION_CHARS} chars)")
        return v

    @field_validator("k")
    @classmethod
    def _v_k(cls, v: int) -> int:
        return clamp(v, MIN_K, MAX_K)

    @field_validator("max_tokens")
    @classmethod
    def _v_tokens(cls, v: int) -> int:
        return clamp(v, 1, MAX_TOKENS)

class AskContext(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None
    source: Optional[str] = None
    score: Optional[float] = None

class AskResponse(BaseModel):
    request_id: str
    model: Optional[str]
    answer: Optional[str]
    contexts: List[AskContext]
    latency_ms: int
    version: str = SERVICE_VERSION

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")

def sanitize_question(q: str) -> str:
    # Remove control chars, trim, collapse whitespace
    q = _CONTROL_CHARS_RE.sub(" ", q or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _retrieve(question: str, k: int) -> List[Dict[str, Any]]:
    """Embed the question and query FAISS."""
    emb = embedding_model.encode(question).astype('float32').tolist()
    results = faiss_adapter.search(emb, k=k)
    return results or []

def _build_prompt(question: str, results: List[Dict[str, Any]]) -> str:
    """Safe prompt with bounded context and injection guards."""
    MAX_CTX_CHARS = 8000
    ctx_chunks = []
    total = 0
    for i, r in enumerate(results):
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        block = f"### Doc {i+1} — {r.get('title')}\n{txt}\n"
        if total + len(block) > MAX_CTX_CHARS:
            break
        ctx_chunks.append(block)
        total += len(block)

    context_block = "\n".join(ctx_chunks) if ctx_chunks else "No relevant context."

    return (
        "You are a careful assistant. Use ONLY the provided context.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer:"
    )

# -----------------------------------------------------------------------------
# Init dependencies
# -----------------------------------------------------------------------------
def initialize_components():
    global redis_client, faiss_adapter, embedding_model, llm_adapter

    # Redis (retry a bit on cold start)
    for attempt in range(10):
        try:
            redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=0,
                decode_responses=True
            )
            redis_client.ping()
            logger.info("Redis connected")
            break
        except Exception as e:
            logger.warning("Redis not ready, retrying... %s", e)
            time.sleep(2)
    if redis_client is None:
        raise RuntimeError("Redis connection failed")

    # Vector store
    os.makedirs(FAISS_DIR, exist_ok=True)
    DIM = 384
    faiss_adapter = get_vector_store(
        index_path=os.path.join(FAISS_DIR, 'index.faiss'),
        idmap_path=os.path.join(FAISS_DIR,'idmap.json'),
        dim=DIM
    )
    logger.info("Vector store initialized")

    # Embeddings (prefer pre-downloaded path)
    local_model_dir = "/data/models/sentence-transformers/all-MiniLM-L6-v2"
    try:
        if os.path.exists(local_model_dir):
            embedding_model = SentenceTransformer(local_model_dir)
        else:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model ready")
    except Exception as e:
        logger.exception("Failed to load embedding model: %s", e)
        raise

    # LLM adapter (Ollama by default; falls back to HF)
    from services.api.llm_adapter import get_llm_adapter
    llm_adapter = get_llm_adapter(settings.ollama_url)
    logger.info("LLM adapter initialized")

@app.on_event("startup")
def startup_event():
    initialize_components()
    logger.info("API started")

@app.on_event("shutdown")
def shutdown_event():
    try:
        if faiss_adapter:
            faiss_adapter.close()
    except Exception:
        logger.exception("error closing faiss")

# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------
def check_jwt(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    auth: Optional[str] = Header(None, alias="Auth"),
):
    """
    Accept either 'Authorization: Bearer <token>' or 'Auth: Bearer <token>'.
    """
    raw = authorization or auth
    if not raw or not raw.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth")
    token = raw.replace("Bearer ", "")
    try:
        jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    except Exception:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/ready")
def ready():
    checks = {}
    try:
        redis_client.ping(); checks['redis']='ok'
    except Exception as e:
        checks['redis']=f"fail:{e}"
    try:
        # If this call works we consider FAISS healthy
        _ = getattr(faiss_adapter, "get_vector_count", lambda: None)()
        checks['faiss'] = 'ok'
    except Exception as e:
        checks['faiss']=f"fail:{e}"
    ready_ok = all(isinstance(v, str) and v == 'ok' for v in checks.values())
    return {"ready": ready_ok, "checks": checks}

@app.post("/auth/token")
def token():
    encoded = jwt.encode({"sub":"dev","iat":int(time.time())}, JWT_SECRET, algorithm='HS256')
    return {"token": encoded}

# ---------- Existing /query (kept as-is; small cleanup) ----------
@app.post("/query")
@REQ_LATENCY.time()
def query(q: Query, auth=Depends(check_jwt)):
    start_time = time.time()
    qhash = hashlib.sha256(q.query.encode()).hexdigest()

    # cache read
    try:
        cached = redis_client.get(f"query:{qhash}")
        if cached:
            resp = json.loads(cached)
            resp['cached'] = True
            resp['processing_time'] = time.time() - start_time
            return resp
    except Exception:
        pass

    # retrieval
    try:
        results = _retrieve(q.query, q.k)
    except Exception:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Vector search failed")

    # generation
    contexts = [r.get('text') for r in results if r]
    prompt = f"Answer the question using the following contexts:\n\n{contexts}\n\nQuestion: {q.query}"
    try:
        answer = llm_adapter.generate(prompt)
    except Exception:
        answer = None

    resp = {"results": results, "answer": answer}
    # cache write
    try:
        redis_client.setex(f"query:{qhash}", 300, json.dumps(resp))
        redis_client.setex(f"fallback:{qhash}", 3600, json.dumps(resp))
    except Exception:
        pass

    resp['processing_time'] = time.time() - start_time
    resp['cached'] = False
    return resp

# ---------- New production-grade /v1/ask (+ /ask alias) ----------
@app.post("/v1/ask", response_model=AskResponse)
@ASK_LATENCY.time()
def ask_v1(payload: AskRequest, auth=Depends(check_jwt)):
    request_id = str(uuid.uuid4())
    t0 = time.time()

    # Retrieval
    try:
        results = _retrieve(payload.question, payload.k)
    except Exception as e:
        logger.exception("[%s] retrieval failed: %s", request_id, e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="retrieval_failed")

    contexts = [
        AskContext(
            id=r.get("id"),
            title=r.get("title"),
            text=r.get("text"),
            source=r.get("source"),
            score=r.get("score"),
        ) for r in (results or [])
    ]

    # If no useful contexts, short-circuit with graceful message
    if not any(c.text for c in contexts):
        latency_ms = int((time.time() - t0) * 1000)
        resp = AskResponse(
            request_id=request_id,
            model=getattr(llm_adapter, "model", None),
            answer="I couldn’t find enough supporting context to answer confidently.",
            contexts=contexts,
            latency_ms=latency_ms,
        )
        logger.info("[%s] ask v1 served (no-context) in %dms", request_id, latency_ms)
        return resp

    # LLM generation
    answer = None
    try:
        prompt = _build_prompt(payload.question, results)
        answer = llm_adapter.generate(prompt, max_tokens=payload.max_tokens)
        if isinstance(answer, str):
            answer = answer.strip()
    except Exception as e:
        logger.warning("[%s] generation failed: %s", request_id, e)
        # keep HTTP 200 with explicit null answer
        answer = None

    latency_ms = int((time.time() - t0) * 1000)
    resp = AskResponse(
        request_id=request_id,
        model=getattr(llm_adapter, "model", None),
        answer=answer,
        contexts=contexts,
        latency_ms=latency_ms,
    )
    logger.info("[%s] ask v1 served in %dms (k=%d, tokens=%d)", request_id, latency_ms, payload.k, payload.max_tokens)
    return resp

@app.post("/ask", response_model=AskResponse)
def ask_alias(payload: AskRequest, auth=Depends(check_jwt)):
    # Backward-compatible alias for demos/clients
    return ask_v1(payload, auth)
