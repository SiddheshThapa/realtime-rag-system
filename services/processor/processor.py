# services/processor/processor.py
import os, sys, json, time, logging, signal, threading
from kafka import KafkaConsumer, KafkaProducer
from pydantic import BaseModel
from prometheus_client import start_http_server, Histogram, Counter

sys.path.append('/app')
from services.utils.settings import settings
from services.utils.logging_config import configure_logging
from services.utils.pii_utils import mask_pii
from services.utils.circuit_breaker import SimpleCircuitBreaker
from sentence_transformers import SentenceTransformer
from services.vector_store.adapter import get_vector_store

configure_logging()
logger = logging.getLogger("processor")

KAFKA = settings.kafka_bootstrap

consumer = None
dlq_producer = None

# metrics
start_http_server(9101)
EMBED_LATENCY = Histogram('processor_embedding_seconds','embedding time')
PROC_SUCC = Counter('processor_success_total','successful processed messages')
PROC_FAIL = Counter('processor_failure_total','failed processed messages')

# settings
MODEL_LOCAL_DIR = "/data/models/sentence-transformers/all-MiniLM-L6-v2"
DIM = 384

# create embedding model (prefer local cached path)
try:
    if os.path.exists(MODEL_LOCAL_DIR):
        model = SentenceTransformer(MODEL_LOCAL_DIR)
    else:
        model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.exception("Failed to load embedding model: %s", e)
    raise

# vector store adapter (will use RedisAdapter by default unless USE_FAISS=true in .env)
faiss = get_vector_store(index_path='/data/faiss/index.faiss', idmap_path='/data/faiss/idmap.json', dim=DIM)
cb = SimpleCircuitBreaker()

r_running = True
def shutdown(sig, frame):
    global r_running
    r_running = False
    logger.info("processor shutdown requested")

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# in services/processor/processor.py
class Edit(BaseModel):
    id: str | None = None
    title: str | None = None
    user: str | None = None
    comment: str | None = None
    text: str | None = None
    timestamp: float | None = None
    source: str | None = None
    correlation_id: str | None = None


def health_http_server():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/health':
                self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
            else:
                self.send_response(404); self.end_headers()
    server = HTTPServer(('0.0.0.0', 9201), H)
    server.serve_forever()

def process_message(msg):
    try:
        data = Edit(**msg)
        text = data.text or data.comment or ""
        text = mask_pii(text)
        cid = data.correlation_id or f"cid-{int(time.time()*1000)}"
        enriched = {
            "id": data.id,
            "title": data.title,
            "text": text,
            "source": data.source,
            "original_ts": data.timestamp,
            "processed_ts": time.time(),
            "correlation_id": cid
        }
        with EMBED_LATENCY.time():
            emb = model.encode(text)
            emb = emb.astype('float32').tolist()
        faiss.add_vector(embedding=emb, metadata=enriched)
        PROC_SUCC.inc()
        logger.info("processed id=%s title=%s", data.id, data.title)
    except Exception as e:
        logger.exception("processor failed: %s", e)
        PROC_FAIL.inc()
        dlq_payload = {"original": msg, "error": str(e), "ts": time.time()}
        try:
            dlq_producer.send("wikipedia.edits.dlq", dlq_payload); dlq_producer.flush(timeout=5)
        except Exception:
            logger.exception("failed to push to kafka dlq")
        try:
            import redis
            r = redis.Redis(host=settings.redis_host, port=settings.redis_port, db=0)
            r.rpush("processor:dlq", json.dumps(dlq_payload))
        except Exception:
            logger.exception("failed to push to redis dlq")

def initialize_components():
    global consumer, dlq_producer
    # kafka consumer
    for attempt in range(10):
        try:
            consumer = KafkaConsumer(
                'wikipedia.edits',
                bootstrap_servers=KAFKA,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',  # start from the end
                enable_auto_commit=True,
                group_id='processor-group-v2'  # new group to avoid old bad records
            )
            dlq_producer = KafkaProducer(bootstrap_servers=KAFKA, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
            logger.info("Kafka connected")
            break
        except Exception as e:
            logger.warning("Kafka not ready, retrying... %s", e)
            time.sleep(3)
    if consumer is None:
        raise RuntimeError("Could not connect to Kafka")

if __name__ == "__main__":
    t = threading.Thread(target=health_http_server, daemon=True)
    t.start()
    initialize_components()
    logger.info("processor started - consuming wikipedia.edits")
    try:
        while r_running:
            for msg in consumer:
                if not r_running:
                    break
                process_message(msg.value)
            time.sleep(0.1)
    finally:
        logger.info("processor shutting down: flushing and closing")
        try: dlq_producer.flush(); dlq_producer.close()
        except: pass
        try: consumer.close()
        except: pass
        try: faiss.close()
        except: pass
        logger.info("processor shutdown complete")
