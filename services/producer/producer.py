# services/producer/producer.py
import os, json, time, signal, sys, logging
import requests
from sseclient import SSEClient
from kafka import KafkaProducer
from kafka.errors import KafkaError

WIKI_SSE_URL   = os.getenv("WIKI_SSE_URL", "https://stream.wikimedia.org/v2/stream/recentchange")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC          = os.getenv("TOPIC", "wikipedia.edits")
USER_AGENT     = os.getenv("USER_AGENT", "realtime-rag/0.1 (+https://example.com; contact@example.com)")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"producer","msg":"%(message)s"}',
)
log = logging.getLogger("producer")

_shutdown = False
def _sigterm(*_):
    global _shutdown
    _shutdown = True
    log.info("received shutdown signal; exiting...")
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, _sigterm)

def make_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP.split(","),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=10,
        retries=5,
        max_in_flight_requests_per_connection=1,
    )

def stream_events():
    # Wikimedia blocks clients without a real User-Agent.
    headers = {
        "Accept": "text/event-stream",
        "User-Agent": USER_AGENT,
        # Optional but harmless:
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    session = requests.Session()
    r = session.get(WIKI_SSE_URL, stream=True, timeout=(10, 310), headers=headers)
    r.raise_for_status()
    return SSEClient(r)

def run():
    log.info(f"bootstrap='{KAFKA_BOOTSTRAP}' topic='{TOPIC}' url='{WIKI_SSE_URL}'")
    producer = None
    backoff = 1
    while not _shutdown:
        try:
            if producer is None:
                producer = make_producer()
                log.info("Kafka producer connected")

            log.info(f"connecting to SSE {WIKI_SSE_URL}")
            messages = 0
            client = stream_events()
            for event in client.events():
                if _shutdown:
                    break
                if not event.data or event.event != "message":
                    continue
                try:
                    data = json.loads(event.data)
                except json.JSONDecodeError:
                    continue

                # Build payload to match processor.Edit
                meta = data.get("meta", {}) or {}
                doc = {
                    "id": str(data.get("id") or int(time.time()*1000)),
                    "title": data.get("title") or data.get("page") or "unknown",
                    "text": data.get("comment") or "",
                    "lang": meta.get("domain", "en"),
                    "user": data.get("user") or "",
                    "comment": data.get("comment") or "",
                    "timestamp": float(data.get("timestamp") or time.time()),
                    "source": "wikimedia",
                    "correlation_id": meta.get("id") or f"cid-{int(time.time()*1000)}",
                }

                producer.send(TOPIC, doc)
                messages += 1
                if messages % 100 == 0:
                    log.info(f"produced {messages} messages so far...")

            log.info("SSE stream ended; flushing")
            producer.flush()
            backoff = 1

        except requests.exceptions.RequestException as e:
            log.warning(f"SSE error: {e}; retrying in {backoff}s")
            time.sleep(backoff); backoff = min(backoff * 2, 60)
        except KafkaError as e:
            log.warning(f"Kafka error: {e}; recreating producer in {backoff}s")
            try:
                if producer:
                    producer.flush(timeout=5); producer.close(timeout=5)
            except Exception:
                pass
            producer = None
            time.sleep(backoff); backoff = min(backoff * 2, 60)
        except Exception as e:
            log.exception(f"unexpected error: {e}; retrying in {backoff}s")
            time.sleep(backoff); backoff = min(backoff * 2, 60)

    try:
        if producer:
            log.info("producer exiting: flushing")
            producer.flush(timeout=5); producer.close(timeout=5)
    except Exception:
        pass

if __name__ == "__main__":
    run()
