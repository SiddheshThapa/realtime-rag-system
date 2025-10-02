# services/producer/sample_producer.py
import os, json, time
from kafka import KafkaProducer

KAFKA = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
producer = KafkaProducer(bootstrap_servers=KAFKA, value_serializer=lambda v: json.dumps(v).encode())

docs = [
    {"id": "10", "title": "AI history", "text": "Artificial intelligence research began in the 1950s."},
    {"id": "11", "title": "Neural networks", "text": "Neural networks are inspired by biological neurons."},
    {"id": "12", "title": "Transformer", "text": "Transformer models use attention mechanisms."}
]

for d in docs:
    msg = {**d, "source":"sample", "correlation_id": f"sample-{int(time.time())}"}
    producer.send("wikipedia.edits", msg)
    print("sent", d["title"])
    time.sleep(0.5)

producer.flush()
