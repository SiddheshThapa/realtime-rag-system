# services/vector_store/redis_adapter.py
import json
import redis
import numpy as np
import threading
from typing import List, Dict, Any

class RedisAdapter:
    def __init__(self, redis_host="redis", redis_port=6379, dim: int = 384):
        self.r = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        self.dim = dim
        self.lock = threading.Lock()
        if not self.r.exists("vectors:next_id"):
            self.r.set("vectors:next_id", 0)

    def _next_id(self) -> int:
        return int(self.r.incr("vectors:next_id")) - 1

    def add_vector(self, embedding: List[float], metadata: Dict[str, Any]) -> int:
        vec = np.array(embedding, dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != self.dim:
            raise ValueError(f"Embedding dimension mismatch. expected {self.dim}, got {vec.shape}")
        with self.lock:
            vid = metadata.get("id")
            if vid is None:
                vid = self._next_id()
            self.r.set(f"vector:{vid}:emb", json.dumps(vec.tolist()))
            self.r.set(f"vector:{vid}:meta", json.dumps(metadata))
            self.r.sadd("vectors:ids", str(vid))
        return int(vid)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        q = np.array(query_embedding, dtype=np.float32)
        if q.shape[0] != self.dim:
            raise ValueError("Query embedding dim mismatch")
        ids = list(self.r.smembers("vectors:ids") or [])
        results = []
        for sid in ids:
            emb_raw = self.r.get(f"vector:{sid}:emb")
            meta_raw = self.r.get(f"vector:{sid}:meta")
            if not emb_raw or not meta_raw:
                continue
            emb = np.array(json.loads(emb_raw), dtype=np.float32)
            denom = (np.linalg.norm(q) * np.linalg.norm(emb))
            score = float(np.dot(q, emb) / denom) if denom > 0 else 0.0
            meta = json.loads(meta_raw)
            meta_copy = meta.copy()
            meta_copy['score'] = score
            results.append(meta_copy)
        results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        return results[:k]

    def get_vector_count(self) -> int:
        return self.r.scard("vectors:ids")

    def close(self):
        pass
