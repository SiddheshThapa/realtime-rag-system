# services/vector_store/adapter.py
import os

USE_FAISS = os.environ.get("USE_FAISS", "false").lower() in ("1", "true", "yes")
if USE_FAISS:
    from services.vector_store.faiss_adapter import FaissAdapter  # type: ignore
    def get_vector_store(**kwargs):
        return FaissAdapter(**kwargs)
else:
    from services.vector_store.redis_adapter import RedisAdapter  # type: ignore
    def get_vector_store(**kwargs):
        return RedisAdapter(redis_host=os.environ.get("REDIS_HOST","redis"), redis_port=int(os.environ.get("REDIS_PORT","6379")), dim=kwargs.get("dim", 384))
