# services/utils/logging_config.py
import logging, json, uuid

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage()
        }
        if hasattr(record, "correlation_id"):
            payload["correlation_id"] = record.correlation_id
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)

def configure_logging():
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(JsonFormatter())
        root.setLevel(logging.INFO)
        root.addHandler(h)
    # quiet noisy libs
    logging.getLogger("kafka").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def new_correlation_id() -> str:
    return str(uuid.uuid4())
