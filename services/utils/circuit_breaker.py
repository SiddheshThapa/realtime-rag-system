# services/utils/circuit_breaker.py
import time, logging
logger = logging.getLogger("circuit_breaker")

class SimpleCircuitBreaker:
    def __init__(self, failures=3, reset_timeout=30):
        self.failures = failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False

    def call(self, func, *args, **kwargs):
        if self.is_open:
            if (time.time() - self.last_failure_time) > self.reset_timeout:
                try:
                    res = func(*args, **kwargs)
                    self.reset()
                    return res
                except Exception as e:
                    self._trip()
                    raise
            else:
                raise Exception("Circuit is open")
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.failures:
                self._trip()
            raise

    def reset(self):
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False

    def _trip(self):
        self.last_failure_time = time.time()
        self.is_open = True
        logger.error("Circuit tripped")
