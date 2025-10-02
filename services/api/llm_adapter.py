# services/api/llm_adapter.py
import os, logging, requests, json
from typing import Optional

logger = logging.getLogger("llm_adapter")


class BaseLLM:
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        raise NotImplementedError


class OllamaAdapter(BaseLLM):
    def __init__(self, ollama_url: str | None):
        self.ollama_url = ollama_url
        self.model = os.getenv("OLLAMA_MODEL", "llama3")  # default to llama3

    def generate(self, prompt: str, max_tokens: int = 256) -> Optional[str]:
        if not self.ollama_url:
            return None
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,                # ✅ required by Ollama
                    "prompt": prompt,
                    "stream": True,                     # use streaming mode
                    "options": {"num_predict": max_tokens},
                },
                stream=True,
                timeout=180,
            )
            resp.raise_for_status()

            full_text = []
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                    chunk = obj.get("response", "")
                    full_text.append(chunk)
                    if obj.get("done", False):  # Ollama signals end
                        break
                except Exception as parse_err:
                    logger.warning("Could not parse Ollama line: %s", parse_err)

            return "".join(full_text).strip() or None

        except Exception as e:
            logger.error("OllamaAdapter failed: %s", e, exc_info=True)
            return None


class HFAdapter(BaseLLM):
    def __init__(self, model_name="google/flan-t5-small"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        out = self.model.generate(**inputs, max_length=max_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


def get_llm_adapter(ollama_url: Optional[str] = None):
    if ollama_url:
        try:
            a = OllamaAdapter(ollama_url)
            return a
        except Exception:
            pass
    return HFAdapter()
