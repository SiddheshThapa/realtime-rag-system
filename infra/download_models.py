# infra/download_models.py
from huggingface_hub import snapshot_download
import os

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    # add more if you want, e.g. "google/flan-t5-small"
]

dst = "/data/models"
os.makedirs(dst, exist_ok=True)

for repo in MODELS:
    print(f"Downloading {repo} into {dst}. This may take a few minutes...")
    snapshot_download(repo_id=repo, cache_dir=dst)
    print(f"Downloaded {repo}")

print("All models downloaded.")
