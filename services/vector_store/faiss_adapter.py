# services/vector_store/faiss_adapter.py
import os, json, threading, logging
import faiss
import numpy as np

logger = logging.getLogger("faiss_adapter")

class FaissAdapter:
    def __init__(self, index_path="/data/faiss/index.faiss", idmap_path="/data/faiss/idmap.json", dim=384):
        self.index_path = index_path
        self.idmap_path = idmap_path
        self.dim = dim
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if os.path.exists(index_path) and os.path.getsize(index_path) > 0:
            try:
                self.index = faiss.read_index(index_path)
            except Exception:
                self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        if os.path.exists(idmap_path):
            try:
                with open(idmap_path,'r') as f: self.idmap = json.load(f)
            except:
                self.idmap = {}
        else:
            self.idmap = {}

    def add_vector(self, embedding, metadata: dict):
        vec = np.array(embedding, dtype='float32').reshape(1, -1)
        if vec.shape[1] != self.dim:
            raise ValueError("Embedding dim mismatch")
        with self.lock:
            vid = metadata.get('id')
            if vid is None:
                vid = max([int(k) for k in self.idmap.keys()]) + 1 if self.idmap else 0
            self.index.add_with_ids(vec, np.array([int(vid)], dtype='int64'))
            self.idmap[str(int(vid))] = metadata
            self._save()
        return int(vid)

    def search(self, query_embedding, k=5):
        q = np.array(query_embedding, dtype='float32').reshape(1, -1)
        with self.lock:
            if self.index.ntotal == 0:
                return []
            D, I = self.index.search(q, k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx != -1:
                m = self.idmap.get(str(int(idx)))
                if m:
                    x = m.copy()
                    x['distance'] = float(dist)
                    results.append(x)
        return results

    def _save(self):
        try:
            faiss.write_index(self.index, self.index_path)
        except Exception:
            pass
        tmp = self.idmap_path + ".tmp"
        with open(tmp,'w') as f:
            json.dump(self.idmap, f)
        os.replace(tmp, self.idmap_path)

    def get_vector_count(self):
        return int(self.index.ntotal)

    def close(self):
        self._save()
