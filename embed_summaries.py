"""Standalone embedding step.

Run as its own process (e.g. invoked by an Airflow task via subprocess) so the
transformer model loads in a clean interpreter, avoiding fork-related crashes
when launched from inside an orchestrator on macOS.

Reads grouped summary records from a JSON file, encodes them with a
sentence-transformer model, and saves normalized embeddings to a .npy file.
"""

import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer


def main(input_json_path: str, output_npy_path: str) -> None:
    with open(input_json_path) as f:
        grouped_records = json.load(f)

    texts = [r["Summary"] for r in grouped_records]
    print(f"[embed] loaded {len(texts)} summaries", flush=True)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    print("[embed] model loaded", flush=True)

    doc_emb = model.encode(texts, convert_to_numpy=True).astype("float32")
    doc_norm = doc_emb / np.maximum(
        np.linalg.norm(doc_emb, axis=1, keepdims=True), 1e-12
    )

    np.save(output_npy_path, doc_norm)
    print(f"[embed] saved embeddings shape={doc_norm.shape} -> {output_npy_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python embed_summaries.py <input_json> <output_npy>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
