import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from airflow.decorators import dag, task


REGIONS = ["North", "South", "East", "West"]
PRODUCTS = ["Widget A", "Widget B", "Widget C", "Gadget X", "Gadget Y"]


@dag(
    dag_id="retail_rag_pipeline",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["rag", "retail", "portfolio"],
)
def retail_rag_pipeline():

    @task
    def generate_data() -> list:
        """Extract step: generate synthetic retail sales data."""
        random.seed(42)
        rows = []
        start_date = datetime(2024, 1, 1)
        for day_offset in range(90):
            date = start_date + timedelta(days=day_offset)
            for region in REGIONS:
                for product in PRODUCTS:
                    units = max(0, int(random.gauss(50, 20)))
                    price = random.uniform(20, 120)
                    revenue = round(units * price, 2)
                    profit_margin = round(random.uniform(0.1, 0.4), 2)
                    rows.append({
                        "Date": date.strftime("%Y-%m-%d"),
                        "Region": region,
                        "Product": product,
                        "Units": units,
                        "Revenue": revenue,
                        "ProfitMargin": profit_margin,
                    })
        df = pd.DataFrame(rows)
        print(f"[generate_data] shape={df.shape}")
        return df.to_dict(orient="records")

    @task
    def build_summaries(records: list) -> list:
        """Transform step: aggregate and build natural-language summaries."""
        df = pd.DataFrame(records)
        grouped = df.groupby(["Product", "Region"]).agg(
            TotalUnits=("Units", "sum"),
            TotalRevenue=("Revenue", "sum"),
            AvgMargin=("ProfitMargin", "mean"),
        ).reset_index()

        def make_summary(row):
            return (
                f"For product {row['Product']} in region {row['Region']}, "
                f"total units sold are {int(row['TotalUnits'])}, "
                f"total revenue is ${row['TotalRevenue']:.2f}, "
                f"and the average profit margin is {row['AvgMargin']:.2f}."
            )

        grouped["Summary"] = grouped.apply(make_summary, axis=1)
        print(f"[build_summaries] {grouped.shape[0]} summaries built")
        return grouped.to_dict(orient="records")

    @task
    def compute_embeddings(grouped_records: list) -> dict:
        """Load step: embed summaries for semantic retrieval."""
        from sentence_transformers import SentenceTransformer

        texts = [r["Summary"] for r in grouped_records]
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        doc_emb = embed_model.encode(texts, convert_to_numpy=True).astype("float32")
        doc_norm = doc_emb / np.maximum(
            np.linalg.norm(doc_emb, axis=1, keepdims=True), 1e-12
        )

        print(f"[compute_embeddings] embeddings shape={doc_norm.shape}")

        out_path = "/tmp/retail_doc_embeddings.npy"
        np.save(out_path, doc_norm)
        return {"embeddings_path": out_path, "n_docs": len(texts)}

    raw_records = generate_data()
    summaries = build_summaries(raw_records)
    compute_embeddings(summaries)


retail_rag_pipeline()
