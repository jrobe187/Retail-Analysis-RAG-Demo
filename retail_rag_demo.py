import re
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

# ─────────────────────────────────────────────
# 1. Generate Synthetic Retail Sales Dataset
# ─────────────────────────────────────────────

random.seed(42)
REGIONS = ["North", "South", "East", "West"]
PRODUCTS = ["Widget A", "Widget B", "Widget C", "Gadget X", "Gadget Y"]


def generate_retail_data(n_days=90):
    rows = []
    start_date = datetime(2024, 1, 1)
    for day_offset in range(n_days):
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
                    "ProfitMargin": profit_margin
                })
    return pd.DataFrame(rows)


df = generate_retail_data()
print(f"Dataset shape: {df.shape}")
print(df.head())


# ─────────────────────────────────────────────
# 2. Build Text Summaries for Retrieval
# ─────────────────────────────────────────────

grouped = df.groupby(["Product", "Region"]).agg(
    TotalUnits=("Units", "sum"),
    TotalRevenue=("Revenue", "sum"),
    AvgMargin=("ProfitMargin", "mean")
).reset_index()


def make_summary(row):
    return (
        f"For product {row['Product']} in region {row['Region']}, "
        f"total units sold are {int(row['TotalUnits'])}, "
        f"total revenue is ${row['TotalRevenue']:.2f}, "
        f"and the average profit margin is {row['AvgMargin']:.2f}."
    )


grouped["Summary"] = grouped.apply(make_summary, axis=1)
print(f"\nKnowledge base: {grouped.shape[0]} summaries")
print(grouped[["Product", "Region", "Summary"]].head())


# ─────────────────────────────────────────────
# 3. Embeddings (in-memory, no vector DB)
# ─────────────────────────────────────────────

texts = grouped["Summary"].tolist()
metas = grouped[
    ["Product", "Region"] + (["Quarter"] if "Quarter" in grouped.columns else [])
].to_dict(orient="records")

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_emb = embed_model.encode(texts, convert_to_numpy=True).astype("float32")
doc_norm = doc_emb / np.maximum(np.linalg.norm(doc_emb, axis=1, keepdims=True), 1e-12)

print("\nEmbeddings shape:", doc_norm.shape)


# ─────────────────────────────────────────────
# 4. Retriever
# ─────────────────────────────────────────────

def retrieve_summaries_local(query: str, k: int = 5):
    """Semantic retrieval with optional hard filters (region / product / quarter)."""
    ql = query.lower()
    region = next((r for r in REGIONS if r.lower() in ql), None)
    product = next((p for p in PRODUCTS if p.lower() in ql), None)

    quarter = None
    if "Quarter" in grouped.columns:
        quarters = sorted(grouped["Quarter"].unique().tolist())
        quarter = next((qt for qt in quarters if str(qt).lower() in ql), None)

    cand_idx = list(range(len(texts)))
    if region:
        cand_idx = [i for i in cand_idx if metas[i]["Region"] == region]
    if product:
        cand_idx = [i for i in cand_idx if metas[i]["Product"] == product]
    if quarter and "Quarter" in metas[0]:
        cand_idx = [i for i in cand_idx if metas[i].get("Quarter") == quarter]
    if not cand_idx:
        cand_idx = list(range(len(texts)))  # fallback to global search

    sub_emb = doc_norm[cand_idx]
    q = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    scores = (sub_emb @ q.T).ravel()
    top = np.argsort(-scores)[:k]
    return [(texts[cand_idx[i]], metas[cand_idx[i]], float(scores[i])) for i in top]


# ─────────────────────────────────────────────
# 5. Local LLM (Flan-T5-base)
# ─────────────────────────────────────────────

#llm = pipeline("text2text-generation", model="google/flan-t5-base", device_map="auto")

device = 0 if torch.cuda.is_available() else -1

print(f"Loading LLM on device: {'CUDA/GPU' if device == 0 else 'CPU'}...")
llm = pipeline(
    "text2text-generation", 
    model="google/flan-t5-base", 
    device=device
)


def call_llm(prompt: str, max_new_tokens=140, temperature=0.2):
    out = llm(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.25,
        num_return_sequences=1,
    )
    return out[0]["generated_text"].strip()


# ─────────────────────────────────────────────
# 6. Deterministic Helpers (for precise queries)
# ─────────────────────────────────────────────

def has_col(df, name):
    return name in df.columns


def slice_grouped(region=None, quarter=None):
    g = grouped.copy()
    if region:
        g = g[g["Region"].str.lower() == region.lower()]
    if quarter and has_col(g, "Quarter"):
        g = g[g["Quarter"] == quarter]
    return g


def top_by_units(region=None, quarter=None):
    g = slice_grouped(region, quarter)
    if has_col(g, "Quarter"):
        g = g.groupby(["Product", "Region"], as_index=False)["TotalUnits"].sum()
    if g.empty:
        return None
    row = g.loc[g["TotalUnits"].idxmax()]
    return {"product": row["Product"], "region": row["Region"],
            "value": float(row["TotalUnits"]), "metric": "units"}


def top_by_revenue(region=None, quarter=None):
    g = slice_grouped(region, quarter)
    if has_col(g, "Quarter"):
        g = g.groupby(["Product", "Region"], as_index=False)["TotalRevenue"].sum()
    if g.empty:
        return None
    row = g.loc[g["TotalRevenue"].idxmax()]
    return {"product": row["Product"], "region": row["Region"],
            "value": float(row["TotalRevenue"]), "metric": "revenue"}


def top_by_total_profit(region=None, quarter=None):
    g = slice_grouped(region, quarter)
    if has_col(g, "Quarter"):
        g = g.groupby(["Product", "Region"], as_index=False).agg(
            TotalRevenue=("TotalRevenue", "sum"),
            AvgMargin=("AvgMargin", "mean")
        )
    if g.empty:
        return None
    gp = g.assign(TotalProfit=g["TotalRevenue"] * g["AvgMargin"])
    row = gp.loc[gp["TotalProfit"].idxmax()]
    return {"product": row["Product"], "region": row["Region"],
            "value": float(row["TotalProfit"]), "metric": "total_profit"}


# ─────────────────────────────────────────────
# 7. Prompt Builder
# ─────────────────────────────────────────────

ANALYST_SYSTEM_PROMPT = (
    "You are a senior retail data analyst. Use only the CONTEXT if provided.\n"
    "Write two short sentences that answer the QUESTION, then a third line starting with 'Next action:'.\n"
    "Do not copy the context verbatim. Do not invent products or numbers."
)


def build_prompt(query: str, retrieved):
    if not retrieved:
        return f"{ANALYST_SYSTEM_PROMPT}\n\nQUESTION: {query}\nANSWER:"
    ctx = "\n".join([f"- {doc}" for (doc, _meta, _s) in retrieved[:4]])
    return f"{ANALYST_SYSTEM_PROMPT}\n\nCONTEXT:\n{ctx}\n\nQUESTION: {query}\nANSWER:"


# ─────────────────────────────────────────────
# 8. Agent Router
# ─────────────────────────────────────────────

KEYWORDS_RETRIEVE = {
    "trend", "growth", "revenue", "profit", "sales",
    "region", "product", "quarter", "margin", "units"
}


def needs_retrieval(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in KEYWORDS_RETRIEVE)


def parse_region_quarter(q: str):
    ql = q.lower()
    region = next((r for r in REGIONS if r.lower() in ql), None)
    quarter = None
    if has_col(grouped, "Quarter"):
        quarters = sorted(grouped["Quarter"].unique().tolist())
        quarter = next((qt for qt in quarters if str(qt).lower() in ql), None)
    return region, quarter


def agentic_answer(q: str, k: int = 4):
    ql = q.lower()
    region, quarter = parse_region_quarter(q)

    # --- Deterministic routes ---
    if "highest units" in ql or ("most" in ql and "units" in ql):
        best = top_by_units(region=region, quarter=quarter)
        stmt = (
            f"{best['product']} has the highest units in {best['region']} with {best['value']:.0f} units."
            if best else "No matching records found."
        )
        prompt = (
            "Rewrite STATEMENT as one concise sentence, "
            "then add a second line starting with 'Next action:' with one follow-up analysis.\n\n"
            f"STATEMENT: {stmt}\nANSWER:"
        )
        return call_llm(prompt, max_new_tokens=90)

    if "highest revenue" in ql or ("most" in ql and "revenue" in ql):
        best = top_by_revenue(region=region, quarter=quarter)
        stmt = (
            f"{best['product']} has the highest revenue in {best['region']} at ${best['value']:,.2f}."
            if best else "No matching records found."
        )
        prompt = (
            "Rewrite STATEMENT as one concise sentence, then add 'Next action:' with a concrete follow-up.\n\n"
            f"STATEMENT: {stmt}\nANSWER:"
        )
        return call_llm(prompt, max_new_tokens=90)

    if "most profitable" in ql or ("highest" in ql and "profit" in ql):
        best = top_by_total_profit(region=region, quarter=quarter)
        stmt = (
            f"{best['product']} has the highest total profit in {best['region']} (${best['value']:,.2f})."
            if best else "No matching records found."
        )
        prompt = (
            "Rewrite STATEMENT as one concise sentence, then add 'Next action:' with a concrete follow-up.\n\n"
            f"STATEMENT: {stmt}\nANSWER:"
        )
        return call_llm(prompt, max_new_tokens=90)

    # --- Default: classic RAG (retrieve → LLM) ---
    retrieved = retrieve_summaries_local(q, k=k) if needs_retrieval(q) else []

    if retrieved:
        print("[Agent] Retrieval results:")
        for (doc, meta, score) in retrieved[:3]:
            tag = f"{meta.get('Product')} | {meta.get('Region')}"
            if "Quarter" in meta:
                tag += f" | {meta.get('Quarter')}"
            print(f"  - {tag} (cos={score:.3f})")

    prompt = build_prompt(q, retrieved)
    return call_llm(prompt)


# ─────────────────────────────────────────────
# 9. Demo Queries
# ─────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        "List the one product with the highest units sold",
        "Which product has the highest revenue?",
        "What is the most profitable product?",
        "How are sales trending in the North region?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        print(f"A: {agentic_answer(query)}")
