# Retail Analysis RAG Demo

A local **agentic Retrieval-Augmented Generation (RAG)** analytics demo built with PyTorch and transformer models. It answers natural-language questions about retail business data by combining **semantic retrieval**, **deterministic analytics routing**, and **LLM-based response generation** — producing grounded, data-backed insights while reducing hallucination.

---

## Overview

Traditional LLMs answer business questions fluently but unreliably: they hallucinate numbers, invent trends, and can't be trusted with real metrics. This project addresses that by grounding every answer in actual retail data through a hybrid pipeline:

1. **Semantic retrieval** surfaces the most relevant pieces of business context for a query using transformer embeddings.
2. **Deterministic analytics routing** sends quantitative questions to real computation (exact figures, aggregations, trends) rather than letting the LLM guess.
3. **LLM-based generation** synthesizes the retrieved context and computed results into a clear, natural-language answer.

The result is an assistant that reasons over retail data the way an analyst would — fast, conversational, and grounded in real numbers.

---

## Architecture

```
Query
  │
  ├─► Semantic Retrieval ───► relevant business summaries (embedding similarity)
  │
  ├─► Analytics Routing ────► deterministic computation on structured data
  │
  └─► LLM Generation ───────► grounded natural-language response
```

The pipeline separates **what must be exact** (analytics, computed via code) from **what benefits from language fluency** (explanation, handled by the LLM), so the numbers are always correct and the delivery is always readable.

---

## Repository Structure

| File | Description |
|------|-------------|
| `retail_rag_demo.py` | Main application — orchestrates retrieval, analytics routing, and LLM response generation to answer retail queries. |
| `embed_summaries.py` | Generates transformer-based embeddings over business/data summaries for the semantic retrieval step. |
| `retail_rag_pipeline_dag.py` | Apache Airflow DAG (TaskFlow API) that orchestrates the end-to-end data and embedding pipeline, with isolated subprocess execution for the transformer embedding step. |
| `requirements.txt` | Python dependencies. |

---

## Key Features

- **Agentic RAG** — routes each query to the appropriate tool (retrieval vs. deterministic analytics) rather than relying on a single generation pass.
- **Grounded, low-hallucination answers** — quantitative results come from real computation, not the model's guesswork.
- **Transformer embeddings** — semantic retrieval over business summaries using HuggingFace models.
- **Orchestrated pipeline** — Apache Airflow TaskFlow DAGs coordinate data preparation and embedding, with the embedding step run in an isolated subprocess to keep dependencies and resources cleanly separated.
- **Local-first** — designed to run end-to-end on a local machine for demonstration and experimentation.

---

## Getting Started

### Prerequisites
- Python 3.9+
- (If using the LLM generation step) an OpenAI API key

### Installation

```bash
git clone https://github.com/jrobe187/Retail-Analysis-RAG-Demo.git
cd Retail-Analysis-RAG-Demo
pip install -r requirements.txt
```

### Configuration

If the demo uses the OpenAI API for response generation, set your key:

```bash
export OPENAI_API_KEY="your-key-here"
```

### Running the Demo

Generate embeddings over the business summaries:

```bash
python embed_summaries.py
```

Run the RAG demo and ask retail questions:

```bash
python retail_rag_demo.py
```

### Running the Pipeline (Airflow)

The Airflow DAG orchestrates the full data + embedding workflow:

```bash
# place retail_rag_pipeline_dag.py in your Airflow dags/ folder,
# then trigger it from the Airflow UI or CLI:
airflow dags trigger retail_rag_pipeline
```

---

## Tech Stack

- **Python**
- **PyTorch** & **HuggingFace Transformers** — embeddings and model inference
- **OpenAI API** — LLM response generation
- **Apache Airflow** — pipeline orchestration (TaskFlow DAGs)

---

## Notes

This is a demonstration project intended to showcase an agentic RAG architecture for business analytics — specifically the pattern of combining semantic retrieval with deterministic analytics routing to keep LLM-generated insights grounded in real data.

---

## Author

**Joseph Roberson** — [github.com/jrobe187](https://github.com/jrobe187)
