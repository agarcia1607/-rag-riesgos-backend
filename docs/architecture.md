# docs/architecture.md

# Arquitectura del sistema RAG

Este sistema implementa un pipeline RAG diseñado con enfoque **baseline-first**.

---

# Componentes principales

## Frontend

React

Interfaz para realizar consultas.

Se comunica con el backend mediante HTTP.

---

## Backend

FastAPI

Archivo principal:

backend/main.py

Endpoint principal:

POST /preguntar

---

# Query Wrapper

Archivo:

backend/query_wrapper.py

Responsable de:

recibir consultas

dirigirlas al modo adecuado

garantizar consistencia en la respuesta

Modos disponibles:

baseline

local (LLM)

---

# Baseline RAG

Archivo:

backend/baseline_rag.py

Responsable de:

retrieval

gating

generación extractiva

abstención

Versión actual:

robust_v5_2026-03-09

---

# Retrieval

Implementado en:

backend/retrievers/

Componentes:

BM25 Retriever

Dense Retriever

Hybrid Retriever

---

# Hybrid Retrieval

Combina:

BM25 (sparse)

Dense embeddings

Fórmula:

score = α * dense_score + (1 - α) * bm25_score

Configuración:

k = 5

alpha = 0.7

---

# Document ingestion

Fuente:

PDF

Loader:

pdf_loader.py

Chunking:

aprox. 200 chunks

Cada chunk incluye metadata:

source

page

chunk_id

---

# Evaluación

Script:

eval/evaluate.py

Dataset:

eval/v1.jsonl

Métricas calculadas:

precision@k

recall@k

abstention accuracy

must_include_token_hit_rate

latency

---

# Modos de ejecución

## Baseline

Sistema determinístico.

No usa LLM.

Respuesta extractiva basada en evidencia.

---

## Local

Usa LLM local mediante Ollama.

Se usa como fallback.

---

# Objetivo de la arquitectura

Diseñar un sistema RAG:

modular

evaluado

reproducible

capaz de evolucionar hacia sistemas generativos más complejos.
