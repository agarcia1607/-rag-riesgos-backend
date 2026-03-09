# docs/baseline_v5.md

# Baseline RAG v5

Version: **robust_v5_2026-03-09**

Este documento describe la baseline actual del sistema RAG de análisis de riesgos.

---

# Arquitectura

Pipeline:

Query
↓
Hybrid Retrieval (BM25 + Dense)
↓
Top-K chunks
↓
Gating (evidence check)
↓
Extractive answer
↓
Abstention si no hay evidencia

---

# Retrieval

Tipo: **Hybrid Retriever**

Componentes:

BM25 (sparse retrieval)

Dense embeddings
Modelo:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Combinación:

score = α * dense + (1 - α) * bm25

Configuración actual:

k = 5
alpha = 0.7

---

# Dataset de evaluación

Archivo:

eval/v1.jsonl

Contenido:

50 preguntas totales

Tipos:

answerable

unanswerable

Cada pregunta incluye:

query

must_include_tokens

expected behavior

---

# Métricas

Resultado de evaluación:

n = 50
k = 5
retrieval_eligible_n = 34

avg_precision_at_k = 0.029
avg_recall_at_k = 0.147

abstention_accuracy = 0.93
false_no_evidence_rate = 0.028

must_include_token_hit_rate = 0.76
must_include_all_hit_rate = 0.62

latency_ms_avg ≈ 97
latency_ms_p95 ≈ 385

---

# Comportamiento del sistema

El sistema puede:

Responder cuando existe evidencia en los documentos.

Abstenerse cuando no existe evidencia suficiente.

Evitar alucinaciones mediante gating determinístico.

---

# Endpoint principal

POST /preguntar

Ejemplo:

{
"text": "¿Cuál es la prima mínima por embarque?",
"mode": "baseline"
}

Respuesta:

{
"respuesta": "...",
"retrieved": [...],
"no_evidence": false,
"gate_reason": "...",
"baseline_version": "robust_v5_2026-03-09"
}

---

# Objetivo de la baseline

Establecer un sistema RAG:

estable

medible

reproducible

antes de introducir modelos generativos o reranking.

---

# Mejoras futuras

Posibles mejoras:

Cross-encoder reranker

mejor chunking

query expansion

evaluación RAGAS

MLflow tracking


