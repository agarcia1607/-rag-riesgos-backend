# RAG Risk Analysis System

Sistema de consulta inteligente sobre documentos de análisis de riesgos basado en **Retrieval-Augmented Generation (RAG)** con una arquitectura **baseline-first**, **grounded** y **reproducible**.

El proyecto está diseñado con foco en **ingeniería de sistemas de IA en producción**, priorizando control, trazabilidad y degradación segura por encima de la dependencia total de modelos generativos.

---

# Principios de Diseño

El sistema sigue una filosofía conservadora para evitar alucinaciones.

## 1. Independencia de LLMs

El sistema **no depende de modelos generativos para funcionar**. Existe un modo determinístico completamente extractivo.

## 2. LLMs como redactores

Cuando se utilizan LLMs, estos **no deciden la evidencia**. Solo redactan respuestas a partir del contexto recuperado.

## 3. Conservadurismo ante incertidumbre

Si no existe evidencia clara en los documentos, el sistema **prefiere abstenerse de responder**.

## 4. Degradación segura

El sistema puede degradar de forma determinística:

LLM remoto → LLM local → Baseline determinístico

sin cambiar el contrato de la API.

---

# Arquitectura

Pipeline completo del sistema:

User Query
│
▼
Query Wrapper
│
▼
Hybrid Retrieval
(BM25 + Dense Embeddings)
│
▼
Top-10 Candidate Chunks
│
▼
Cross-Encoder Reranker
│
▼
Top-5 Chunks
│
▼
Baseline Extractor / LLM Generator
│
▼
Evidence Gate
│
▼
Answer or Abstention

---

# Componentes Principales

### BM25 Retriever

Recuperación lexical robusta basada en coincidencias de términos.

Ventajas:

* alta precisión para consultas con términos exactos
* robusto a dominios técnicos

---

### Dense Retriever

Recuperación semántica basada en embeddings.

Ventajas:

* captura similitud conceptual
* mejora recall en preguntas parafraseadas

---

### Hybrid Retriever

Combina BM25 y Dense Retrieval mediante fusión de scores.

Beneficios:

* robustez lexical
* generalización semántica

---

### Cross-Encoder Reranker

Un modelo cross-encoder reordena los fragmentos recuperados.

Objetivo:

* mejorar el orden de relevancia
* eliminar ruido del top-k inicial

Pipeline típico:

Hybrid retrieval → top-10 candidatos → reranking → top-5 finales

---

### Baseline Extractor

Modo determinístico que **extrae evidencia textual directamente**.

Características:

* cero consumo de tokens
* completamente reproducible
* latencia mínima

---

### Evidence Gate

Antes de responder, el sistema valida:

* score mínimo de recuperación
* presencia explícita de evidencia
* coincidencia léxica entre respuesta y contexto
* ausencia de patrones meta o disclaimers

Si las condiciones no se cumplen → **abstención**.

---

# Modos de Operación

## Baseline (Default)

Modo determinístico sin LLMs.

Retrieval:

Hybrid Retriever (BM25 + Dense)

Generación:

Extracción textual directa.

Ventajas:

* determinístico
* cero costo de tokens
* ideal para entornos productivos

Variable:

RAG_MODE=baseline

---

## Local Mode

Retrieval:

Hybrid Retriever

Generación:

LLM local vía Ollama.

Ejemplo de modelo:

qwen2.5:3b

Ventajas:

* sin dependencia externa
* totalmente controlado
* grounded en documentos

Variable:

RAG_MODE=local

---

## Remote LLM Mode (Opcional)

Retrieval:

Hybrid Retriever

Generación:

LLM remoto (ej: Gemini)

El sistema incluye **fallback automático al baseline**.

Variables:

RAG_MODE=llm
GOOGLE_API_KEY=your_api_key

---

# Flujo de Consulta

1. El usuario envía una pregunta en lenguaje natural.

2. El Query Wrapper determina el modo de operación.

3. El Hybrid Retriever recupera los fragmentos relevantes.

4. El Reranker reordena los candidatos.

5. El sistema genera respuesta o decide abstenerse.

6. La API devuelve:

* respuesta
* fragmentos recuperados
* metadata de ejecución

---

# Metadata de Respuesta

Cada respuesta incluye información para auditoría:

{
"respuesta": "...",
"retrieved": [...],
"no_evidence": false,
"baseline_version": "robust_v5"
}

Esto permite:

* reproducibilidad
* trazabilidad
* debugging del sistema

---

# Evaluación

Dataset:

eval/v1.jsonl

Contiene:

50 preguntas
35 answerable
15 unanswerable

---

# Resultados Experimentales

| Metric              | Value  |
| ------------------- | ------ |
| Precision@5         | 0.029  |
| Recall@5            | 0.147  |
| Must Include Tokens | 0.676  |
| Abstention Accuracy | 0.866  |
| False No Evidence   | 0.028  |
| Average Latency     | ~0.8 s |

El sistema prioriza **abstención segura** sobre respuestas incorrectas.

---

# Experimental Findings

Durante el desarrollo se realizaron múltiples experimentos.

Hallazgos principales:

* El reranking por sí solo no mejoró significativamente las métricas.
* La mayor mejora provino del **ajuste del chunking de documentos**.
* Reducir candidatos del reranker mejoró latencia sin degradar calidad.

Esto refleja un comportamiento común en sistemas RAG donde **la calidad del chunking impacta más que el modelo generativo**.

---

# Quick Start

Clonar repositorio

```
 git clone <repo-url>
 cd rag-riesgos
```

---

# Instalación

Crear entorno

```
 python -m venv .venv
 source .venv/bin/activate
```

Instalar dependencias

```
 pip install -r requirements.txt
```

---

# Ejecutar Backend

```
 uvicorn backend.main:app --port 8000
```

Servidor disponible en:

[http://localhost:8000](http://localhost:8000)

Documentación automática:

[http://localhost:8000/docs](http://localhost:8000/docs)

---

# Ejemplo de Consulta

Request

POST /preguntar

{
"texto": "¿Cuál es la prima mínima por embarque?",
"mode": "baseline"
}

Response

{
"respuesta": "...",
"retrieved": [...],
"no_evidence": false,
"baseline_version": "robust_v5"
}

---

# Evaluación Automática

Ejecutar benchmark:

```
 python eval/evaluate.py \
  --base_url http://127.0.0.1:8000 \
  --dataset eval/v1.jsonl \
  --modes baseline
```

Resultados se guardan en:

```
 eval/runs/
```

Experimentos pueden registrarse con MLflow.

---

# Estructura del Proyecto

backend/

main.py
query_wrapper.py
baseline_rag.py

retrievers/

bm25_retriever.py
dense_retriever.py
hybrid_retriever.py
reranker.py

services/

pdf_loader.py
config.py

eval/

data/

README.md

---

# Roadmap

## Completado

* RAG baseline determinístico
* Hybrid retrieval
* Cross-encoder reranking
* API FastAPI
* evaluación automática
* MLflow tracking
* gates anti-alucinación

## Futuro

* RAGAS evaluation
* observabilidad avanzada
* caching
* optimización de retrieval

---

# Limitaciones Conocidas

* retrieval aún puede mejorar en recall
* heurísticas conservadoras pueden descartar respuestas válidas
* no existe razonamiento multi-documento complejo

---

# Autor

Andrés García
Computer Science
Universidad Nacional de Colombia

---

# Licencia

MIT
