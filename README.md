# RAG Risk Analysis System

Sistema de consulta inteligente sobre documentos de análisis de riesgos basado en **Retrieval‑Augmented Generation (RAG)** con una arquitectura **baseline-first**, **grounded** y **reproducible**.

El proyecto está diseñado con foco en **ingeniería de sistemas de IA en producción**, priorizando control, trazabilidad y degradación segura por encima de la dependencia de modelos generativos.

---

# Principios de Diseño

El sistema sigue una filosofía conservadora para evitar alucinaciones.

## 1. Independencia de LLMs

El sistema no depende de modelos generativos para funcionar.

## 2. LLMs como redactores

Los modelos generativos no deciden la evidencia; únicamente redactan a partir del contexto recuperado.

## 3. Conservadurismo ante incertidumbre

Si no existe evidencia clara en los documentos, el sistema **prefiere abstenerse de responder**.

## 4. Degradación segura

El sistema puede degradar de forma determinística:

```
LLM remoto → LLM local → Baseline determinístico
```

sin cambiar el contrato de la API.

---

# Arquitectura

Pipeline simplificado:

```
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
Top-K Chunks
   │
   ▼
Baseline Extractor / LLM Generator
   │
   ▼
Answer or Abstention
```

Componentes principales:

* **BM25** para recuperación lexical robusta
* **Dense embeddings** para similitud semántica
* **Hybrid retriever** que combina ambos
* **Baseline extractivo determinístico**
* **LLM opcional para redacción**

---

# Flujo de Consulta

1. El usuario envía una pregunta en lenguaje natural.

2. El **Query Wrapper** selecciona el modo de operación.

3. El **Hybrid Retriever** recupera los fragmentos relevantes.

4. El sistema decide:

   * responder usando evidencia explícita
   * o abstenerse si no existe evidencia suficiente.

5. La API devuelve:

* respuesta
* fragmentos recuperados
* metadata de ejecución

---

# Modos de Operación

## Baseline (Default)

Modo **determinístico y sin LLMs**.

Retrieval

```
Hybrid Retriever
BM25 + Dense Embeddings
```

Generación

```
Extracción textual directa
```

Ventajas:

* cero consumo de tokens
* determinístico
* latencia muy baja
* ideal para producción estable

```
RAG_MODE=baseline
```

---

## Local Mode

Retrieval:

```
Hybrid Retriever
```

Generación:

```
LLM local vía Ollama
```

Ejemplo de modelo:

```
qwen2.5:3b
```

Ventajas:

* sin conocimiento externo
* grounded en documentos
* control total del entorno

```
RAG_MODE=local
```

---

## Remote LLM Mode (Opcional)

Retrieval:

```
Hybrid Retriever
```

Generación:

```
LLM remoto (ej: Gemini)
```

El sistema incluye **fallback automático al baseline**.

```
RAG_MODE=llm
GOOGLE_API_KEY=your_api_key
```

---

# Transparencia y Grounding

El sistema implementa múltiples mecanismos para evitar alucinaciones.

## Gates Anti-Alucinación

El sistema valida:

* score mínimo de retrieval
* presencia explícita de evidencia textual
* solapamiento léxico entre respuesta y contexto
* bloqueo de respuestas meta o disclaimers

---

# Metadata de Respuesta

Cada respuesta incluye:

```
{
  "respuesta": "...",
  "retrieved": [...],
  "no_evidence": false,
  "baseline_version": "robust_v5_2026-03-09"
}
```

Esto permite auditoría completa del sistema.

---

# Evaluación

Dataset:

```
eval/v1.jsonl
```

Contiene:

```
50 preguntas
35 answerable
15 unanswerable
```

Resultados actuales:

| Metric                 | Value |
| ---------------------- | ----- |
| Precision@5            | 0.029 |
| Recall@5               | 0.147 |
| Abstention Accuracy    | 0.93  |
| False No Evidence Rate | 0.028 |
| Average Latency        | 97 ms |

El sistema prioriza **abstención segura** sobre respuestas incorrectas.

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

```
http://localhost:8000
```

Documentación API:

```
http://localhost:8000/docs
```

---

# Ejemplo de Consulta

Request

```
POST /preguntar
```

```
{
  "texto": "¿Cuál es la prima mínima por embarque?",
  "mode": "baseline"
}
```

Response

```
{
  "respuesta": "...",
  "retrieved": [...],
  "no_evidence": false,
  "baseline_version": "robust_v5_2026-03-09"
}
```

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

---

# Estructura del Proyecto

```
backend/
│
├── main.py
├── query_wrapper.py
├── baseline_rag.py
│
├── retrievers/
│   ├── bm25_retriever.py
│   ├── dense_retriever.py
│   └── hybrid_retriever.py
│
├── services/
│
├── pdf_loader.py
├── config.py
│
eval/
scripts/
data/

README.md
```

---

# Roadmap

## Completado

* RAG baseline determinístico
* Hybrid retrieval
* API FastAPI
* evaluación automática
* gates anti-alucinación

## En progreso

* evaluación comparativa entre modos
* integración MLflow

## Planificado

* RAGAS evaluation
* reranking con cross-encoder
* observabilidad
* caching

---

# Limitaciones Conocidas

* retrieval aún puede mejorar en recall
* heurísticas conservadoras pueden descartar respuestas válidas
* no hay razonamiento multi-documento avanzado

---

# Autor

**Andrés García**
Computer Science
Universidad Nacional de Colombia

---

# Licencia

MIT
