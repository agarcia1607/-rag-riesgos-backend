# RAG Risk Analysis System

> **Demo:** https://rag-riesgos.vercel.app  
> **API:** https://rag-riesgos-backend-2.onrender.com/docs

Sistema de consulta inteligente sobre documentos de análisis de riesgos basado en **Retrieval-Augmented Generation (RAG)** con una arquitectura **baseline-first**, **grounded** y **reproducible**.

El proyecto está diseñado con foco en **ingeniería de sistemas de IA en producción**, priorizando control, trazabilidad y degradación segura por encima de la dependencia de modelos generativos.

---

## Principios de Diseño

**1. Independencia de LLMs** — el sistema no depende de modelos generativos para funcionar. Existe un modo determinístico completamente extractivo.

**2. LLMs como redactores** — cuando se utilizan LLMs, estos no deciden la evidencia. Solo redactan respuestas a partir del contexto recuperado.

**3. Conservadurismo ante incertidumbre** — si no existe evidencia clara en los documentos, el sistema prefiere abstenerse de responder.

**4. Degradación segura** — el sistema puede degradar de forma determinística sin cambiar el contrato de la API:

```
LLM remoto → Baseline determinístico
```

---

## Arquitectura

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
Top-10 Candidate Chunks
   │
   ▼
Cross-Encoder Reranker
   │
   ▼
Top-5 Chunks
   │
   ▼
Baseline Gate (no_evidence check)
   │
   ├── no_evidence=True → Abstención
   │
   ▼
Baseline Extractor / LLM Generator (Claude)
   │
   ▼
Evidence Gate (lexical + abstention check)
   │
   ▼
Answer or Abstention
```

### Componentes principales

**BM25 Retriever** — recuperación lexical robusta, alta precisión para consultas con términos exactos, robusto a dominios técnicos.

**Dense Retriever** — recuperación semántica basada en embeddings, captura similitud conceptual y mejora recall en preguntas parafraseadas.

**Hybrid Retriever** — combina BM25 y Dense Retrieval mediante fusión de scores, balanceando robustez lexical y generalización semántica.

**Cross-Encoder Reranker** — reordena los fragmentos recuperados para mejorar el orden de relevancia y eliminar ruido del top-k inicial.

**Baseline Extractor** — modo determinístico que extrae evidencia textual directamente, con cero consumo de tokens y latencia mínima.

**Evidence Gate** — antes de responder valida score mínimo de recuperación, presencia explícita de evidencia, cobertura léxica entre pregunta y contexto, y ausencia de patrones meta. Si las condiciones no se cumplen → abstención.

---

## Modos de Operación

### Baseline (default)

Modo determinístico sin LLMs. Extracción textual directa sobre los chunks recuperados.

```bash
RAG_MODE=baseline
```

Ventajas: determinístico, cero costo de tokens, latencia muy baja, ideal para producción estable.

### LLM — Claude (recomendado)

Retrieval híbrido + Claude Sonnet como redactor. El baseline actúa como gate: si no hay evidencia, Claude no es invocado.

```bash
RAG_MODE=llm
ANTHROPIC_API_KEY=sk-ant-...
```

Ventajas: respuestas en lenguaje natural, fallback automático al baseline ante errores, abstention accuracy 1.0 con gates configurados.

### Local

Retrieval híbrido + LLM local vía Ollama (ej. `qwen2.5:3b`). Sin dependencia externa, totalmente controlado.

```bash
RAG_MODE=local
```

---

## Gates Anti-Alucinación (modo LLM)

El modo LLM aplica tres gates en secuencia:

**Gate 1 — Baseline:** si el baseline retorna `no_evidence=True`, Claude no es invocado. Se reutilizan los gates extractivos del modo determinístico (patrones de email, teléfono, preguntas vagas, etc.).

**Gate 2 — Cobertura léxica:** si menos del 40% de los tokens clave de la pregunta aparecen en los chunks recuperados, se abstiene sin llamar a Claude.

**Gate 3 — Claude:** Claude puede retornar `NO_EVIDENCE` explícitamente. Si lo hace, la respuesta final es abstención.

---

## Evaluación

Dataset: `eval/v1.jsonl` — 50 preguntas (35 answerable, 15 unanswerable).

### Resultados comparativos

| Métrica               | Baseline | Claude (llm) |
|-----------------------|----------|--------------|
| Precision@5           | 0.029    | 0.029        |
| Recall@5              | 0.147    | 0.147        |
| Token hit rate        | 0.810    | 0.724        |
| All tokens hit        | 0.735    | 0.676        |
| Abstention accuracy   | 0.867    | **1.000**    |
| False no-evidence     | 0.029    | 0.171        |
| Latencia avg          | 451 ms   | 2 718 ms     |

El modo LLM prioriza **cero alucinaciones** sobre cobertura de respuesta. El trade-off es mayor false no-evidence rate — el sistema se abstiene en más preguntas answerable para garantizar que nunca responde incorrectamente en preguntas unanswerable.

---

## Quick Start

```bash
git clone <repo-url>
cd rag-riesgos

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configura el `.env`:

```bash
RAG_MODE=llm
ANTHROPIC_API_KEY=sk-ant-...
PDF_PATH=data/Doc chatbot.pdf
```

Levanta el backend:

```bash
uvicorn backend.main:app --port 8000
```

Levanta el frontend:

```bash
cd frontend && npm install && npm start
```

Backend disponible en `http://localhost:8000` — documentación automática en `http://localhost:8000/docs`.  
Frontend disponible en `http://localhost:3000`.

---

## Ejemplo de Consulta

```bash
curl -X POST https://rag-riesgos-backend-2.onrender.com/preguntar \
  -H "Content-Type: application/json" \
  -d '{"texto": "¿Cuál es la prima mínima por embarque?", "mode": "llm"}'
```

Respuesta:

```json
{
  "respuesta": "La prima mínima por embarque es de USD 12.-",
  "retrieved": [...],
  "no_evidence": false,
  "mode": "llm",
  "model": "claude-sonnet-4-5",
  "gate_reason": null,
  "baseline_version": "robust_v5_2026-03-09"
}
```

---

## Evaluación Automática

```bash
# Baseline
python eval/evaluate.py \
  --base_url http://127.0.0.1:8000 \
  --dataset eval/v1.jsonl \
  --modes baseline

# Claude
python eval/evaluate.py \
  --base_url http://127.0.0.1:8000 \
  --dataset eval/v1.jsonl \
  --modes llm \
  --timeout_s 120
```

Resultados guardados en `eval/runs/`. Experimentos registrados con MLflow en `mlruns/`.

---

## Estructura del Proyecto

```
backend/
├── main.py
├── query_wrapper.py
├── baseline_rag.py
├── local_rag.py
├── retrievers/
│   ├── bm25_retriever.py
│   ├── dense_retriever.py
│   ├── hybrid_retriever.py
│   └── reranker.py
├── services/
│   └── claude_generator.py
├── pdf_loader.py
└── config.py

eval/
├── v1.jsonl
├── evaluate.py
└── runs/

frontend/
└── src/

data/
mlruns/
```

---

## Despliegue

| Servicio | Plataforma | URL |
|----------|------------|-----|
| Backend  | Render     | https://rag-riesgos-backend-2.onrender.com |
| Frontend | Vercel     | https://rag-riesgos.vercel.app |

---

## Roadmap

**Completado**
- RAG baseline determinístico
- Hybrid retrieval (BM25 + Dense)
- Cross-encoder reranking
- API FastAPI con contrato estable
- Evaluación automática (50 preguntas)
- MLflow tracking
- Gates anti-alucinación (3 niveles)
- Integración Claude Sonnet (modo LLM)
- Frontend React con selector de modo
- Despliegue en Render + Vercel

**Futuro**
- RAGAS evaluation (faithfulness, answer relevancy)
- Observabilidad avanzada
- Caching de embeddings
- Optimización de recall en retrieval

---

## Limitaciones Conocidas

- Retrieval aún puede mejorar en recall (Precision@5 = 0.029).
- El modo LLM con gates conservadores tiene false no-evidence rate elevado (0.171).
- No existe razonamiento multi-documento complejo.

---

## Autor

Andrés García — Computer Science, Universidad Nacional de Colombia

## Licencia

MIT