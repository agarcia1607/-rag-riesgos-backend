# RAG de Análisis de Riesgos

Sistema de consulta inteligente sobre documentos de riesgos usando arquitectura **Retrieval‑Augmented Generation (RAG)** con enfoque **baseline‑first**, robusto y reproducible.

Diseñado para funcionar **sin dependencias de modelos generativos** y escalar opcionalmente a LLMs, manteniendo control, trazabilidad y estabilidad en producción.

---

##  Tabla de Contenidos

* [Objetivo](#-objetivo)
* [Características Principales](#-características-principales)
* [Arquitectura](#-arquitectura)
* [Estructura del Proyecto](#-estructura-del-proyecto)
* [Quick Start (Docker)](#-quick-start-docker)
* [Instalación](#-instalación)
* [Configuración](#-configuración)
* [Uso](#-uso)
* [Modos de Operación](#-modos-de-operación)
* [API Reference](#-api-reference)
* [Testing](#-testing)
* [Roadmap](#-roadmap)
* [Licencia](#-licencia)
* [Contribución](#-contribución)

---

##  Objetivo

Permitir consultas en lenguaje natural sobre documentos de riesgos (PDFs), entregando:

* **Respuestas claras y justificadas** con contexto relevante
* **Evidencia textual explícita** con referencias a fuentes
* **Comportamiento estable** incluso ante fallos de APIs externas
* **Trazabilidad completa** de cada respuesta generada

Este proyecto prioriza **ingeniería de sistemas de IA en producción**, no solo experimentación.

> **Filosofía de diseño**: control, reproducibilidad y degradación segura antes que dependencia de modelos externos.

---

##  Características Principales

###  Arquitectura Resiliente

* **Modo Baseline (default)**: BM25 + extracción extractiva (cero tokens)
* **Modo LLM (opcional)**: embeddings + Gemini con fallback automático
* **Degradación elegante**: si el LLM falla, el sistema continúa en baseline

###  Producción‑Ready

* Zero downtime por cuotas de API
* Respuestas determinísticas y reproducibles
* Logging estructurado
* Manejo robusto de errores

###  Transparencia

* Fuentes citadas explícitamente
* Scores de relevancia por fragmento
* Metadata completa (modo usado, latencia, chunks recuperados)

---

##  Arquitectura

```
┌─────────┐     ┌─────────┐     ┌──────────┐     ┌─────┐     ┌──────────┐
│  PDFs   │ ──▶ │ Ingesta │ ──▶ │ Indexing │ ──▶ │ RAG │ ──▶ │ Frontend │
└─────────┘     └─────────┘     └──────────┘     └─────┘     └──────────┘
                                       │              │
                                       ▼              ▼
                                 ┌──────────┐   ┌─────────┐
                                 │   BM25   │   │ Chroma  │
                                 │ Baseline │   │  (LLM)  │
                                 └──────────┘   └─────────┘
```

### Flujo de Consulta

1. Usuario envía pregunta en lenguaje natural
2. Query Wrapper decide el modo (baseline / llm / auto)
3. Retriever obtiene fragmentos relevantes
4. Generator produce respuesta (extractiva o generativa)
5. API devuelve respuesta + fuentes + metadata

---

##  Estructura del Proyecto

```
RAG_riegos/
│
├── backend/
│   ├── main.py
│   ├── query_wrapper.py
│   ├── baseline_rag.py
│   ├── baseline_store.py
│   ├── pdf_loader.py
│   ├── vector_store.py
│   └── config.py
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── Dockerfile
│   └── package.json
│
├── data/
│   └── documentos.pdf
│
├── chroma_db_riesgos/
├── baseline_index/
├── tests/
│
├── Dockerfile.backend
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

##  Quick Start (Docker)

```bash
docker compose up --build
```

* Backend: [http://localhost:8000](http://localhost:8000)
* Frontend: [http://localhost:3000](http://localhost:3000)
* Docs API: [http://localhost:8000/docs](http://localhost:8000/docs)

---

##  Instalación

### Prerrequisitos

* Python 3.11+
* Node.js 18+
* Docker + Docker Compose (recomendado)

### Backend (local)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Frontend (local)

```bash
cd frontend
npm install
npm start
```

---

##  Configuración

```env
RAG_MODE=baseline
GOOGLE_API_KEY=opcional
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
BACKEND_PORT=8000
FRONTEND_PORT=3000
```

---

##  Uso

### Backend

```bash
uvicorn backend.main:app --reload
```

### Frontend

```bash
npm start
```

---

##  Modos de Operación

### Baseline (default)

* BM25 + extracción textual
* Latencia < 100ms
* Determinístico y reproducible

```bash
export RAG_MODE=baseline
```

### LLM (opcional)

* Embeddings + Gemini
* Fallback automático

```bash
export RAG_MODE=llm
export GOOGLE_API_KEY=tu_key
```

---

##  API Reference

### `POST /query`

```json
{
  "question": "¿Cuáles son los riesgos de liquidez?",
  "mode": "auto"
}
```

### `GET /health`

```json
{
  "status": "healthy",
  "mode": "baseline"
}
```

---

##  Testing

Suite con **pytest**, baseline‑first y LLM‑agnóstica.

Cobertura:

* Healthcheck
* Contrato API
* Baseline determinístico
* Fallback automático
* Estabilidad y latencia
* Ingesta y chunking de PDFs

```bash
pytest -q
```

---

##  Roadmap

###  Completado

* Sistema RAG baseline
* API FastAPI
* Frontend React
* CI con GitHub Actions

###  En progreso

* Dockerización completa
* Tests de integración

###  Futuro

* Multi‑documento
* Evaluación automática (RAGAS)
* Observabilidad
* Autenticación

---

##  Licencia

MIT

---

##  Contribución

Pull requests bienvenidos.

```bash
git checkout -b feature/nueva-funcionalidad
git commit -m "feat: nueva funcionalidad"
git push origin feature/nueva-funcionalidad
```

