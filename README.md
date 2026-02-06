#  RAG de Análisis de Riesgos

Sistema de consulta inteligente sobre documentos de riesgos usando arquitectura **Retrieval-Augmented Generation (RAG)** con enfoque baseline-first, robusto y reproducible.

Diseñado para funcionar **sin dependencias de modelos generativos** y escalar opcionalmente a LLMs, manteniendo control, trazabilidad y estabilidad en producción.

---

## 📋 Tabla de Contenidos

- [Objetivo](#-objetivo)
- [Características Principales](#-características-principales)
- [Arquitectura](#-arquitectura)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)
- [Modos de Operación](#-modos-de-operación)
- [API Reference](#-api-reference)
- [Roadmap](#-roadmap)
- [Contribución](#-contribución)

---

##  Objetivo

Permitir consultas en lenguaje natural sobre documentos de riesgos (PDFs), entregando:

-  **Respuestas claras y justificadas** con contexto relevante
-  **Evidencia textual explícita** con referencias a fuentes
-  **Comportamiento estable** incluso ante fallos de APIs externas
-  **Trazabilidad completa** de cada respuesta generada

Este proyecto prioriza **ingeniería de sistemas de IA en producción**, no solo experimentación.

---

##  Características Principales

###  Arquitectura Resiliente
- **Modo Baseline** (predeterminado): BM25 + extracción extractiva sin uso de tokens
- **Modo LLM** (opcional): Gemini + embeddings semánticos con fallback automático
- **Degradación elegante**: Si LLM falla, el sistema continúa funcionando en modo baseline

###  Producción-Ready
- Zero downtime por cuotas de API
- Respuestas determinísticas y reproducibles
- Logging estructurado y métricas de rendimiento
- Manejo robusto de errores

###  Transparencia
- Fuentes citadas explícitamente
- Scores de relevancia por fragmento
- Metadata de cada respuesta (modo usado, latencia, chunks recuperados)

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

1. **Usuario** envía pregunta en lenguaje natural
2. **Query Wrapper** determina modo (baseline/LLM)
3. **Retriever** obtiene chunks relevantes (BM25 o vectorial)
4. **Generator** produce respuesta (extractiva o generativa)
5. **API** devuelve respuesta + fuentes + metadata

---

##  Estructura del Proyecto
```
RAG_riesgos/
│
├── backend/
│   ├── main.py                # FastAPI server + endpoints
│   ├── query_wrapper.py       # Orquestador RAG (modo selector)
│   ├── baseline_rag.py        # Motor extractivo (BM25)
│   ├── baseline_store.py      # Índice BM25 + persistencia
│   ├── Pdf_loader.py          # Procesamiento y chunking de PDFs
│   ├── Vector_store.py        # Embeddings + ChromaDB (modo LLM)
│   └── config.py              # Configuración centralizada
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx
│   │   │   └── SourcesPanel.jsx
│   │   ├── App.jsx
│   │   └── index.js
│   └── package.json
│
├── data/
│   └── Doc chatbot.pdf        # Documento(s) fuente
│
├── chroma_db_riesgos/         # Vector DB (solo modo LLM)
├── baseline_index/            # Índice BM25 persistido
│
├── tests/                     # Tests unitarios + integración
├── .env.example
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

##  Instalación

### Prerrequisitos

- Python 3.11+
- Node.js 18+ y npm
- (Opcional) Docker y Docker Compose

### Backend
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/RAG_riesgos.git
cd RAG_riesgos

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu configuración
```

### Frontend
```bash
cd frontend
npm install
```

---

##  Configuración

### Variables de Entorno (`.env`)
```bash
# Modo de operación (baseline | llm)
RAG_MODE=baseline

# API Keys (solo para modo LLM)
GOOGLE_API_KEY=tu_api_key_aqui

# Configuración de chunking
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Configuración de retrieval
TOP_K=5
MIN_SCORE=0.3

# Configuración de servidor
BACKEND_PORT=8000
FRONTEND_PORT=3000
```

---

##  Uso

### Iniciar Backend
```bash
# Desarrollo
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Producción
uvicorn backend.main:app --workers 4 --host 0.0.0.0 --port 8000
```

La API estará disponible en:
- **Aplicación**: `http://localhost:8000`
- **Documentación interactiva**: `http://localhost:8000/docs`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

### Iniciar Frontend
```bash
cd frontend
npm start
```

Interfaz disponible en: `http://localhost:3000`

### Docker (Recomendado para Producción)
```bash
# Construir y levantar servicios
docker-compose up --build

# Solo backend
docker-compose up backend

# Detener servicios
docker-compose down
```

---

##  Modos de Operación

###  Modo Baseline (Predeterminado)

**Características:**
- Búsqueda léxica con BM25
- Extracción de frases relevantes
- **Cero consumo de tokens**
- Latencia < 100ms
- 100% reproducible

**Cuándo usar:**
- Entornos de producción estables
- Cumplimiento regulatorio estricto
- Restricciones de presupuesto
- Documentos técnicos/legales donde la cita exacta es crítica

**Activación:**
```bash
export RAG_MODE=baseline
```

###  Modo LLM (Opcional)

**Características:**
- Embeddings semánticos (Google Gemini)
- Búsqueda vectorial con ChromaDB
- Respuestas generativas contextuales
- **Fallback automático** a baseline si hay errores

**Cuándo usar:**
- Consultas complejas que requieren síntesis
- Usuarios no técnicos
- Disponibilidad de presupuesto para APIs

**Activación:**
```bash
export RAG_MODE=llm
export GOOGLE_API_KEY=tu_api_key
```

**Manejo de Errores:**
```
LLM Request
    │
    ├─ Success ──▶ Respuesta generativa
    │
    └─ Error (429/Quota/Network)
            │
            └──▶ Automatic Fallback ──▶ Baseline Response
```

---

##  API Reference

### `POST /query`

Procesa una consulta en lenguaje natural.

**Request:**
```json
{
  "question": "¿Cuáles son los riesgos de liquidez?",
  "mode": "auto"
}
```

**Response:**
```json
{
  "answer": "Los riesgos de liquidez identificados son...",
  "sources": [
    {
      "content": "Fragmento relevante del documento...",
      "page": 5,
      "score": 0.89,
      "metadata": {"document": "Doc chatbot.pdf"}
    }
  ],
  "metadata": {
    "mode_used": "baseline",
    "latency_ms": 87,
    "chunks_retrieved": 5,
    "fallback_triggered": false
  }
}
```

### `POST /ingest`

Procesa nuevos documentos PDF.

**Request:**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@documento.pdf"
```

### `GET /health`

Verifica estado del sistema.

**Response:**
```json
{
  "status": "healthy",
  "mode": "baseline",
  "index_loaded": true,
  "documents_count": 1
}
```

---

##  Roadmap

###  Completado
-  Sistema RAG baseline funcional
-  API REST con FastAPI
-  Frontend React
-  Modo LLM con fallback
-  Documentación completa

###  En Progreso
-  Dockerización completa
-  Tests de integración (>80% coverage)
-  CI/CD pipeline

###  Futuro
-  Soporte multi-documento (colecciones)
-  Sistema de evaluación automática (RAGAS)
-  Panel administrativo
-  Autenticación y permisos
-  Caché de consultas frecuentes
-  Soporte para más formatos (DOCX, TXT, HTML)
-  Integración con S3 para almacenamiento
-  Métricas y observabilidad (Prometheus + Grafana)

---


##  Licencia

[MIT](LICENSE)

---

##  Contribución

Pull requests bienvenidos. Para cambios mayores, por favor abrir un issue primero.
```bash
# Fork y clonar
git checkout -b feature/nueva-funcionalidad
git commit -m "Agrega nueva funcionalidad"
git push origin feature/nueva-funcionalidad
```

---

**¿Preguntas?** Abre un issue o contacta al equipo.
