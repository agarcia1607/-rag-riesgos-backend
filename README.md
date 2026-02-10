# RAG de Análisis de Riesgos

Sistema de consulta inteligente sobre documentos de riesgos basado en **Retrieval‑Augmented Generation (RAG)** con una arquitectura **baseline‑first**, **grounded** y **reproducible**.

El proyecto está diseñado con foco en **ingeniería de sistemas de IA en producción**, priorizando control, trazabilidad y degradación segura por encima de la dependencia de modelos generativos.

---

## Tabla de Contenidos

* Principios de Diseño
* Arquitectura
* Flujo de Consulta
* Modos de Operación
* Transparencia y Grounding
* Quick Start
* Instalación
* Configuración
* API Reference
* Testing
* Estructura del Proyecto
* Roadmap
* Limitaciones Conocidas
* Contribución
* Licencia

---

## Principios de Diseño

El sistema se rige por los siguientes principios fundamentales:

* **Independencia de LLMs**: el sistema nunca depende exclusivamente de un modelo generativo.
* **LLMs como redactores**: los LLMs no deciden evidencia; solo redactan a partir del contexto recuperado.
* **Conservadurismo ante incertidumbre**: ante ambigüedad o falta de evidencia explícita, el sistema prefiere no responder.
* **Degradación segura y determinística**: ante fallos, el sistema degrada sin downtime y sin cambiar el contrato de salida.

---

## Arquitectura

```
┌─────────┐     ┌─────────┐     ┌──────────┐     ┌────────────┐
│  PDFs   │ ──▶ │ Ingesta │ ──▶ │ Retrieval│ ──▶ │  Generator │
└─────────┘     └─────────┘     └──────────┘     └────────────┘
                                       │               │
                                       ▼               ▼
                                 ┌──────────┐   ┌────────────┐
                                 │   BM25   │   │ LLM Local / │
                                 │ Baseline │   │   Remoto    │
                                 └──────────┘   └────────────┘
```

El **retrieval** es siempre explícito y controlado. El **generator** nunca introduce información externa al contexto.

---

## Flujo de Consulta

1. El usuario envía una pregunta en lenguaje natural.
2. El *Query Wrapper* selecciona el modo de operación.
3. BM25 recupera los fragmentos más relevantes del índice.
4. El generador:

   * extrae directamente texto (baseline), o
   * redacta usando exclusivamente el contexto recuperado (LLM).
5. La API devuelve:

   * respuesta,
   * fuentes textuales,
   * metadata completa de la ejecución.

---

## Modos de Operación

### Baseline (Default)

* **Retrieval**: BM25
* **Generación**: extracción textual directa
* **Ventajas**:

  * cero consumo de tokens,
  * completamente determinístico,
  * latencia menor a 100 ms,
  * sin dependencias externas.

Uso recomendado para producción estable.

```
RAG_MODE=baseline
```

---

### Local

* **Retrieval**: BM25
* **Generación**: LLM local vía Ollama (por ejemplo `qwen2.5:3b`)
* **Ventajas**:

  * sin conocimiento externo,
  * gates anti‑alucinación explícitos,
  * control total del entorno.

Requiere Ollama en ejecución.

```
RAG_MODE=local
```

---

### LLM Remoto (Opcional)

* **Retrieval**: embeddings + Chroma
* **Generación**: LLM externo (Gemini)
* **Ventajas**:

  * mayor capacidad generativa,
  * fallback automático a baseline.

```
RAG_MODE=llm
GOOGLE_API_KEY=tu_api_key
```

---

## Transparencia y Grounding

El sistema implementa múltiples mecanismos de validación:

### Gates Anti‑Alucinación

* score mínimo de retrieval (BM25),
* detección explícita de definiciones en texto,
* validación post‑LLM mediante solapamiento léxico,
* bloqueo de respuestas meta o de tipo disclaimer.

### Metadata Completa

Cada respuesta incluye información estructurada sobre:

* modo utilizado,
* latencia de ejecución,
* fragmentos recuperados,
* scores de relevancia,
* fuentes citadas explícitamente.

---

## Quick Start

### Docker (Recomendado)

```
git clone <repo-url>
cd RAG_riegos
docker compose up --build
```

Servicios disponibles:

* Backend: [http://localhost:8000](http://localhost:8000)
* Frontend: [http://localhost:3000](http://localhost:3000)
* Documentación API: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Instalación

### Prerrequisitos

* Python 3.11+
* Node.js 18+
* Docker + Docker Compose (opcional pero recomendado)

### Backend

```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Frontend

```
cd frontend
npm install
npm start
```

---

## Configuración

Variables de entorno principales:

```
RAG_MODE=baseline              # baseline | local | llm
GOOGLE_API_KEY=opcional
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
BACKEND_PORT=8000
FRONTEND_PORT=3000
```

---

## API Reference

### POST /preguntar

```
{
  "texto": "¿Cuáles son los tres niveles de riesgo?"
}
```

Respuesta:

```
{
  "respuesta": "Los tres niveles de riesgo son...",
  "fuentes": ["fragmento 1", "fragmento 2"],
  "mode": "baseline"
}
```

---

### GET /health

```
{
  "status": "healthy",
  "mode": "baseline"
}
```

---

## Testing

Ejecución de tests:

```
pytest
pytest -v
pytest --cov=backend
```

La suite cubre:

* contrato de la API,
* comportamiento determinístico del baseline,
* fallback automático,
* estabilidad y latencia,
* validación de gates anti‑alucinación.

---

## Estructura del Proyecto

```
RAG_riegos/
├── backend/
│   ├── main.py
│   ├── query_wrapper.py
│   ├── baseline_rag.py
│   ├── baseline_store.py
│   ├── local_rag.py
│   ├── ollama_client.py
│   ├── pdf_loader.py
│   └── config.py
├── frontend/
├── data/
├── tests/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Roadmap

### Completado

* RAG baseline con BM25
* Modo local grounded (BM25 + Ollama)
* API FastAPI
* Frontend React
* Suite de tests reproducible
* Containerización

### En progreso

* Tests de integración end‑to‑end
* Comparativa cuantitativa entre modos

### Planificado

* Evaluación automática (RAGAS)
* Observabilidad (métricas y logs)
* Cache de respuestas
* Autenticación

---

## Limitaciones Conocidas

Las siguientes limitaciones son **decisiones de diseño conscientes**:

* Solo se responden definiciones explícitas en el texto.
* El modo local requiere Ollama activo.
* Heurísticas conservadoras pueden descartar respuestas válidas en textos muy parafraseados.
* No hay razonamiento multi‑documento avanzado.
* La evaluación automática no está integrada por defecto.

---

## Contribución

Las contribuciones son bienvenidas siguiendo el flujo estándar:

```
git checkout -b feature/nueva-funcionalidad
git commit -m "feat: descripción"
git push origin feature/nueva-funcionalidad
```

Se utilizan **Conventional Commits**.

---

## Licencia

MIT
