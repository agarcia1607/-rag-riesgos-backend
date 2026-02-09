# RAG de Análisis de Riesgos

Sistema de consulta inteligente sobre documentos de riesgos basado en **Retrieval-Augmented Generation (RAG)** con arquitectura baseline-first, grounded y reproducible.

## Descripción

El sistema permite consultas en lenguaje natural sobre documentos de riesgos (PDFs), priorizando **ingeniería de sistemas de IA en producción** sobre la mera experimentación con modelos.

### Características Clave

- **Respuestas fundamentadas**: Solo usa evidencia documental explícita
- **Trazabilidad completa**: Fragmentos textuales con scores de relevancia
- **Resiliencia ante fallos**: Degradación elegante sin downtime
- **Sin dependencias críticas de LLMs**: Funciona sin conexión a APIs externas

---

## Tabla de Contenidos

- [Principios de Diseño](#principios-de-diseño)
- [Arquitectura](#arquitectura)
- [Quick Start](#quick-start)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Modos de Operación](#modos-de-operación)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Roadmap](#roadmap)
- [Limitaciones Conocidas](#limitaciones-conocidas)
- [Contribución](#contribución)
- [Licencia](#licencia)

---

## Principios de Diseño

El sistema se rige por los siguientes principios fundamentales:

1. **Independencia de LLMs**: El sistema nunca depende exclusivamente de un LLM
2. **LLMs como redactores**: Los LLMs no deciden evidencia, solo redactan a partir del contexto recuperado
3. **Conservadurismo ante incertidumbre**: Ante ambigüedad o falta de evidencia, el sistema prefiere no responder
4. **Degradación segura**: La degradación ante fallos es segura y determinística

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

### Flujo de Consulta

1. El usuario envía una pregunta en lenguaje natural
2. El Query Wrapper selecciona el modo de operación
3. BM25 recupera fragmentos relevantes del índice
4. El generador extrae directamente (baseline) o redacta usando solo el contexto (LLM)
5. La API devuelve la respuesta, las fuentes y la metadata completa

### Modos de Operación

#### 🔷 Baseline (Default)
- **Retrieval**: BM25
- **Generación**: Extracción textual directa
- **Ventajas**: Cero consumo de tokens, determinístico, latencia < 100ms
- **Uso**: Producción estable, sin dependencias externas

#### 🔷 Local
- **Retrieval**: BM25
- **Generación**: LLM local (Ollama, ej: `qwen2.5:3b`)
- **Ventajas**: Sin conocimiento externo, gates anti-alucinación
- **Uso**: Redacción mejorada manteniendo control local

#### 🔷 LLM Remoto (Opcional)
- **Retrieval**: Embeddings + Chroma
- **Generación**: LLM externo (Gemini)
- **Ventajas**: Mayor capacidad generativa
- **Fallback**: Automático a baseline ante fallas

### Transparencia y Grounding

El sistema implementa múltiples mecanismos de validación:

**Gates Anti-Alucinación**
- Score mínimo de retrieval
- Detección explícita de definiciones en el texto
- Validación post-LLM (overlap léxico)
- Bloqueo de meta-respuestas

**Metadata Completa**
- Modo utilizado
- Latencia de respuesta
- Chunks recuperados con scores
- Fuentes citadas explícitamente

---

## Quick Start

### Con Docker (Recomendado)

```bash
# Clonar el repositorio
git clone <repo-url>
cd RAG_riegos

# Iniciar todos los servicios
docker compose up --build
```

**URLs de acceso:**
- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- Documentación API: http://localhost:8000/docs

---

## Instalación

### Prerrequisitos

- Python 3.11+
- Node.js 18+
- Docker + Docker Compose (recomendado)

### Instalación Local

#### Backend

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
```

#### Frontend

```bash
cd frontend
npm install
```

---

## Configuración

### Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

```env
# Modo de operación
RAG_MODE=baseline              # baseline | local | llm

# API Keys (opcional para modo LLM)
GOOGLE_API_KEY=tu_api_key_aqui

# Configuración de chunking
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Configuración de retrieval
TOP_K=5

# Puertos
BACKEND_PORT=8000
FRONTEND_PORT=3000
```

### Configuración por Modo

| Variable | Baseline | Local | LLM Remoto |
|----------|----------|-------|------------|
| `RAG_MODE` | `baseline` | `local` | `llm` |
| `GOOGLE_API_KEY` | No requiere | No requiere | **Requerido** |
| Ollama | No requiere | **Requerido** | No requiere |

---

## Modos de Operación

### Modo Baseline

**Ideal para producción estable**

```bash
export RAG_MODE=baseline
uvicorn backend.main:app --reload
```

**Características:**
- ✅ Sin consumo de tokens
- ✅ 100% determinístico
- ✅ Latencia < 100ms
- ✅ Sin dependencias externas

### Modo Local

**Redacción mejorada con control local**

```bash
# Iniciar Ollama (en terminal separada)
ollama serve

# Descargar modelo (primera vez)
ollama pull qwen2.5:3b

# Iniciar backend
export RAG_MODE=local
uvicorn backend.main:app --reload
```

**Características:**
- ✅ Evidencia únicamente de BM25
- ✅ Sin conocimiento externo del LLM
- ✅ Gates anti-alucinación explícitos
- ⚠️ Requiere Ollama en ejecución

### Modo LLM Remoto

**Mayor capacidad generativa (opcional)**

```bash
export RAG_MODE=llm
export GOOGLE_API_KEY=tu_key
uvicorn backend.main:app --reload
```

**Características:**
- ✅ Embeddings semánticos (Chroma)
- ✅ LLM potente (Gemini)
- ✅ Fallback automático a baseline
- ⚠️ Requiere API key válida
- ⚠️ Consumo de tokens

---

## API Reference

### Endpoints

#### `POST /preguntar`

Envía una consulta al sistema.

**Request Body:**
```json
{
  "texto": "¿Cuáles son los tres niveles de riesgo?"
}
```

**Response:**
```json
{
  "respuesta": "Los tres niveles de riesgo son...",
  "fuentes": [
    {
      "texto": "Fragmento relevante del documento...",
      "score": 0.85,
      "metadata": {
        "source": "documentos.pdf",
        "page": 5
      }
    }
  ],
  "metadata": {
    "modo": "baseline",
    "latencia_ms": 87,
    "chunks_recuperados": 5
  }
}
```

#### `GET /health`

Verifica el estado del sistema.

**Response:**
```json
{
  "status": "healthy",
  "mode": "baseline",
  "timestamp": "2025-02-09T10:30:00Z"
}
```

### Códigos de Estado

| Código | Descripción |
|--------|-------------|
| 200 | Consulta exitosa |
| 400 | Solicitud mal formada |
| 404 | Endpoint no encontrado |
| 500 | Error interno del servidor |

---

## Testing

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/test_baseline.py

# Con cobertura
pytest --cov=backend --cov-report=html

# Modo verbose
pytest -v
```

### Cobertura de Tests

La suite de tests incluye:

- ✅ Healthcheck del sistema
- ✅ Contrato del endpoint `/preguntar`
- ✅ Comportamiento determinístico baseline
- ✅ Fallback automático ante fallos
- ✅ Estabilidad y latencia
- ✅ Ingesta y chunking de PDFs
- ✅ Validación de gates anti-alucinación

**Características de los tests:**
- Baseline-first: No dependen de LLMs
- LLM-agnósticos: Funcionales en cualquier modo
- Reproducibles: Resultados consistentes en CI/CD

---

## Estructura del Proyecto

```
RAG_riegos/
│
├── backend/                    # Backend FastAPI
│   ├── main.py                # Punto de entrada de la API
│   ├── query_wrapper.py       # Orquestador de modos
│   ├── baseline_rag.py        # Implementación baseline
│   ├── baseline_store.py      # Índice BM25
│   ├── local_rag.py           # Implementación local (Ollama)
│   ├── ollama_client.py       # Cliente Ollama
│   ├── pdf_loader.py          # Carga y chunking de PDFs
│   └── config.py              # Configuración centralizada
│
├── frontend/                   # Frontend React
│   ├── src/                   # Código fuente
│   ├── public/                # Archivos estáticos
│   ├── Dockerfile             # Imagen Docker
│   └── package.json           # Dependencias npm
│
├── data/                       # Documentos fuente
│   └── documentos.pdf         # PDFs de riesgos
│
├── chroma_db_riesgos/         # Base vectorial (modo LLM)
├── baseline_index/            # Índice BM25 (baseline/local)
│
├── tests/                      # Suite de tests
│   ├── test_baseline.py
│   ├── test_api.py
│   └── test_integration.py
│
├── Dockerfile.backend          # Imagen Docker backend
├── docker-compose.yml          # Orquestación de servicios
├── requirements.txt            # Dependencias Python
├── .env.example               # Plantilla de configuración
└── README.md                  # Este archivo
```

---

## Roadmap

### ✅ Completado

- [x] RAG baseline con BM25
- [x] Modo local grounded (BM25 + Ollama)
- [x] API FastAPI con endpoints documentados
- [x] Frontend React para consultas
- [x] CI/CD con GitHub Actions
- [x] Suite de tests reproducible
- [x] Containerización con Docker

### 🚧 En Progreso

- [ ] Tests de integración end-to-end
- [ ] Comparativa de rendimiento entre modos
- [ ] Documentación de arquitectura detallada

### 📋 Planificado

- [ ] Evaluación automática con RAGAS
- [ ] Soporte multi-documento avanzado
- [ ] Sistema de observabilidad (métricas, logs)
- [ ] Autenticación y autorización
- [ ] Cache de respuestas frecuentes
- [ ] Interfaz de administración

---

## Limitaciones Conocidas

Las siguientes limitaciones son **decisiones de diseño conscientes**, priorizando control y seguridad sobre cobertura máxima:

1. **Definiciones explícitas**: El sistema no infiere definiciones implícitas. Una definición debe aparecer textualmente en el documento.

2. **Dependencia de Ollama (modo local)**: El modo local requiere que Ollama esté en ejecución en el entorno.

3. **Heurística conservadora**: El grounding es deliberadamente conservador y puede descartar respuestas válidas en documentos muy parafraseados.

4. **Sin razonamiento multi-documento**: No se realiza razonamiento avanzado que combine información de múltiples documentos.

5. **Sin evaluación automática integrada**: La evaluación automática (ej: RAGAS) está planificada pero no implementada por defecto.

---

## Contribución

Las contribuciones son bienvenidas. Por favor, sigue este flujo:

```bash
# Fork del repositorio y clonación
git clone https://github.com/tu-usuario/RAG_riegos.git
cd RAG_riegos

# Crear rama para nueva funcionalidad
git checkout -b feature/nueva-funcionalidad

# Realizar cambios y commits
git add .
git commit -m "feat: descripción de la nueva funcionalidad"

# Push y Pull Request
git push origin feature/nueva-funcionalidad
```

### Guía de Commits

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` Nueva funcionalidad
- `fix:` Corrección de bugs
- `docs:` Cambios en documentación
- `test:` Añadir o modificar tests
- `refactor:` Refactorización de código
- `chore:` Tareas de mantenimiento

---

## Licencia

Este proyecto está licenciado bajo la **Licencia MIT**.

---

## Soporte

Para preguntas, problemas o sugerencias:

- **Issues**: [GitHub Issues](https://github.com/tu-usuario/RAG_riegos/issues)
- **Documentación API**: http://localhost:8000/docs (cuando el servidor esté en ejecución)

---

**Desarrollado con enfoque en producción, reproducibilidad y control.**
