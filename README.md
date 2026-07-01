# CogniTeam 🤖

Sistema multi-agente autónomo con memoria híbrida, scoping inteligente de tareas y soporte multi-LLM (local + cloud).

Clasifica tareas en **13 dominios** y **55 arquetipos**, genera preguntas de clarificación, y ejecuta planes multi-paso usando agentes especializados (planner, developer, UI designer, debugger).

---

## Arquitectura

```
                    ┌─────────────────────────┐
                    │     Scoping Agent        │
                    │  (clasifica + clarifica)  │
                    └──────┬──────────────────┘
                           │ tarea clarificada
                           ▼
┌─────────────────────────────────────────────┐
│              Orchestrator                    │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Planner │─▶│Developer │─▶│  Debugger   │  │
│  │ Agent   │  │ Agent    │  │  Agent      │  │
│  └─────────┘  └──────────┘  └────────────┘  │
│                    │                         │
│              ┌─────┴─────┐                   │
│              │ UI Agent  │                   │
│              └───────────┘                   │
└─────────────────────────────────────────────┘
       │                            │
       ▼                            ▼
┌──────────────┐          ┌──────────────────┐
│  Memoria     │          │  Herramientas     │
│  H-MEM       │          │  filesystem, git  │
│  GraphRAG    │          │  terminal, web,   │
│  Skills      │          │  UI, scripting    │
│  MATM        │          │  integraciones    │
│  Fast-Slow   │          └──────────────────┘
└──────────────┘
```

### Componentes

| Componente | Rol |
|------------|-----|
| **Scoping Agent** | Clasifica la tarea en dominio/arquetipo, hace preguntas hasta entenderla, genera un `TaskManifest` |
| **Planner Agent** | Genera un plan multi-paso a partir de la tarea clarificada |
| **Developer Agent** | Ejecuta pasos del plan (escribir archivos, correr comandos, llamar APIs) |
| **UI Designer Agent** | Genera código HTML/CSS/JS |
| **Debugger Agent** | Diagnostica y corrige errores en outputs |
| **Memoria** | 5 módulos: H-MEM (grafos), GraphRAG (RAG), Skills (habilidades aprendidas), MATM (episódica), Fast-Slow (refuerzo) |

### Soporte multi-LLM

| Provider | Uso | Modelo por defecto |
|----------|-----|-------------------|
| **Groq** (cloud) | Razonamiento pesado (70B) | `llama-3.3-70b-versatile` |
| **Ollama** (local) | Clasificación rápida | `gemma3:latest` |
| **Ollama** (local) | Código | `deepseek-coder:6.7b` |
| **Groq** (cloud) | Extracción/clasificación | `llama-3.1-8b-instant` |

El sistema tiene **rate limiting** automático: cuando Groq alcanza su límite gratuito (14.400 req/día), fallback a Ollama local sin errores.

---

## Requisitos

- **Python 3.11+**
- **Ollama** (para modelos locales) — opcional si usas Groq para todo
- **Groq API key** (gratis, sin tarjeta) — opcional si usas solo Ollama
- **16GB RAM** recomendado (funciona con 8GB usando modelos pequeños)

---

## Instalación

```bash
# 1. Clonar
git clone https://github.com/tuusuario/cogniteam.git
cd cogniteam

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# 3. Instalar dependencias
pip install -e .

# 4. Configurar (editar .env)
cp .env.example .env
```

### Configuración mínima (.env)

```ini
# Para usar solo Ollama (100% local):
use_ollama=True
ollama_base_url=http://localhost:11434
ollama_model_fast=gemma3:latest

# Para añadir Groq (razonamiento cloud, opcional):
use_groq=True
groq_api_key=gsk_tu_key_aqui
```

### Modelos recomendados para Ollama

```bash
ollama pull gemma3:latest        # Clasificación (3.3GB)
ollama pull deepseek-coder:6.7b  # Código (3.8GB)
ollama pull qwen3:8b             # Razonamiento (5.2GB)
```

---

## Uso

```bash
python main.py
```

Escribe tu tarea y termina con `FIN_TAREA`:

```
Introduce la tarea (multilínea, 'FIN_TAREA' en línea nueva para terminar):
TAREA> Crea una landing page para mi negocio de reformas
TAREA> que muestre servicios, antes/después con imágenes,
TAREA> formulario de contacto y presupuesto automático.
TAREA> FIN_TAREA
```

### Ejemplo: clasificación multi-arquetipo

```
[Scoping Agent] Analizando tarea...

  ★ PRINCIPAL: 1_web_development.landing-page (conf: 85%)
     La tarea requiere una landing page para captar clientes

  SECUNDARIO: 2_software_development.mobile-apps (conf: 70%)
     La tarea requiere una app para gestionar proyectos

  SECUNDARIO: 5_video_production.video-production (conf: 60%)
     La tarea requiere crear videos para redes sociales
```

### Output

Los resultados se guardan en `proyectos_finalizados/RUN_YYYYMMDD_HHMMSS/`:

```
proyectos_finalizados/
└── RUN_20260701_140416/
    ├── artefactos/
    │   ├── index.html
    │   ├── styles.css
    │   └── script.js
    └── reporte.md
```

---

## Dominios soportados (13)

| # | Dominio | Ejemplos |
|---|---------|----------|
| 1 | Web Development | landing-page, ecommerce, saas-dashboard |
| 2 | Software Development | api-development, mobile-apps, desktop-apps |
| 3 | 3D Design | 3d-archviz, 3d-product, 3d-animation |
| 4 | Graphic Design | ui-ux, branding, video-production |
| 5 | Video Production | video-production, motion-graphics |
| 6 | Marketing & Growth | campaign-launch, seo-sem, social-media |
| 7 | Software Architecture | system-design, microservices, security |
| 8 | Game Development | game-development, game-assets |
| 9 | Data Science | data-pipeline, ml-model, analytics |
| 10 | Content Writing | copywriting, blog-article, documentation |
| 11 | DevOps & Infrastructure | cloud-infrastructure, ci-cd, monitoring |
| 12 | Cybersecurity | security-audit, pentesting, compliance |
| 13 | Legal Compliance | legal-compliance, privacy, contracts |

---

## Tests

```bash
pytest tests/ -v
```

---

## Stack técnico

- **Python 3.12** — asyncio, pydantic, typing
- **LLM Providers** — Groq SDK, Ollama API, litellm (fallback)
- **Memoria** — networkx, sentence-transformers, chromadb, faiss
- **Orquestación** — arquitectura propia basada en contexto compartido
- **Seguridad** — sandboxing de rutas, confirmación de comandos, rate limiting

---

## Licencia

MIT
