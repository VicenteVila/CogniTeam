# CogniTeam 🤖

[![Test](https://github.com/VicenteVila/CogniTeam/actions/workflows/test.yml/badge.svg)](https://github.com/VicenteVila/CogniTeam/actions/workflows/test.yml)

**Sistema multi-agente autónomo** con scoping inteligente de tareas, modelo de mundo, planificación dinámica, ejecución orquestada y trazabilidad completa de llamadas LLM.

Arquitectura basada en agentes especializados que colaboran secuencialmente para completar tareas complejas en 5+ dominios, desde Web Development hasta Game Development, con soporte experimental para 8 dominios adicionales.

## Product Story

CogniTeam es un framework de agentes LLM que **planifica, ejecuta y valida** tareas
complejas de forma autónoma.  A diferencia de chatbots o chains lineales, CogniTeam
tiene un modelo de mundo que evalúa qué información falta antes de planificar, un
planificador que selecciona few‑shots por dominio, y un orquestador con replanning
y grounding semántico que verifica que los entregables coincidan con lo prometido.

Está diseñado para brillar en **generación de UI, scripting, educación, diseño
gráfico y game‑dev**, donde la validación funcional (Playwright headless, tests de
integración) es tan importante como la generación misma.

---


## Flujo agentico completo

Cada tarea atraviesa 5 fases orquestadas:

```mermaid
flowchart TD
    U(("User"))
    U -->|Task| S["1. Scoping Agent<br/>classification + clarification"]
    S -->|TaskManifest| W["2. World Model<br/>gaps + confidence + grounding"]
    W -->|Constraints + Keywords| P["3. Planner Agent<br/>few-shot plan per domain"]
    P -->|Plan JSON| O["4. Orchestrator<br/>resolves vars, routes to agents,<br/>executes steps, validates outputs"]
    O --> R["5. Post-execution<br/>TraceForge report + save artifacts"]
    O -.-> DA["Developer Agent"]
    O -.-> UA["UI Designer Agent"]
    O -.-> DBA["Debugger Agent"]
    DA -.->|results| O
    UA -.->|results| O
    DBA -.->|results| O
```

### 1. Scoping Agent

El proceso comienza con el **Scoping Agent**, que analiza la tarea del usuario y:

1. **Clasifica** la tarea en hasta 3 dominios (principal + 2 secundarios) con nivel de confianza
2. **Selecciona arquetipos** específicos dentro de cada dominio (55 arquetipos total)
3. **Presenta la clasificación** al usuario y permite modificarla si no coincide
4. **Genera preguntas de clarificación** para resolver ambigüedades (stack tecnológico, audiencia, entregables, etc.)
5. **Produce un `TaskManifest`** con la tarea clarificada, parámetros recogidos y requisitos detallados

El usuario puede responder las preguntas o aceptar la clasificación. El sistema ajusta dinámicamente el plan según las respuestas.

### 2. World Model

El **World Model Agent** evalúa la tarea clarificada y genera:

- **Gaps de información**: qué datos faltan para completar la tarea
- **Resolubilidad**: qué herramientas del sistema pueden cubrir cada gap (`web_search_real`, `extract_info_from_text`, `browse_web_page`, etc.)
- **Confianza**: puntuación de 0-100 sobre la viabilidad de la tarea
- **Keywords de dominio**: términos relevantes para grounding semántico
- **Riesgos potenciales**: dependencias externas, APIs no disponibles, etc.

Si el World Model detecta un gap irresoluble, el sistema puede solicitar más información al usuario antes de planificar.

### 3. Planner Agent

El **Planner Agent** genera un plan JSON con pasos secuenciales:

1. Recibe la tarea clarificada y el contexto del World Model
2. Selecciona el **few-shot example** según el dominio detectado (5 plantillas: UI, Scripting, Education, Graphic Design, Game Dev)
3. Genera una lista de pasos con:
   - `tool_to_use`: herramienta específica del sistema
   - `inputs`: parámetros exactos de la tool, con referencias `{{output_variable}}` a pasos anteriores
   - `output_variable_name`: nombre para referenciar el resultado en pasos posteriores
   - `expected_output_format`: validación del tipo de output
4. Las variables se encadenan automáticamente: si el paso 2 extrae `{{paper_info}}`, el paso 3 puede usarlo en su `description`

### 4. Ejecución orquestada

El **Orchestrator** ejecuta cada paso del plan:

1. **Resolución de inputs**: `{{var}}` → valor real almacenado (soporta dot-path `{{var.sub}}` y embebido en texto)
2. **Routing**: asigna cada tool al agente correspondiente (DeveloperAgent, UIDesignerAgent, DebuggerAgent)
3. **Ejecución**: llama la tool con los inputs resueltos
4. **Validación**: verifica formato del output (JSON, HTML, text, etc.)
5. **Almacenamiento**: guarda el output en el `StepContext` para reutilización entre pasos
6. **Detección de errores**: si falla, activa el Debugger Agent para diagnóstico y corrección

**Agentes de ejecución:**

| Agente | Tools asignadas | Responsabilidad |
|--------|----------------|-----------------|
| **DeveloperAgent** | `web_search_real`, `extract_info_from_text`, `browse_web_page`, `write_file_sandboxed`, `read_file_sandboxed`, `execute_terminal_command_safe`, scripting tools, git tools, API calls, PDF, TTS | Ejecución general: buscar información, procesar datos, escribir archivos, ejecutar scripts, integrar con APIs externas |
| **UIDesignerAgent** | `generate_ui_code`, `generate_css_code`, `generate_js_code`, `combine_ui_to_html`, `generate_textual_artifact`, `analyze_html_js`, `fix_ui_code` | Generación de interfaz de usuario: HTML, CSS, JavaScript, artefactos textuales, depuración visual |
| **DebuggerAgent** | `analyze_html_js`, `fix_ui_code`, `read_file_sandboxed`, `write_file_sandboxed`, `generate_textual_artifact`, `extract_info_from_text`, scripting tools | Diagnóstico y corrección de errores en outputs generados |

### 5. Post-ejecución

- **TraceForge**: genera reporte HTML + Markdown con todas las llamadas LLM (modelo, tokens, latencia, errores)
- **`_save_result`**: mueve los archivos nuevos/modificados a `proyectos_finalizados/RUN_TIMESTAMP/artefactos/`
- **Reporte**: `reporte.md` con clasificación, plan ejecutado, outputs de cada paso y archivos generados

---

## Trazabilidad con TraceForge

Cada llamada LLM queda registrada automáticamente como un **span** de TraceForge:

```
LLM Call ──→ traceforge.span(agent="llm_planning", model="groq/llama-3.3-70b")
                ├── input_tokens: 2450
                ├── output_tokens: 890
                ├── duration_ms: 12340
                ├── error: None (o mensaje de error si falla)
                └── tags: ["task=planning", "domain=1_web_development"]
```

Al finalizar la ejecución se generan:
- `traceforge_report.html` — panel visual interactivo
- `traceforge_report.md` — reporte en Markdown

---

## Dominios

### Soportados (con few‑shot + world model dedicados)

| Dominio | Few‑Shot | World Model | Arquetipos |
|---------|----------|-------------|------------|
| **1. Web Development** | `FEWSHOT_UI` | Sí | landing‑page, ecommerce, saas‑dashboard, blog‑content, portfolio‑creative, corporate‑business |
| **2. Software Development** | `FEWSHOT_SCRIPT` | Sí | mobile‑apps, backend‑apis, databases, devops‑infra, machine‑learning |
| **3. Education** | `FEWSHOT_EDUCATION` | Sí | bootcamps, moocs, certifications, corporate‑training, k12‑curriculum |
| **4. Graphic Design** | `FEWSHOT_GRAPHIC_DESIGN` | Sí | branding‑identity, editorial‑layout, ui‑ux‑prototyping |
| **8. Game Development** | `FEWSHOT_GAME_DEV` | Sí | mobile‑casual, pc‑console, multiplayer‑online, vr‑ar, game‑tools |

### Experimentales (fallback genérico, sin instrumentación dedicada)

| # | Dominio |
|---|---------|
| 5 | Data Science |
| 6 | Content Writing |
| 7 | DevOps |
| 9 | Cybersecurity |
| 10 | Legal & Compliance |
| 11 | Architecture & Spatial |
| 12 | Marketing & Growth |
| 13 | Audiovisual & Management |

> Ver [docs/domain_consolidation.md](docs/domain_consolidation.md) para el detalle de la consolidación y criterios de promoción. 

---

## Instalación

```bash
git clone https://github.com/tuusuario/cogniteam.git
cd cogniteam
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Opcionales:
pip install -e ".[ml]"     # RAG con ChromaDB + FAISS
pip install -e ".[web]"    # Web scraping (BeautifulSoup4)
pip install git+https://github.com/VicenteVila/TraceForge.git

cp .env.example .env
# Editar .env con API keys
```

### Configuración (.env)

```ini
# Local (mínimo para funcionar):
use_ollama=True
ollama_base_url=http://localhost:11434
ollama_model_fast=gemma3:latest

# Cloud (razonamiento pesado):
use_groq=True
groq_api_key=gsk_tu_key_aqui

# Adicionales:
use_nvidia=True    # Embeddings
use_mistral=True   # Razonamiento alternativo
use_cerebras=True  # Inferencia rápida
```

---

## Uso

```bash
python main.py
```

El sistema te guiará por el flujo completo:

```
TAREA> Analiza ventas.csv, calcula el total por categoria
TAREA> y genera un grafico de barras como resultado.png
TAREA> FIN_TAREA

[Scoping Agent] Clasificando...
  ★ PRINCIPAL: 9_data_science.exploratory-analysis (92%)
    ¿Es correcta esta clasificación? (s/n): s

[World Model] Evaluando...
  Confianza: 85% | Keywords: csv, pandas, bar-plot, matplotlib

[Planificación] Generando plan...
  Pasos: 6

[Ejecución] ...
  Paso 1/6 (generate_textual_artifact) → script_generado ✅
  Paso 2/6 (write_file_sandboxed) → archivo_escrito ✅
  Paso 3/6 (execute_terminal_command_safe) → ejecucion ✅
  ...

✅ Flujo completado en 135.6s
  Resultados: proyectos_finalizados/RUN_20260710_131620/
```

### Output

```
proyectos_finalizados/
└── RUN_20260710_131620/
    ├── artefactos/
    │   ├── index.html
    │   ├── blog.md
    │   └── resultado.png
    ├── reporte.md
    ├── traceforge_report.html
    └── traceforge_report.md
```

---

## Stack técnico

| Capa | Tecnología |
|------|-----------|
| **Lenguaje** | Python 3.12+ — asyncio, pydantic, typing |
| **LLM Providers** | Groq SDK, Ollama API, litellm, OpenAI-compatible (NVIDIA, Mistral, Cerebras) |
| **Rate Limiting** | Circuit breaker con fallback automático entre providers (14.400 req/día Groq → Ollama local) |
| **Orquestación** | StepContext con resolución de variables `{{var}}`, soporte dot-path y embebido en texto |
| **Planificación** | Few-shot por dominio (5 plantillas: UI, Scripting, Education, Graphic Design, Game Dev), modelo de mundo para detección de gaps |
| **Scoping** | Clasificación multi-arquetipo con 55 arquetipos en 5+ dominios, clarificación interactiva |
| **Trazabilidad** | TraceForge — span capture por llamada LLM con modelo, tokens, latencia y errores |
| **Seguridad** | Sandboxing de rutas (no permite rutas absolutas fuera del proyecto), hidden tools, confirmación de comandos |
| **Tests** | pytest + asyncio |

---

## Licencia

MIT
