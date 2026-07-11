import json
import random
import re
import time
from typing import Any, Dict, List, Optional

import traceforge

from cogniteam.config.settings import settings
from cogniteam.tools.utils.llm import llm_complete


WORLD_MODEL_PROMPT = """Eres la capa "World Model" dentro del Planner Agent de CogniTeam.

Tu trabajo NO es ejecutar la tarea, sino SIMULAR la trayectoria futura si se sigue el plan,
identificar gaps en el estado actual, y estimar la probabilidad de éxito.

TAREA: "{task}"
DOMINIO: {domain}
ARQUETIPO: {archetype}

Responde ÚNICAMENTE JSON válido (sin markdown, sin texto extra):
{{
  "trajectory": "Roadmap abstracto de 3-5 pasos sobre cómo se desarrollaría la ejecución...",
  "gap": "Qué información crítica falta en el estado actual para ejecutar el plan...",
  "confidence": 0-100 (entero, estimación realista de éxito),
  "keywords": ["5", "términos", "clave", "que", "deberían", "aparecer", "en", "el", "output"],
  "analysis": "Análisis breve del riesgo y viabilidad"
}}
"""

WEB_DEV_WORLD_MODEL_PROMPT = """Eres la capa "World Model" dentro del Planner Agent de CogniTeam.

Tu trabajo NO es ejecutar la tarea, sino SIMULAR la trayectoria futura si se sigue el plan,
identificar gaps en el estado actual, y estimar la probabilidad de éxito.

REGLAS CRÍTICAS:
1. NO SPOILERS: Usa placeholders abstractos (ej. "definir paleta de colores"), NO valores específicos de la tarea a menos que se hayan dado explícitamente.
2. SÉ ESPECÍFICO: En la Trajectory, indica hitos concretos, evita frases genéricas como "hacer el trabajo".
3. CALIBRACIÓN DE CONFIANZA: Sé honesto. Si la tarea carece de inputs críticos, la confianza debe ser baja.
4. KEYWORDS: Extrae hasta 5 términos que DEBEN aparecer en la ejecución real para considerar el计划 anclado.

FORMATO OUTPUT (JSON estricto):
{{
  "trajectory": "Roadmap paso a paso...",
  "gap": "Por qué el estado actual es insuficiente...",
  "confidence": 0-100 (entero),
  "keywords": ["keyword1", "keyword2", ...],
  "analysis": "Evaluación del plan..."
}}

--- EJEMPLOS ---

Ejemplo 1 (Arquetipo: landing-page)
Tarea: "Create a landing page for my renovation business showing services, before/after images, contact form and automatic budget."
{{
  "trajectory": "1) Scaffold semantic HTML structure (hero, services grid, before/after gallery, contact section, footer). 2) Implement responsive CSS with brand variables. 3) Build contact form with client-side validation. 4) Add budget calculator logic in JS. 5) Validate accessibility and Core Web Vitals.",
  "gap": "The current state lacks brand color palette, logo assets, and copy text. The budget logic requires pricing rules that have not been provided.",
  "confidence": 70,
  "keywords": ["html", "responsive", "contact-form", "budget-calculator", "accessibility"],
  "analysis": "The path is direct but requires clarification on branding and pricing rules. Without them, the landing page will be structurally correct but commercially incomplete."
}}

Ejemplo 2 (Arquetipo: ecommerce)
Tarea: "Build an online store for handmade candles with Stripe payments."
{{
  "trajectory": "1) Define product catalog schema and database models. 2) Build product listing and detail pages. 3) Implement shopping cart session management. 4) Integrate Stripe checkout with webhook confirmation. 5) Add order dashboard for the admin.",
  "gap": "No product images, descriptions, or Stripe API keys are present. Tax rules and shipping zones are undefined.",
  "confidence": 55,
  "keywords": ["stripe", "cart", "webhook", "product-schema", "order-dashboard"],
  "analysis": "High uncertainty due to missing payment credentials and logistics rules. The technical path is clear, but business logic blocks are unresolved."
}}

Ejemplo 3 (Arquetipo: saas-dashboard)
Tarea: "Create a SaaS analytics dashboard with user auth and real-time charts."
{{
  "trajectory": "1) Set up authentication flow (JWT/OAuth). 2) Design dashboard layout with sidebar navigation. 3) Integrate charting library for real-time data visualization. 4) Build API endpoints for data aggregation. 5) Implement role-based access control.",
  "gap": "The current state does not specify the data source, chart types required, or user roles. OAuth provider is not defined.",
  "confidence": 60,
  "keywords": ["jwt", "dashboard", "charting", "api-endpoints", "rbac"],
  "analysis": "The architecture is standard, but the absence of data schema and auth provider specifics introduces integration risk."
}}

--- TAREA A SIMULAR ---
Domain: {domain}
Archetype: {archetype}
Task Description: {task_description}
Interaction History: {history}

Genera el JSON del world model ahora.
"""

SOFTWARE_DEV_WORLD_MODEL_PROMPT = """Eres la capa "World Model" dentro del Planner Agent de CogniTeam.

Tu trabajo NO es ejecutar la tarea, sino SIMULAR la trayectoria futura si se sigue el plan,
identificar gaps en el estado actual, y estimar la probabilidad de éxito.

REGLAS CRÍTICAS:
1. NO SPOILERS: Usa placeholders abstractos (ej. "definir arquitectura del backend"), NO valores específicos de la tarea a menos que se hayan dado explícitamente.
2. SÉ ESPECÍFICO: En la Trajectory, indica hitos concretos, evita frases genéricas como "hacer el trabajo".
3. CALIBRACIÓN DE CONFIANZA: Sé honesto. Si la tarea carece de inputs críticos, la confianza debe ser baja.
4. KEYWORDS: Extrae hasta 5 términos que DEBEN aparecer en la ejecución real para considerar el plan anclado.

FORMATO OUTPUT (JSON estricto):
{{
  "trajectory": "Roadmap paso a paso...",
  "gap": "Por qué el estado actual es insuficiente...",
  "confidence": 0-100 (entero),
  "keywords": ["keyword1", "keyword2", ...],
  "analysis": "Evaluación del plan..."
}}

--- EJEMPLOS ---

Ejemplo 1 (Arquetipo: backend-apis)
Tarea: "Design a RESTful API for a ride-sharing platform with driver/passenger matching."
{{
  "trajectory": "1) Define data models for users, rides, payments. 2) Design endpoints for auth, ride CRUD, matching engine. 3) Implement request validation with Pydantic/Zod. 4) Add rate limiting and CORS configuration. 5) Document with OpenAPI spec.",
  "gap": "No concurrency model for matching algorithm nor payment gateway choice provided. Driver geolocation precision requirements unspecified.",
  "confidence": 60,
  "keywords": ["rest-api", "validation", "auth", "endpoints", "openapi"],
  "analysis": "Standard REST architecture is well-understood but matching logic and payment integration introduce significant complexity without specification."
}}

Ejemplo 2 (Arquetipo: mobile-apps)
Tarea: "Build a cross-platform fitness tracker app with workout logging and progress charts."
{{
  "trajectory": "1) Scaffold Flutter/React Native project with navigation skeleton. 2) Implement local DB schema for workouts and exercises. 3) Build workout logging UI with form validation. 4) Integrate charting library for progress visualization. 5) Add offline sync and background notifications.",
  "gap": "No design mockups, exercise catalog, or target platform OS versions specified. Wearable device integration not defined.",
  "confidence": 50,
  "keywords": ["flutter", "local-db", "charts", "offline-sync", "notifications"],
  "analysis": "Feasible but lacks design and exercise domain inputs. Offline-first approach is correct but multiplies scope without clear requirements."
}}

Ejemplo 3 (Arquetipo: databases)
Tarea: "Design a transactional database schema for an inventory management system."
{{
  "trajectory": "1) Analyze inventory domain entities and relationships. 2) Design normalized schema with indexes for performance. 3) Define stored procedures for stock reconciliation. 4) Set up connection pooling and backup strategy. 5) Document schema with migration scripts.",
  "gap": "Inventory valuation method (FIFO/LIFO) not specified. Expected transaction volume and concurrency requirements unknown.",
  "confidence": 65,
  "keywords": ["schema", "indexes", "migrations", "normalization", "transactions"],
  "analysis": "Schema design is straightforward but valuation method and volume assumptions could require significant rework if mismatched."
}}

--- TAREA A SIMULAR ---
Domain: {domain}
Archetype: {archetype}
Task Description: {task_description}
Interaction History: {history}

Genera el JSON del world model ahora.
"""

EDUCATION_WORLD_MODEL_PROMPT = """Eres la capa "World Model" dentro del Planner Agent de CogniTeam.

Tu trabajo NO es ejecutar la tarea, sino SIMULAR la trayectoria futura si se sigue el plan,
identificar gaps en el estado actual, y estimar la probabilidad de éxito.

REGLAS CRÍTICAS:
1. NO SPOILERS: Usa placeholders abstractos (ej. "definir estructura del curso"), NO valores específicos de la tarea a menos que se hayan dado explícitamente.
2. SÉ ESPECÍFICO: En la Trajectory, indica hitos concretos, evita frases genéricas como "hacer el trabajo".
3. CALIBRACIÓN DE CONFIANZA: Sé honesto. Si la tarea carece de inputs críticos, la confianza debe ser baja.
4. KEYWORDS: Extrae hasta 5 términos que DEBEN aparecer en la ejecución real para considerar el plan anclado.

FORMATO OUTPUT (JSON estricto):
{{
  "trajectory": "Roadmap paso a paso...",
  "gap": "Por qué el estado actual es insuficiente...",
  "confidence": 0-100 (entero),
  "keywords": ["keyword1", "keyword2", ...],
  "analysis": "Evaluación del plan..."
}}

--- EJEMPLOS ---

Ejemplo 1 (Arquetipo: bootcamps)
Tarea: "Create a 12-week full-stack bootcamp platform with cohort management and milestone tracking."
{{
  "trajectory": "1) Define curriculum modules and weekly milestones. 2) Build cohort creation and student enrollment flow. 3) Implement progress tracking with automated checkpoints. 4) Integrate video infrastructure for recorded lectures. 5) Add grading dashboard and certificate generation.",
  "gap": "No syllabus content, instructor profiles, or assessment rubric provided. Video hosting platform selection missing.",
  "confidence": 55,
  "keywords": ["cohort", "milestones", "curriculum", "progress-tracking", "certificates"],
  "analysis": "Platform structure is clear but content depth and instructor workflow are undefined, creating moderate execution risk."
}}

Ejemplo 2 (Arquetipo: moocs)
Tarea: "Build a massive open online course platform with auto-graded quizzes and video streaming."
{{
  "trajectory": "1) Design course catalog with lesson hierarchy. 2) Implement video streaming with CDN delivery. 3) Build quiz engine with auto-grading logic. 4) Add user progress persistence across devices. 5) Deploy on edge infrastructure for global scalability.",
  "gap": "No curriculum content, quiz question bank, or video assets exist. CDN budget and DRM requirements not specified.",
  "confidence": 45,
  "keywords": ["video-streaming", "quiz-engine", "auto-grading", "cdn", "progress"],
  "analysis": "Technical architecture for scale is demanding. Missing content and media assets represent major blockers to launch."
}}

Ejemplo 3 (Arquetipo: certifications)
Tarea: "Develop an online certification exam platform with anti-cheating and PDF certificate issuance."
{{
  "trajectory": "1) Design exam engine with timed sections. 2) Implement anti-cheat measures (tab switch detection, browser lockdown). 3) Build scoring engine with passing threshold logic. 4) Generate PDF certificates with cryptographic verification. 5) Add proctoring dashboard for review.",
  "gap": "No question bank, passing score rules, or certificate design template provided. Proctoring policy (AI vs human) undefined.",
  "confidence": 50,
  "keywords": ["exam-engine", "anti-cheat", "pdf-generation", "scoring", "proctoring"],
  "analysis": "Core exam logic is straightforward but anti-cheat robustness and certificate security add validation complexity."
}}

--- TAREA A SIMULAR ---
Domain: {domain}
Archetype: {archetype}
Task Description: {task_description}
Interaction History: {history}

Genera el JSON del world model ahora.
"""

GRAPHIC_DESIGN_WORLD_MODEL_PROMPT = """Eres la capa "World Model" dentro del Planner Agent de CogniTeam.

Tu trabajo NO es ejecutar la tarea, sino SIMULAR la trayectoria futura si se sigue el plan,
identificar gaps en el estado actual, y estimar la probabilidad de éxito.

REGLAS CRÍTICAS:
1. NO SPOILERS: Usa placeholders abstractos (ej. "definir paleta cromática"), NO valores específicos de la tarea a menos que se hayan dado explícitamente.
2. SÉ ESPECÍFICO: En la Trajectory, indica hitos concretos, evita frases genéricas como "hacer el trabajo".
3. CALIBRACIÓN DE CONFIANZA: Sé honesto. Si la tarea carece de inputs críticos, la confianza debe ser baja.
4. KEYWORDS: Extrae hasta 5 términos que DEBEN aparecer en la ejecución real para considerar el plan anclado.

FORMATO OUTPUT (JSON estricto):
{{
  "trajectory": "Roadmap paso a paso...",
  "gap": "Por qué el estado actual es insuficiente...",
  "confidence": 0-100 (entero),
  "keywords": ["keyword1", "keyword2", ...],
  "analysis": "Evaluación del plan..."
}}

--- EJEMPLOS ---

Ejemplo 1 (Arquetipo: branding-identity)
Tarea: "Design a complete brand identity for a fintech startup including logo, typography, and color system."
{{
  "trajectory": "1) Research competitor brand landscape and target audience. 2) Develop mood board with visual direction. 3) Design primary logo lockup with variations. 4) Define color palette, typography hierarchy, and brand tokens. 5) Create application guidelines with mockups.",
  "gap": "No brand name, mission statement, or competitor list provided. Industry positioning and tone undefined.",
  "confidence": 60,
  "keywords": ["brand-identity", "logo", "color-palette", "typography", "guidelines"],
  "analysis": "Standard branding process is well-defined but lack of strategic inputs will result in generic output without course correction."
}}

Ejemplo 2 (Arquetipo: ui-ux-prototyping)
Tarea: "Design a mobile app UI/UX prototype for a food delivery service including ordering flow and payment."
{{
  "trajectory": "1) Define user personas and journey maps for ordering flow. 2) Create wireframes for restaurant browsing, cart, and checkout. 3) Design high-fidelity mockups with component library. 4) Build interactive prototype with transitions. 5) Conduct usability validation and iterate.",
  "gap": "No restaurant catalog, user demographics, or payment method preferences provided. Platform (iOS/Android) not specified.",
  "confidence": 55,
  "keywords": ["wireframes", "mockups", "prototype", "user-flow", "interaction"],
  "analysis": "Design methodology is sound but missing UX inputs will force assumptions that may not reflect real user needs."
}}

Ejemplo 3 (Arquetipo: editorial-layout)
Tarea: "Design an editorial layout for a 48-page architecture magazine with print-ready specifications."
{{
  "trajectory": "1) Define grid system and master page templates. 2) Establish typography scale and image treatment rules. 3) Design section openers and article page layouts. 4) Apply consistent styling to spreads with bleeds. 5) Export print-ready PDF with color profiles.",
  "gap": "No article content, photography assets, or ad placement map provided. Print vendor specifications unknown.",
  "confidence": 65,
  "keywords": ["grid-system", "typography", "spreads", "print-ready", "master-pages"],
  "analysis": "Layout structure is achievable but empty content slots and missing vendor specs make final output quality unpredictable."
}}

--- TAREA A SIMULAR ---
Domain: {domain}
Archetype: {archetype}
Task Description: {task_description}
Interaction History: {history}

Genera el JSON del world model ahora.
"""

GAME_DEV_WORLD_MODEL_PROMPT = """Eres la capa "World Model" dentro del Planner Agent de CogniTeam.

Tu trabajo NO es ejecutar la tarea, sino SIMULAR la trayectoria futura si se sigue el plan,
identificar gaps en el estado actual, y estimar la probabilidad de éxito.

REGLAS CRÍTICAS:
1. NO SPOILERS: Usa placeholders abstractos (ej. "definir mecánica principal"), NO valores específicos de la tarea a menos que se hayan dado explícitamente.
2. SÉ ESPECÍFICO: En la Trajectory, indica hitos concretos, evita frases genéricas como "hacer el trabajo".
3. CALIBRACIÓN DE CONFIANZA: Sé honesto. Si la tarea carece de inputs críticos, la confianza debe ser baja.
4. KEYWORDS: Extrae hasta 5 términos que DEBEN aparecer en la ejecución real para considerar el plan anclado.

FORMATO OUTPUT (JSON estricto):
{{
  "trajectory": "Roadmap paso a paso...",
  "gap": "Por qué el estado actual es insuficiente...",
  "confidence": 0-100 (entero),
  "keywords": ["keyword1", "keyword2", ...],
  "analysis": "Evaluación del plan..."
}}

--- EJEMPLOS ---

Ejemplo 1 (Arquetipo: mobile-casual)
Tarea: "Create a 2D endless runner mobile game with power-ups and ad-based monetization."
{{
  "trajectory": "1) Set up Unity project with 2D sprite pipeline. 2) Implement procedural level generation algorithm. 3) Build player controller with swipe input handling. 4) Add power-up system with timed effects. 5) Integrate AdMob for rewarded and interstitial ads.",
  "gap": "No character sprites, environment tiles, or sound effects provided. Ad unit IDs and monetization waterfall not configured.",
  "confidence": 55,
  "keywords": ["2d-sprites", "procedural-generation", "swipe-input", "power-ups", "admob"],
  "analysis": "Core loop is standard for the genre but missing art assets and ad configuration block release readiness."
}}

Ejemplo 2 (Arquetipo: pc-console)
Tarea: "Develop a third-person action-adventure game with open world exploration using Unreal Engine 5."
{{
  "trajectory": "1) Prototype player character with basic movement and camera. 2) Design open world terrain with landscape tool. 3) Implement combat system with animation blueprints. 4) Add quest system and NPC interaction framework. 5) Optimize with Nanite and Lumen for target hardware.",
  "gap": "No game design document, concept art, or narrative outline provided. Target platform (PC/console) and performance target unspecified.",
  "confidence": 35,
  "keywords": ["unreal-engine-5", "open-world", "combat-system", "animation", "nanite"],
  "analysis": "AAA-scope project with insufficient specification. Without a design document and art direction the project will fail due to scope creep."
}}

Ejemplo 3 (Arquetipo: vr-ar)
Tarea: "Build a VR training simulation for manufacturing assembly line workers using Unity XR."
{{
  "trajectory": "1) Set up Unity XR project with OpenXR integration. 2) Implement hand interaction system for grabbing tools. 3) Design step-by-step assembly tutorial with visual cues. 4) Add haptic feedback for correct/incorrect actions. 5) Optimize for 90fps with foveated rendering.",
  "gap": "No 3D models of tools or assembly parts provided. Training curriculum and assessment criteria not defined.",
  "confidence": 50,
  "keywords": ["openxr", "hand-interaction", "haptics", "tutorial", "foveated-rendering"],
  "analysis": "VR training sim follows established patterns but missing 3D assets and curriculum make the simulation hollow."
}}

--- TAREA A SIMULAR ---
Domain: {domain}
Archetype: {archetype}
Task Description: {task_description}
Interaction History: {history}

Genera el JSON del world model ahora.
"""


# Prompts específicos por dominio. Cada entrada puede reemplazar al prompt genérico.
# Las claves son los domain_key del sistema (ej. "1_web_development").
DOMAIN_WORLD_MODEL_PROMPTS: Dict[str, str] = {
    "1_web_development": WEB_DEV_WORLD_MODEL_PROMPT,
    "2_software_development": SOFTWARE_DEV_WORLD_MODEL_PROMPT,
    "3_education": EDUCATION_WORLD_MODEL_PROMPT,
    "4_graphic_design": GRAPHIC_DESIGN_WORLD_MODEL_PROMPT,
    "8_game_development": GAME_DEV_WORLD_MODEL_PROMPT,
}

# Keywords de grounding por defecto por arquetipo (desde cogniteam_archetypes.yaml)
GROUNDING_KEYWORDS_FALLBACK: Dict[str, List[str]] = {
    # --- 1_web_development ---
    "landing-page": ["html", "responsive", "hero", "services", "contact-form", "cta", "footer"],
    "ecommerce": ["product-grid", "cart", "checkout", "payment", "catalog", "stripe", "inventory"],
    "saas-dashboard": ["dashboard", "charts", "auth", "sidebar", "data-table", "jwt", "real-time"],
    "blog-content": ["blog", "markdown", "seo", "rss", "categories", "typography", "articles"],
    "portfolio-creative": ["portfolio", "gallery", "animations", "creative", "showcase", "framer-motion", "gsap"],
    "corporate-business": ["corporate", "business", "services", "contact", "legal-pages", "multilingual", "branding"],
    # --- 2_software_development ---
    "mobile-apps": ["mobile", "flutter", "react-native", "offline", "notifications", "local-db", "cross-platform"],
    "backend-apis": ["rest-api", "fastapi", "nestjs", "endpoints", "auth", "validation", "openapi"],
    "databases": ["database", "schema", "sql", "nosql", "migrations", "indexes", "postgresql"],
    "devops-infra": ["devops", "docker", "ci-cd", "terraform", "kubernetes", "monitoring", "iac"],
    "machine-learning": ["machine-learning", "model", "training", "pytorch", "sklearn", "inference", "mlflow"],
    # --- 3_education ---
    "bootcamps": ["bootcamp", "cohort", "curriculum", "milestones", "progress", "certificates", "intensive"],
    "moocs": ["mooc", "online-course", "video", "scalable", "quiz", "auto-grading", "cdn"],
    "certifications": ["certification", "exam", "anti-cheat", "scoring", "pdf", "proctoring", "credentials"],
    "corporate-training": ["corporate-training", "lms", "scorm", "tenant", "learning-paths", "sso", "analytics"],
    "k12-curriculum": ["k12", "education", "gamification", "curriculum", "parental-controls", "accessibility", "rewards"],
    # --- 4_graphic_design ---
    "branding-identity": ["branding", "logo", "color-palette", "typography", "identity", "guidelines", "tokens"],
    "editorial-layout": ["editorial", "layout", "grid", "typography", "magazine", "print", "spreads"],
    "ui-ux-prototyping": ["ui", "ux", "prototype", "wireframes", "mockups", "figma", "user-flow"],
    # --- 8_game_development ---
    "mobile-casual": ["mobile-game", "casual", "unity", "2d", "ads", "procedural", "touch-input"],
    "pc-console": ["pc-game", "console", "unreal-engine", "open-world", "combat", "animation", "hdrp"],
    "multiplayer-online": ["multiplayer", "netcode", "photon", "dedicated-server", "matchmaking", "anti-cheat", "replication"],
    "vr-ar": ["vr", "ar", "openxr", "hand-tracking", "haptics", "room-scale", "stereoscopic"],
    "game-tools": ["game-tools", "editor", "automation", "pipeline", "procedural", "workflow", "python-scripting"],
}


def _select_world_model_prompt(domain: str, archetype: str) -> str:
    """Selecciona el prompt de world model según dominio, con fallback al genérico."""
    return DOMAIN_WORLD_MODEL_PROMPTS.get(domain, WORLD_MODEL_PROMPT)


def generate_world_model(
    task: str,
    domain: str,
    archetype: str,
    history: str = "",
    tools_description: str = "",
) -> Optional[Dict[str, Any]]:
    """Genera un <world_model> block antes del plan: trajectory, gap, confidence, keywords."""
    print(f"\n[World Model] Simulando trayectoria para {domain}.{archetype}...")
    prompt_template = _select_world_model_prompt(domain, archetype)
    prompt = prompt_template.format(
        task=task,
        domain=domain,
        archetype=archetype,
        task_description=task,
        history=history or "No previous interactions.",
    )
    # Añadir contexto sobre herramientas disponibles para que el world model
    # evalúe los gaps correctamente (sabe qué puede resolverse en ejecución)
    if tools_description:
        prompt += (
            f"\n\nHERRAMIENTAS DISPONIBLES PARA EJECUTAR EL PLAN:\n{tools_description}\n\n"
            f"IMPORTANTE: Evalúa los gaps considerando que estas herramientas están disponibles "
            f"para resolverlos durante la ejecución. Por ejemplo: si falta contenido externo, "
            f"web_search_real + browse_web_page pueden obtenerlo; si falta un archivo, "
            f"generate_textual_artifact + write_file_sandboxed pueden crearlo. "
            f"NO marques como gap algo que estas herramientas pueden resolver."
        )
    raw = llm_complete(prompt=prompt, task="world_model", max_tokens=2048, temperature=0.3, timeout_seconds=120)
    if not raw:
        print("  [World Model] No se obtuvo respuesta del LLM.")
        return None
    result = _extract_json(raw)
    if result:
        result.setdefault("confidence", 50)
        if not result.get("keywords"):
            result["keywords"] = GROUNDING_KEYWORDS_FALLBACK.get(archetype, [])
        print(f"  [World Model] Confianza: {result['confidence']}% | Keywords: {result['keywords'][:3]}...")
        return result
    print(f"  [World Model] No se pudo extraer JSON. Respuesta: {raw[:400]}...")
    return None


# --- Few-shot examples por dominio ---
# Cada entrada es el ejemplo que se inyecta en PLANNER_INSTRUCTION_TEMPLATE.
# Solo se muestra UN ejemplo por ejecución (el del dominio actual).
FEWSHOT_EDUCATION = """EJEMPLO (Educación - plataforma de curso online):
{"steps":[
  {"step":1, "agent":"DeveloperAgent_CogniTeam", "action_description":"Buscar recursos educativos sobre el tema solicitado", "tool_to_use":"web_search_real", "inputs":{"query":"contenido didactico sobre el tema de la tarea", "num_results":5}, "output_variable_name":"recursos", "expected_output_format":"json"},
  {"step":2, "agent":"DeveloperAgent_CogniTeam", "action_description":"Generar estructura del curso con modulos y lecciones", "tool_to_use":"generate_textual_artifact", "inputs":{"description_of_artifact":"Estructura de curso con modulos, lecciones, objetivos de aprendizaje y recursos basado en {{recursos}}"}, "output_variable_name":"estructura_curso", "expected_output_format":"text"},
  {"step":3, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar HTML de pagina de curso con progreso", "tool_to_use":"generate_ui_code", "inputs":{"description":"Pagina de curso online con {{estructura_curso}}. Incluye header, modulo de lecciones con expansion, barra de progreso y seccion de materiales"}, "output_variable_name":"html_code", "expected_output_format":"html"},
  {"step":4, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar CSS moderno responsive para plataforma educativa", "tool_to_use":"generate_css_code", "inputs":{"description":"CSS moderno responsive con modo claro/oscuro, tipografia legible, cards de lecciones con expansion, barra de progreso animada"}, "output_variable_name":"css_code", "expected_output_format":"css"},
  {"step":5, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Combinar assets en curso.html", "tool_to_use":"combine_ui_to_html", "inputs":{"html":"{{html_code}}", "css":"{{css_code}}", "js":"{{js_code}}", "filepath":"curso.html"}, "output_variable_name":"html_final", "expected_output_format":"html"}
]}"""

FEWSHOT_GRAPHIC_DESIGN = """EJEMPLO (Diseño Gráfico - portafolio de identidad visual):
{"steps":[
  {"step":1, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar HTML con portafolio de identidad visual", "tool_to_use":"generate_ui_code", "inputs":{"description":"Pagina de portafolio de diseno grafico mostrando identidad visual, paleta de colores, tipografia y aplicaciones de marca"}, "output_variable_name":"html_code", "expected_output_format":"html"},
  {"step":2, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar CSS con diseno editorial moderno", "tool_to_use":"generate_css_code", "inputs":{"description":"CSS con diseno editorial, tipografia expresiva, grid de portafolio, animaciones sutiles y paleta cromatica de marca"}, "output_variable_name":"css_code", "expected_output_format":"css"},
  {"step":3, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Combinar assets en portfolio.html", "tool_to_use":"combine_ui_to_html", "inputs":{"html":"{{html_code}}", "css":"{{css_code}}", "js":"{{js_code}}", "filepath":"portfolio.html"}, "output_variable_name":"html_final", "expected_output_format":"html"}
]}"""

FEWSHOT_GAME_DEV = """EJEMPLO (Game Development - juego canvas autocontenido con validacion funcional):
{"steps":[
  {"step":1, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar HTML autocontenido del juego (CSS in `<style>` en <head>, JS in `<script>` al final del <body>, justo antes de `</body>`)", "tool_to_use":"generate_ui_code", "inputs":{"description":"Archivo HTML UNICO y autocontenido. CSS va en <style> dentro del <head>. JS va en <script> al FINAL del <body> justo antes de </body> (despues de todos los elementos HTML). El script NO debe asumir elementos que no existen en el HTML. Toda la logica (game loop, input, colisiones, puntuacion) va en ese unico script. NO generar HTML, CSS y JS por separado — todo en un solo archivo."}, "output_variable_name":"html_code", "expected_output_format":"html"},
  {"step":2, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Persistir el HTML a juego.html", "tool_to_use":"write_file_sandboxed", "inputs":{"relative_filepath":"juego.html", "content":"{{html_code}}"}, "output_variable_name":"archivo_html", "expected_output_format":"text"},
  {"step":3, "agent":"DeveloperAgent_CogniTeam", "action_description":"Validar funcionalidad del juego en navegador (sin test script externo — solo carga la pagina y verifica que no haya errores JS en consola)", "tool_to_use":"validate_html_functional", "inputs":{"filepath":"juego.html", "capture_screenshot":true}, "output_variable_name":"validacion", "expected_output_format":"json"}
]}"""

FEWSHOT_UI = """EJEMPLO (UI + Contenido - landing page enlazada a blog.md):
{"steps":[
  {"step":1, "agent":"DeveloperAgent_CogniTeam", "action_description":"Buscar URLs de papers recientes", "tool_to_use":"web_search_real", "inputs":{"query":"tres papers mas recientes sobre el tema solicitado", "num_results":5}, "output_variable_name":"urls_encontradas", "expected_output_format":"json"},
  {"step":2, "agent":"DeveloperAgent_CogniTeam", "action_description":"Navegar cada URL para obtener detalles completos de cada paper", "tool_to_use":"browse_web_page", "inputs":{"url":"{{urls_encontradas}}"}, "output_variable_name":"detalle_completo", "expected_output_format":"text"},
  {"step":3, "agent":"DeveloperAgent_CogniTeam", "action_description":"Extraer info estructurada: titulo, autor, resumen por cada paper", "tool_to_use":"extract_info_from_text", "inputs":{"text_content":"{{detalle_completo}}", "question_or_instruction":"Extrae titulo, autor, resumen y fecha de cada paper. Devuelve una lista por paper."}, "output_variable_name":"papers_estructurados", "expected_output_format":"text"},
  {"step":4, "agent":"DeveloperAgent_CogniTeam", "action_description":"Generar blog.md con un articulo por paper, cada uno con anchor HTML <a name='paper-1'></a>, <a name='paper-2'></a>", "tool_to_use":"generate_textual_artifact", "inputs":{"description_of_artifact":"Blog en markdown con {{papers_estructurados}}. Cada paper es una seccion con anchor HTML (<a name='paper-N'></a>) para enlace directo desde el HTML"}, "output_variable_name":"blog_content", "expected_output_format":"text"},
  {"step":5, "agent":"DeveloperAgent_CogniTeam", "action_description":"Persistir blog.md a disco", "tool_to_use":"write_file_sandboxed", "inputs":{"relative_filepath":"blog.md", "content":"{{blog_content}}"}, "output_variable_name":"blog_file", "expected_output_format":"text"},
  {"step":6, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar HTML con tarjetas de cada paper y links a blog.md#paper-1, blog.md#paper-2, blog.md#paper-3", "tool_to_use":"generate_ui_code", "inputs":{"description":"Landing page moderna. Datos de los papers: {{papers_estructurados}}. Cada tarjeta enlaza a blog.md#paper-N para el detalle completo"}, "output_variable_name":"html_code", "expected_output_format":"html"},
  {"step":7, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar CSS", "tool_to_use":"generate_css_code", "inputs":{"description":"CSS moderno responsive"}, "output_variable_name":"css_code", "expected_output_format":"css"},
  {"step":8, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Combinar HTML, CSS y JS en index.html", "tool_to_use":"combine_ui_to_html", "inputs":{"html":"{{html_code}}", "css":"{{css_code}}", "js":"{{js_code}}", "filepath":"index.html"}, "output_variable_name":"html_final", "expected_output_format":"html"}
]}"""

FEWSHOT_SCRIPT = """EJEMPLO (Scripting - cadena genera→escribe→ejecuta→verifica):
{"steps":[
  {"step":1, "agent":"DeveloperAgent_CogniTeam", "action_description":"Generar script Python para analizar datos y producir output", "tool_to_use":"generate_textual_artifact", "inputs":{"description_of_artifact":"Script Python que procesa datos y genera resultados"}, "output_variable_name":"script_generado", "expected_output_format":"text"},
  {"step":2, "agent":"DeveloperAgent_CogniTeam", "action_description":"Escribir el script a disco", "tool_to_use":"write_file_sandboxed", "inputs":{"relative_filepath":"script.py", "content":"{{script_generado}}"}, "output_variable_name":"archivo_escrito", "expected_output_format":"text"},
  {"step":3, "agent":"DeveloperAgent_CogniTeam", "action_description":"Ejecutar el script para validar que funciona", "tool_to_use":"execute_terminal_command_safe", "inputs":{"command":"python3 script.py"}, "output_variable_name":"ejecucion", "expected_output_format":"text"},
  {"step":4, "agent":"DeveloperAgent_CogniTeam", "action_description":"Verificar que los archivos generados existen", "tool_to_use":"list_files_sandboxed", "inputs":{"relative_dirpath":"."}, "output_variable_name":"verificacion", "expected_output_format":"text"}
]}

EJEMPLO (HTML + testing - genera HTML, valida con navegador):
{"steps":[
  {"step":1, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar HTML del juego", "tool_to_use":"generate_ui_code", "inputs":{"description":"Juego space shooter en canvas con nave, enemigos, puntuacion y game over"}, "output_variable_name":"html_code", "expected_output_format":"html"},
  {"step":2, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar CSS", "tool_to_use":"generate_css_code", "inputs":{"description":"CSS moderno responsive"}, "output_variable_name":"css_code", "expected_output_format":"css"},
  {"step":3, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar JS del juego", "tool_to_use":"generate_js_code", "inputs":{"description":"JS del juego con game loop, input handling, colisiones y puntuacion"}, "output_variable_name":"js_code", "expected_output_format":"js"},
  {"step":4, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Combinar HTML, CSS y JS en juego.html", "tool_to_use":"combine_ui_to_html", "inputs":{"html":"{{html_code}}", "css":"{{css_code}}", "js":"{{js_code}}", "filepath":"juego.html"}, "output_variable_name":"html_final", "expected_output_format":"html"},
  {"step":5, "agent":"DeveloperAgent_CogniTeam", "action_description":"Generar test funcional JS para el juego", "tool_to_use":"generate_textual_artifact", "inputs":{"description_of_artifact":"Script JS que se inyectara en el HTML via validate_html_functional. Debe verificar: (1) que existe un elemento canvas, (2) que no hay errores JS en consola, (3) simular teclas y verificar que la puntuacion es un numero valido. Debe retornar un dict con los resultados."}, "output_variable_name":"test_script", "expected_output_format":"text"},
  {"step":6, "agent":"DeveloperAgent_CogniTeam", "action_description":"Ejecutar validacion funcional del HTML con el test script", "tool_to_use":"validate_html_functional", "inputs":{"filepath":"juego.html", "test_script":"{{test_script}}", "capture_screenshot":true}, "output_variable_name":"validacion", "expected_output_format":"json"},
  {"step":7, "agent":"DeveloperAgent_CogniTeam", "action_description":"Verificar resultado de validacion", "tool_to_use":"execute_terminal_command_safe", "inputs":{"command":"python3 -c \\\"import json; r=json.loads('{{validacion}}'); assert r.get('passed'), f'Validacion fallo: {r.get(\\\"test_results\\\", {})}'\\\" && echo 'VALIDACION OK'"}, "output_variable_name":"verificacion_test", "expected_output_format":"text"}
]}"""

FEWSHOT_CONTENT = """EJEMPLO (Generación de contenido):
{"steps":[
  {"step":1, "agent":"DeveloperAgent_CogniTeam", "action_description":"Generar el contenido solicitado", "tool_to_use":"generate_textual_artifact", "inputs":{"description_of_artifact":"Contenido/documento según los requisitos de la tarea"}, "output_variable_name":"contenido", "expected_output_format":"text"},
  {"step":2, "agent":"DeveloperAgent_CogniTeam", "action_description":"Persistir el contenido a disco como archivo", "tool_to_use":"write_file_sandboxed", "inputs":{"relative_filepath":"output.txt", "content":"{{contenido}}"}, "output_variable_name":"archivo", "expected_output_format":"text"}
]}"""

DOMAIN_FEWSHOT_EXAMPLES: Dict[str, str] = {
    "1_web_development": FEWSHOT_UI,
    "2_software_development": FEWSHOT_SCRIPT,
    "3_education": FEWSHOT_EDUCATION,
    "4_graphic_design": FEWSHOT_GRAPHIC_DESIGN,
    "8_game_development": FEWSHOT_GAME_DEV,
}

PLANNER_INSTRUCTION_TEMPLATE = """Eres un Planificador Senior IA. Genera un plan JSON detallado paso a paso.

REGLAS:
1. Output ÚNICAMENTE JSON válido. Objeto raíz con clave "steps" (lista plana).
2. Cada paso tiene: "step" (int), "agent" (string), "action_description" (string), "tool_to_use" (string), "inputs" (dict), "output_variable_name" (string), "expected_output_format" (string).
3. NO te auto-asignes pasos (PlannerAgent nunca en steps).
4. Asigna tools a agentes según corresponda (los agentes se asignan en base a la tool).
5. En "inputs", usa los NOMBRES EXACTOS de los parámetros de la tool. Los parámetros req son obligatorios, opt son opcionales.
6. NO uses apply_script, delete_*, git_*, move_or_rename*. Prefiere write_file_sandboxed y validate_script. Usa execute_terminal_command_safe SOLO para ejecutar scripts que hayas generado y escrito a disco.
7. IMPORTANTE: Usa SIEMPRE rutas RELATIVAS, sin "/" inicial ni "..". NO uses rutas absolutas como /tmp/ o /mnt/.
8. write_file_sandboxed ya crea directorios automáticamente. NO necesitas create_directory_sandboxed antes.
9. validate_script valida sintaxis SOLO para scripts bash. NO funciona para Python. Para Python usa execute_terminal_command_safe directamente.
10. generate_textual_artifact genera contenido SOLO EN MEMORIA. Siempre usa write_file_sandboxed despues para persistirlo a disco.
11. Para tareas que requieran ejecutar scripts, la cadena completa es: generate_textual_artifact (crear) → write_file_sandboxed (persistir) → execute_terminal_command_safe (ejecutar) → list_files_sandboxed (verificar output).
12. IMPORTANTE: Encadena datos entre pasos usando {{{{output_variable_name}}}} en los inputs. Si el paso 2 extrae informacion (ej: {{{{paper_info}}}}), pasala al paso 3 como "description": "Landing page con: {{{{paper_info}}}}". NO generes contenido vacio o placeholder - los datos fluyen mediante referencias.
13. Para tareas que requieran extraer datos de sitios web: usa web_search_real para encontrar URLs, luego browse_web_page en CADA URL individual para obtener el contenido completo, y extract_info_from_text sobre el resultado de browse_web_page para estructurarlo.
14. SI generas MULTIPLES artefactos (ej: index.html y blog.md), ENLAZALOS entre si: el HTML debe tener links <a href=\"blog.md#articulo-1\"> a las secciones del .md, y el .md debe tener anchors HTML (<a name=\"articulo-1\">) en cada seccion para que los links funcionen.
15. NUNCA generes dos implementaciones diferentes del mismo componente. Genera UNA sola implementacion completa y coherente. No mezcles canvas y DOM para el mismo juego, ni dos bibliotecas/frameworks distintos para la misma funcionalidad. Si generas un juego con canvas, TODO debe ir dentro del canvas (dibujado, logica, input handling). No añadas una copia DOM-based paralela.
16. DESPUES de generar artefactos HTML, incluye pasos de validacion funcional: (a) generar un test script JS que verifique funcionalidades clave (ej: "simula 5 disparos, verifica que la puntuacion > 0"), (b) llamar a validate_html_functional con filepath="./juego.html" y test_script="<codigo JS>", (c) verificar que el resultado contenga "passed": true. Si falla, replanifica corrigiendo errores.
17. El orquestador fuerza automaticamente que todos los archivos se escriban en el directorio "artifacts/<session_id>/". NO necesitas incluir ningun prefijo de ruta - escribe rutas RELATIVAS simples como "juego.html" o "index.html". El orquestador se encarga de redirigirlas al directorio de artefactos.
18. CRITICO para juegos canvas: genera UN SOLO archivo HTML autocontenido con CSS en <style> dentro del <head> y JS en <script> al FINAL del <body> (despues de todos los elementos HTML). NO separes HTML, CSS y JS en pasos distintos — usa UN solo generate_ui_code. El script JS debe ejecutarse despues de que el DOM este completamente cargado, usando DOMContentLoaded o colocando el <script> al final del body. Si separas en pasos, el orden de los scripts fallara porque el JS se ejecutara antes de que sus dependencias (clases, config) esten definidas.

{fewshot_example}

NO incluyas comentarios ni texto fuera del JSON. Output ÚNICAMENTE el objeto JSON.

Herramientas disponibles:
{tools_description}

Agentes:
{agents_description}"""


class PlannerAgent:
    """Planner agent - generates execution plans via LLM."""
    name: str = "PlannerAgent"

    def __init__(self, instruction: str = ""):
        self.instruction = instruction


def create_planner_agent(agent_name: str, instruction: str = "", tools=None) -> PlannerAgent:
    agent = PlannerAgent(instruction=instruction)
    agent.name = agent_name
    return agent


def _build_prompt(
    requirements: str,
    tools_description: str,
    agents_description: str,
    replan_context: Optional[str] = None,
    domain: str = "",
) -> str:
    fewshot = DOMAIN_FEWSHOT_EXAMPLES.get(domain, FEWSHOT_CONTENT)
    prompt = PLANNER_INSTRUCTION_TEMPLATE.format(
        fewshot_example=fewshot,
        tools_description=tools_description,
        agents_description=agents_description,
    )
    prompt = f"Genera un plan JSON para:\n{requirements}\n\n{prompt}"
    if replan_context:
        prompt += (
            f"\n\nContexto del intento anterior:\n{replan_context}\n"
            f"Genera un NUEVO plan CORREGIDO."
        )
    return prompt


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    if raw.lower().startswith("```json"):
        raw = raw[len("```json"):].strip()
    if raw.lower().startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-len("```")].strip()

    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group()

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(raw):
        try:
            obj, end = decoder.raw_decode(raw, idx)
            if isinstance(obj, dict):
                return obj
            idx = end
        except json.JSONDecodeError:
            idx += 1

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


@traceforge.trace(agent="planner.generate", tags=["llm", "planning"])
async def generate_plan(
    planner_agent: PlannerAgent,
    requirements: str,
    tools_description: str = "",
    agents_description: str = "",
    replan_context: Optional[str] = None,
    domain: str = "",
) -> Optional[Dict[str, Any]]:
    """Generate an execution plan via direct LLM call (no ADK)."""
    print(f"\n[Planificación] Generando plan para dominio: {domain or 'ninguno'}...")

    prompt = _build_prompt(requirements, tools_description, agents_description, replan_context, domain)

    raw_output = llm_complete(
        prompt=prompt,
        task="planning",
        max_tokens=4096,
        temperature=0.1,
        timeout_seconds=300,
    )

    if not raw_output:
        print("  ERROR: No se obtuvo respuesta del LLM.")

    plan = _extract_json(raw_output) if raw_output else None
    if plan:
        if "plan_id" not in plan:
            plan["plan_id"] = f"plan_{int(time.time())}_{random.randint(1000, 9999)}"
        print(f"  Plan generado: {plan.get('plan_id', 'N/A')}")
        print(f"  Pasos: {len(plan.get('steps', []))}")
        return plan

    print("  ERROR: No se pudo extraer JSON válido del Planner.")
    return None


async def generate_plan_with_world_model(
    planner_agent: PlannerAgent,
    requirements: str,
    domain: str = "",
    archetype: str = "",
    tools_description: str = "",
    agents_description: str = "",
    replan_context: Optional[str] = None,
    calibration_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Genera world model ANTES del plan; retorna RECLARIFY si confianza baja."""
    wm_status = "skipped"
    if domain and archetype:
        wm = generate_world_model(requirements, domain, archetype, tools_description=tools_description)
        wm_status = "ok" if wm else "failed"
    else:
        wm = None

    if wm:
        threshold = calibration_threshold or 0.5
        confidence_ratio = wm.get("confidence", 0) / 100.0
        if confidence_ratio < threshold:
            return {
                "action": "RECLARIFY",
                "reason": f"Confianza {wm['confidence']}% < umbral {threshold*100:.0f}%",
                "gaps": [wm.get("gap", "")],
                "world_model": wm,
                "plan": None,
                "wm_status": wm_status,
            }

        context = (
            f"\n\n--- WORLD MODEL (simulación prospectiva) ---\n"
            f"Trayectoria esperada: {wm.get('trajectory', '')}\n"
            f"Gap identificado: {wm.get('gap', '')}\n"
            f"Confianza estimada: {wm['confidence']}%\n"
            f"Keywords de verificación: {', '.join(wm.get('keywords', []))}\n"
            f"Análisis: {wm.get('analysis', '')}\n"
            f"Considera estos insights al generar el plan."
        )
        requirements = requirements + context
    else:
        print(f"  [World Model] No disponible (status={wm_status}). Procediendo sin simulación prospectiva.")

    plan = await generate_plan(
        planner_agent=planner_agent,
        requirements=requirements,
        tools_description=tools_description,
        agents_description=agents_description,
        replan_context=replan_context,
        domain=domain,
    )

    return {"action": "EXECUTE", "world_model": wm, "plan": plan, "wm_status": wm_status}
