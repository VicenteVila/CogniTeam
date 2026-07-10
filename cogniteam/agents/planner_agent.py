import json
import random
import re
import time
from typing import Any, Dict, List, Optional

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

ARCHITECTURE_SPATIAL_WORLD_MODEL_PROMPT = """Eres la capa "World Model" dentro del Planner Agent de CogniTeam.

Tu trabajo NO es ejecutar la tarea, sino SIMULAR la trayectoria futura si se sigue el plan,
identificar gaps en el estado actual, y estimar la probabilidad de éxito.

REGLAS CRÍTICAS:
1. NO SPOILERS: Usa placeholders abstractos (ej. "definir restricciones estructurales"), NO valores específicos de la tarea a menos que se hayan dado explícitamente.
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

Ejemplo 1 (Arquetipo: bim-modeling)
Tarea: "Create a BIM model for a 3-story office building with structural, MEP, and architectural layers."
{{
  "trajectory": "1) Set up BIM project with shared coordinates and levels. 2) Model structural grid and load-bearing elements. 3) Add architectural walls, floors, roofs with material parameters. 4) Integrate MEP systems with clash detection setup. 5) Generate quantity takeoffs and LOD 300 documentation.",
  "gap": "No structural engineering calculations, MEP load requirements, or local building code references provided. Site context model missing.",
  "confidence": 55,
  "keywords": ["bim", "structural", "mep", "clash-detection", "lod-300"],
  "analysis": "BIM workflow is standard but absence of engineering inputs and code references may invalidate the model for permit submission."
}}

Ejemplo 2 (Arquetipo: render-visualization)
Tarea: "Produce photorealistic exterior renderings for a modern villa project with daytime and night scenes."
{{
  "trajectory": "1) Import and clean 3D model with proper material IDs. 2) Set up PBR materials and texture mapping. 3) Configure HDRI lighting for daytime and artificial for night. 4) Place cameras with composition rules. 5) Render passes and composite final images.",
  "gap": "No 3D model file, material references, or landscape context provided. Output resolution and delivery format unspecified.",
  "confidence": 50,
  "keywords": ["pbr-materials", "hdr-lighting", "cameras", "render-passes", "compositing"],
  "analysis": "Render pipeline is well-understood but missing source model and material references make the final quality entirely dependent on assumptions."
}}

--- TAREA A SIMULAR ---
Domain: {domain}
Archetype: {archetype}
Task Description: {task_description}
Interaction History: {history}

Genera el JSON del world model ahora.
"""

MARKETING_GROWTH_WORLD_MODEL_PROMPT = """Eres la capa "World Model" dentro del Planner Agent de CogniTeam.

Tu trabajo NO es ejecutar la tarea, sino SIMULAR la trayectoria futura si se sigue el plan,
identificar gaps en el estado actual, y estimar la probabilidad de éxito.

REGLAS CRÍTICAS:
1. NO SPOILERS: Usa placeholders abstractos (ej. "definir segmentos de audiencia"), NO valores específicos de la tarea a menos que se hayan dado explícitamente.
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

Ejemplo 1 (Arquetipo: campaign-launch)
Tarea: "Launch a multi-channel ad campaign for a new D2C skincare product targeting women 25-40."
{{
  "trajectory": "1) Define campaign KPIs and budget allocation per channel. 2) Create ad creative variants (visuals + copy) for Meta, Google, TikTok. 3) Set up tracking pixels and conversion API. 4) Configure audience segments with lookalike targeting. 5) Launch with A/B testing and monitoring dashboard.",
  "gap": "No product photos, ad copy, or landing page URL provided. Daily budget and start date not specified.",
  "confidence": 45,
  "keywords": ["ad-creatives", "audience-segments", "conversion-tracking", "ab-testing", "kpi"],
  "analysis": "Campaign structure is correct but without creative assets and copy the launch is blocked. Audience definition is too broad for efficient spend."
}}

Ejemplo 2 (Arquetipo: seo-content-strategy)
Tarea: "Develop an SEO content strategy for a SaaS project management tool targeting SMBs."
{{
  "trajectory": "1) Perform keyword research with competitor gap analysis. 2) Build topical cluster map and content calendar. 3) Write cornerstone articles for high-volume terms. 4) Implement technical SEO recommendations (schema, internal linking). 5) Track rankings and iterate based on performance.",
  "gap": "No product USPs, pricing page, or competitor list provided. Target geography and language undefined.",
  "confidence": 55,
  "keywords": ["keyword-research", "content-clusters", "seo", "schema-markup", "rankings"],
  "analysis": "SEO methodology is sound but missing product positioning inputs may lead to targeting terms that don't convert."
}}

Ejemplo 3 (Arquetipo: automation-emailing)
Tarea: "Set up an automated email sequence for abandoned cart recovery in an e-commerce store."
{{
  "trajectory": "1) Define trigger events and user segments for the workflow. 2) Write email sequence with 3-step follow-up logic. 3) Design responsive email templates with dynamic product placeholders. 4) Configure DKIM/SPF for deliverability. 5) Set up analytics to track open and conversion rates.",
  "gap": "No product catalog API, brand email templates, or sending platform account provided. Discount strategy for recovery not specified.",
  "confidence": 60,
  "keywords": ["email-sequence", "trigger-events", "dynamic-content", "deliverability", "analytics"],
  "analysis": "Standard abandoned cart flow is well-defined but missing creative assets and incentive strategy reduce conversion potential."
}}

--- TAREA A SIMULAR ---
Domain: {domain}
Archetype: {archetype}
Task Description: {task_description}
Interaction History: {history}

Genera el JSON del world model ahora.
"""

AUDIOVISUAL_MANAGEMENT_WORLD_MODEL_PROMPT = """Eres la capa "World Model" dentro del Planner Agent de CogniTeam.

Tu trabajo NO es ejecutar la tarea, sino SIMULAR la trayectoria futura si se sigue el plan,
identificar gaps en el estado actual, y estimar la probabilidad de éxito.

REGLAS CRÍTICAS:
1. NO SPOILERS: Usa placeholders abstractos (ej. "definir estrategia de gancho inicial"), NO valores específicos de la tarea a menos que se hayan dado explícitamente.
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

Ejemplo 1 (Arquetipo: video-production)
Tarea: "Produce a 60-second promotional video for a mobile game launch targeting Gen Z on TikTok."
{{
  "trajectory": "1) Develop hook strategy within first 3 seconds for retention. 2) Write script with visual storyboard and timing. 3) Record voiceover or source music track. 4) Edit timeline with pacing optimized for platform. 5) Color grade, normalize audio to -14 LUFS, and export.",
  "gap": "No game footage, brand assets, or script outline provided. Platform aspect ratio (vertical/horizontal) not specified.",
  "confidence": 45,
  "keywords": ["hook-strategy", "storyboard", "editing", "audio-normalization", "color-grading"],
  "analysis": "Production pipeline is clear but missing source footage and script make this impossible to execute as specified."
}}

Ejemplo 2 (Arquetipo: business-operations)
Tarea: "Create an operational dashboard tracking KPIs for a video production agency with sprint-based workflows."
{{
  "trajectory": "1) Define KPI tree across sales, production, and delivery. 2) Build sprint tracking system with milestone checkpoints. 3) Integrate timesheet and budget burn data. 4) Design dashboard views for producers and executives. 5) Set up automated weekly reporting via webhook.",
  "gap": "No existing data sources, team structure, or current baseline metrics provided. Tooling preferences for the dashboard unknown.",
  "confidence": 60,
  "keywords": ["kpi-dashboard", "sprints", "timesheet", "budget-tracking", "reporting"],
  "analysis": "Operational framework is standard but without data integration points the dashboard will remain a shell."
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

DATA_SCIENCE_WORLD_MODEL_PROMPT = """Eres la capa "World Model" dentro del Planner Agent de CogniTeam.

Tu trabajo NO es ejecutar la tarea, sino SIMULAR la trayectoria futura si se sigue el plan,
identificar gaps en el estado actual, y estimar la probabilidad de éxito.

REGLAS CRÍTICAS:
1. NO SPOILERS: Usa placeholders abstractos (ej. "definir métricas de evaluación"), NO valores específicos de la tarea a menos que se hayan dado explícitamente.
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

Ejemplo 1 (Arquetipo: exploratory-analysis)
Tarea: "Perform exploratory data analysis on a customer churn dataset to identify key drivers."
{{
  "trajectory": "1) Load and profile dataset for missing values and distributions. 2) Create univariate and bivariate visualizations. 3) Analyze correlations between features and churn flag. 4) Segment customers by behavioral clusters. 5) Document findings with actionable recommendations.",
  "gap": "No dataset provided, column descriptions, or business context for churn definition. Time period of analysis not specified.",
  "confidence": 40,
  "keywords": ["eda", "visualizations", "correlation", "segmentation", "churn-analysis"],
  "analysis": "Analysis methodology is correct but without data or business context the output will be generic and not actionable."
}}

Ejemplo 2 (Arquetipo: ml-modeling)
Tarea: "Train a classification model to predict loan default risk with interpretable features."
{{
  "trajectory": "1) Perform feature engineering and selection from raw data. 2) Split data with stratified sampling for class imbalance. 3) Train baseline models (logistic regression, XGBoost). 4) Tune hyperparameters with cross-validation. 5) Evaluate with precision/recall and generate SHAP explanations.",
  "gap": "No dataset, target definition, or feature schema provided. Acceptable false positive rate not specified.",
  "confidence": 45,
  "keywords": ["classification", "feature-engineering", "xgboost", "hyperparameter-tuning", "shap"],
  "analysis": "Standard ML pipeline is well-defined but missing data and business constraints make model quality unpredictable."
}}

Ejemplo 3 (Arquetipo: etl-pipeline)
Tarea: "Build an ETL pipeline that ingests sales data from CSV, transforms it, and loads into PostgreSQL."
{{
  "trajectory": "1) Define source schema and target table structure. 2) Set up Airflow DAG with ingestion task. 3) Implement transformation logic for cleaning and aggregation. 4) Add data quality checks and error handling. 5) Schedule incremental loads with backfill strategy.",
  "gap": "No sample CSV files, target database credentials, or transformation rules provided. Incremental load key not identified.",
  "confidence": 55,
  "keywords": ["etl", "airflow", "postgresql", "data-quality", "incremental-load"],
  "analysis": "Pipeline architecture is solid but missing source data samples and transformation specs will require iterative refinement."
}}

--- TAREA A SIMULAR ---
Domain: {domain}
Archetype: {archetype}
Task Description: {task_description}
Interaction History: {history}

Genera el JSON del world model ahora.
"""

CONTENT_WRITING_WORLD_MODEL_PROMPT = """Eres la capa "World Model" dentro del Planner Agent de CogniTeam.

Tu trabajo NO es ejecutar la tarea, sino SIMULAR la trayectoria futura si se sigue el plan,
identificar gaps en el estado actual, y estimar la probabilidad de éxito.

REGLAS CRÍTICAS:
1. NO SPOILERS: Usa placeholders abstractos (ej. "definir tono y voz"), NO valores específicos de la tarea a menos que se hayan dado explícitamente.
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

Ejemplo 1 (Arquetipo: technical-documentation)
Tarea: "Write API documentation for a payment gateway REST API targeting third-party developers."
{{
  "trajectory": "1) Review API endpoints, request/response schemas, and authentication flow. 2) Write endpoint reference with parameters and examples. 3) Create integration guides for common scenarios. 4) Add error code reference and troubleshooting section. 5) Review for technical accuracy and publish to developer portal.",
  "gap": "No API specification, endpoint list, or authentication details provided. Target developer experience level unknown.",
  "confidence": 50,
  "keywords": ["api-docs", "endpoints", "integration-guide", "error-codes", "developer-portal"],
  "analysis": "Documentation structure is standard but without API specs the content will be placeholder-level only."
}}

Ejemplo 2 (Arquetipo: copywriting-marketing)
Tarea: "Write landing page copy for a SaaS project management tool focused on remote teams."
{{
  "trajectory": "1) Define unique value proposition and key differentiators. 2) Draft headline and subheadline with AIDA structure. 3) Write feature sections with benefit-driven copy. 4) Craft CTAs and social proof elements. 5) Optimize for readability and SEO keywords.",
  "gap": "No product features list, pricing information, or customer testimonials provided. Target audience persona undefined.",
  "confidence": 55,
  "keywords": ["value-proposition", "aida", "cta", "benefit-copy", "social-proof"],
  "analysis": "Copywriting methodology is correct but missing product inputs will result in generic messaging that fails to differentiate."
}}

Ejemplo 3 (Arquetipo: blog-article)
Tarea: "Write a 2000-word blog article about AI trends in healthcare for a B2B audience."
{{
  "trajectory": "1) Research current AI healthcare trends and statistics. 2) Outline article with H1/H2 structure and keyword placement. 3) Write introduction with hook and thesis statement. 4) Develop body sections with evidence and examples. 5) Review readability and optimize for SEO.",
  "gap": "No specific angle, target keywords, or source materials provided. Company brand voice guidelines unavailable.",
  "confidence": 60,
  "keywords": ["blog-writing", "seo", "readability", "outline", "research"],
  "analysis": "Article structure is straightforward but without a defined angle the content may lack focus and authority."
}}

--- TAREA A SIMULAR ---
Domain: {domain}
Archetype: {archetype}
Task Description: {task_description}
Interaction History: {history}

Genera el JSON del world model ahora.
"""

DEVOPS_INFRASTRUCTURE_WORLD_MODEL_PROMPT = """Eres la capa "World Model" dentro del Planner Agent de CogniTeam.

Tu trabajo NO es ejecutar la tarea, sino SIMULAR la trayectoria futura si se sigue el plan,
identificar gaps en el estado actual, y estimar la probabilidad de éxito.

REGLAS CRÍTICAS:
1. NO SPOILERS: Usa placeholders abstractos (ej. "definir pipeline de CI/CD"), NO valores específicos de la tarea a menos que se hayan dado explícitamente.
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

Ejemplo 1 (Arquetipo: ci-cd-pipeline)
Tarea: "Set up a CI/CD pipeline for a microservices project with automated testing and deployment to staging."
{{
  "trajectory": "1) Define build stages: lint, test, build, deploy. 2) Configure GitHub Actions workflow with matrix builds. 3) Set up Docker image building and registry push. 4) Implement staging deployment with health check verification. 5) Add rollback mechanism and notification hooks.",
  "gap": "No repository URL, test framework, or Dockerfile provided. Staging environment credentials and URL not specified.",
  "confidence": 55,
  "keywords": ["github-actions", "docker", "ci-cd", "deployment", "health-check"],
  "analysis": "Pipeline structure is standard but missing repository and environment access make this a template rather than an executable plan."
}}

Ejemplo 2 (Arquetipo: kubernetes-deployment)
Tarea: "Deploy a set of microservices to a Kubernetes cluster with ingress, service mesh, and autoscaling."
{{
  "trajectory": "1) Design namespace structure and resource quotas. 2) Write Kubernetes manifests for deployments and services. 3) Configure ingress controller with TLS termination. 4) Set up service mesh with Istio for traffic management. 5) Implement HPA autoscaling based on CPU/memory metrics.",
  "gap": "No container images, cluster access credentials, or service discovery requirements provided. Ingress domain and TLS cert source unknown.",
  "confidence": 50,
  "keywords": ["kubernetes", "ingress", "service-mesh", "hpa", "manifests"],
  "analysis": "K8s deployment pattern is correct but missing container artifacts and cluster context make this a theoretical exercise."
}}

Ejemplo 3 (Arquetipo: terraform-infra)
Tarea: "Provision AWS infrastructure with Terraform including VPC, EC2, RDS, and S3 for a web application."
{{
  "trajectory": "1) Define Terraform module structure and remote state backend. 2) Create VPC with public/private subnets and NAT gateway. 3) Provision EC2 instances with auto-scaling group. 4) Set up RDS PostgreSQL with read replica. 5) Configure S3 buckets with lifecycle policies and IAM roles.",
  "gap": "No AWS account ID, region preference, or instance sizing provided. VPC CIDR range and high-availability requirements undefined.",
  "confidence": 60,
  "keywords": ["terraform", "aws", "vpc", "rds", "infrastructure-as-code"],
  "analysis": "Infrastructure pattern is well-established but missing sizing and regional inputs may lead to costly provisioning mistakes."
}}

--- TAREA A SIMULAR ---
Domain: {domain}
Archetype: {archetype}
Task Description: {task_description}
Interaction History: {history}

Genera el JSON del world model ahora.
"""

CYBERSECURITY_WORLD_MODEL_PROMPT = """Eres la capa "World Model" dentro del Planner Agent de CogniTeam.

Tu trabajo NO es ejecutar la tarea, sino SIMULAR la trayectoria futura si se sigue el plan,
identificar gaps en el estado actual, y estimar la probabilidad de éxito.

REGLAS CRÍTICAS:
1. NO SPOILERS: Usa placeholders abstractos (ej. "definir alcance de auditoría"), NO valores específicos de la tarea a menos que se hayan dado explícitamente.
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

Ejemplo 1 (Arquetipo: security-audit)
Tarea: "Conduct a security audit for a web application following OWASP Top 10 methodology."
{{
  "trajectory": "1) Define audit scope and interview stakeholders. 2) Perform automated scanning with OWASP ZAP. 3) Manual review of authentication, session management, and access controls. 4) Test input validation and injection points. 5) Document findings with severity ratings and remediation recommendations.",
  "gap": "No target application URL, credentials for authenticated scanning, or source code access provided. Audit timeline and compliance framework not specified.",
  "confidence": 45,
  "keywords": ["owasp", "security-audit", "vulnerability-scanning", "penetration-test", "remediation"],
  "analysis": "Audit methodology is comprehensive but without target access the assessment cannot begin. Effort estimation is impossible without scope clarity."
}}

Ejemplo 2 (Arquetipo: penetration-testing)
Tarea: "Perform a penetration test on a corporate network to identify exploitable vulnerabilities."
{{
  "trajectory": "1) Perform reconnaissance and footprinting of target IP ranges. 2) Scan open ports and enumerate services. 3) Attempt exploitation of identified vulnerabilities. 4) Attempt privilege escalation and lateral movement. 5) Document findings with proof-of-concept and risk assessment.",
  "gap": "No target IP ranges, rules of engagement, or authorization letter provided. Testing window and exclusion list not defined.",
  "confidence": 35,
  "keywords": ["reconnaissance", "exploitation", "privilege-escalation", "pentest", "risk-assessment"],
  "analysis": "Technical approach is standard but without authorization and scope this engagement is both impossible and unethical to execute."
}}

Ejemplo 3 (Arquetipo: compliance-hardening)
Tarea: "Harden a Linux server fleet to meet CIS Benchmark Level 2 compliance."
{{
  "trajectory": "1) Assess current configuration against CIS benchmarks. 2) Apply kernel parameter hardening and disable unused services. 3) Configure file permissions and audit logging. 4) Implement SSH hardening and PAM configuration. 5) Automate compliance scanning with OpenSCAP and generate report.",
  "gap": "No server inventory, OS versions, or current baseline configuration provided. Exception policy for production-impacting changes not defined.",
  "confidence": 55,
  "keywords": ["cis-benchmark", "hardening", "openscap", "audit-logging", "compliance"],
  "analysis": "Hardening playbook is well-established but without server inventory and OS diversity assessment, the effort is unpredictable."
}}

--- TAREA A SIMULAR ---
Domain: {domain}
Archetype: {archetype}
Task Description: {task_description}
Interaction History: {history}

Genera el JSON del world model ahora.
"""

LEGAL_COMPLIANCE_WORLD_MODEL_PROMPT = """Eres la capa "World Model" dentro del Planner Agent de CogniTeam.

Tu trabajo NO es ejecutar la tarea, sino SIMULAR la trayectoria futura si se sigue el plan,
identificar gaps en el estado actual, y estimar la probabilidad de éxito.

REGLAS CRÍTICAS:
1. NO SPOILERS: Usa placeholders abstractos (ej. "definir jurisdicción aplicable"), NO valores específicos de la tarea a menos que se hayan dado explícitamente.
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

Ejemplo 1 (Arquetipo: privacy-policy)
Tarea: "Draft a privacy policy for a mobile health tracking app compliant with GDPR and CCPA."
{{
  "trajectory": "1) Identify all data collection points and processing purposes. 2) Draft data categories, legal basis, and retention periods. 3) Write user rights section with exercise procedures. 4) Add international transfer safeguards and third-party disclosures. 5) Include version history and effective date.",
  "gap": "No company legal name, registered address, or data processor list provided. Health data classification under GDPR Art 9 not addressed.",
  "confidence": 50,
  "keywords": ["privacy-policy", "gdpr", "ccpa", "data-categories", "user-rights"],
  "analysis": "Policy structure is standard but missing company details and health data classification create legal exposure if incorrect."
}}

Ejemplo 2 (Arquetipo: terms-of-service)
Tarea: "Draft terms of service for a SaaS platform with subscription billing and user-generated content."
{{
  "trajectory": "1) Define service description and account terms. 2) Draft payment, cancellation, and refund policies. 3) Write user conduct and content ownership clauses. 4) Include limitation of liability and disclaimers. 5) Add dispute resolution and governing law provisions.",
  "gap": "No company details, pricing model, or jurisdiction for governing law provided. UGC moderation policy undefined.",
  "confidence": 55,
  "keywords": ["terms-of-service", "subscription", "user-content", "liability", "governing-law"],
  "analysis": "ToS framework is complete but missing business model inputs will result in generic clauses that may not protect the company adequately."
}}

Ejemplo 3 (Arquetipo: data-processing-agreement)
Tarea: "Draft a Data Processing Agreement between a SaaS company and its customers under GDPR Article 28."
{{
  "trajectory": "1) Identify processing purposes and data categories. 2) Draft controller and processor obligations. 3) Include subprocessor authorization mechanism. 4) Add data breach notification procedures. 5) Incorporate SCCs for international transfers if applicable.",
  "gap": "No company names, processing activities description, or subprocessor list provided. International transfer mechanisms not specified.",
  "confidence": 60,
  "keywords": ["dpa", "data-processing", "gdpr-art-28", "subprocessors", "scc"],
  "analysis": "DPA structure is well-defined by regulation, but missing party details and processing descriptions make this a template rather than a usable agreement."
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
    "5_architecture_spatial": ARCHITECTURE_SPATIAL_WORLD_MODEL_PROMPT,
    "6_marketing_growth": MARKETING_GROWTH_WORLD_MODEL_PROMPT,
    "7_audiovisual_management": AUDIOVISUAL_MANAGEMENT_WORLD_MODEL_PROMPT,
    "8_game_development": GAME_DEV_WORLD_MODEL_PROMPT,
    "9_data_science": DATA_SCIENCE_WORLD_MODEL_PROMPT,
    "10_content_writing": CONTENT_WRITING_WORLD_MODEL_PROMPT,
    "11_devops_infrastructure": DEVOPS_INFRASTRUCTURE_WORLD_MODEL_PROMPT,
    "12_cybersecurity": CYBERSECURITY_WORLD_MODEL_PROMPT,
    "13_legal_compliance": LEGAL_COMPLIANCE_WORLD_MODEL_PROMPT,
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
    # --- 5_architecture_spatial ---
    "bim-modeling": ["bim", "revit", "archicad", "structural", "lod", "clash-detection", "mep"],
    "render-visualization": ["render", "visualization", "unreal-engine", "v-ray", "pbr", "lighting", "photorealistic"],
    # --- 6_marketing_growth ---
    "campaign-launch": ["campaign", "ads", "audience", "conversion", "creatives", "tracking", "ab-testing"],
    "seo-content-strategy": ["seo", "keywords", "content-strategy", "clusters", "rankings", "backlinks", "schema"],
    "automation-emailing": ["email", "automation", "sequences", "deliverability", "segments", "triggers", "analytics"],
    # --- 7_audiovisual_management ---
    "video-production": ["video", "production", "editing", "storyboard", "audio", "color-grading", "lufs"],
    "business-operations": ["operations", "kpis", "sprints", "dashboard", "budget", "milestones", "reporting"],
    # --- 8_game_development ---
    "mobile-casual": ["mobile-game", "casual", "unity", "2d", "ads", "procedural", "touch-input"],
    "pc-console": ["pc-game", "console", "unreal-engine", "open-world", "combat", "animation", "hdrp"],
    "multiplayer-online": ["multiplayer", "netcode", "photon", "dedicated-server", "matchmaking", "anti-cheat", "replication"],
    "vr-ar": ["vr", "ar", "openxr", "hand-tracking", "haptics", "room-scale", "stereoscopic"],
    "game-tools": ["game-tools", "editor", "automation", "pipeline", "procedural", "workflow", "python-scripting"],
    # --- 9_data_science ---
    "exploratory-analysis": ["eda", "pandas", "visualization", "statistics", "correlation", "cleaning", "profiling"],
    "ml-modeling": ["ml-model", "classification", "regression", "pytorch", "xgboost", "hyperparameters", "shap"],
    "dashboard-reporting": ["dashboard", "streamlit", "grafana", "kpi", "visualization", "metrics", "real-time"],
    "etl-pipeline": ["etl", "airflow", "pipeline", "data-warehouse", "dbt", "incremental", "data-quality"],
    "ab-testing": ["ab-testing", "experiment", "hypothesis", "statistics", "significance", "metrics", "segmentation"],
    # --- 10_content_writing ---
    "technical-documentation": ["documentation", "api-docs", "markdown", "guide", "tutorial", "technical-writing", "openapi"],
    "copywriting-marketing": ["copywriting", "marketing", "aida", "cta", "landing-page-copy", "conversion", "brand-voice"],
    "blog-article": ["blog", "article", "seo", "writing", "readability", "outline", "research"],
    "translation-localization": ["translation", "localization", "i18n", "l10n", "glossary", "locale", "internationalization"],
    "ux-writing": ["ux-writing", "microcopy", "interface", "usability", "error-messages", "onboarding", "plain-language"],
    # --- 11_devops_infrastructure ---
    "ci-cd-pipeline": ["ci-cd", "github-actions", "gitlab-ci", "jenkins", "docker", "automation", "deployment"],
    "kubernetes-deployment": ["kubernetes", "k8s", "helm", "istio", "ingress", "hpa", "namespaces"],
    "terraform-infra": ["terraform", "iac", "aws", "azure", "gcp", "state-backend", "provisioning"],
    "monitoring-observability": ["monitoring", "prometheus", "grafana", "loki", "alerting", "observability", "slo"],
    "disaster-recovery": ["disaster-recovery", "backup", "rto", "rpo", "failover", "velero", "high-availability"],
    # --- 12_cybersecurity ---
    "security-audit": ["security-audit", "owasp", "vulnerability", "scanning", "compliance", "remediation", "sarif"],
    "penetration-testing": ["penetration-test", "pentest", "exploitation", "reconnaissance", "burpsuite", "metasploit", "risk-assessment"],
    "compliance-hardening": ["hardening", "cis-benchmark", "compliance", "nist", "openscap", "secure-config", "audit-logging"],
    "incident-response": ["incident-response", "security-incident", "forensics", "containment", "playbook", "wazuh", "thehive"],
    # --- 13_legal_compliance ---
    "privacy-policy": ["privacy-policy", "gdpr", "ccpa", "data-protection", "user-rights", "cookies", "data-categories"],
    "terms-of-service": ["terms-of-service", "tos", "conditions", "liability", "governing-law", "subscription", "disclaimer"],
    "compliance-checklist": ["compliance-checklist", "audit", "iso-27001", "soc2", "controls", "evidence", "remediation"],
    "data-processing-agreement": ["dpa", "data-processing", "gdpr-art-28", "subprocessors", "scc", "controller", "processor"],
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
    raw = llm_complete(prompt=prompt, task="world_model", max_tokens=1024, temperature=0.3, timeout_seconds=120)
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
    print("  [World Model] No se pudo extraer JSON.")
    return None


# --- Few-shot examples por dominio ---
# Cada entrada es el ejemplo que se inyecta en PLANNER_INSTRUCTION_TEMPLATE.
# Solo se muestra UN ejemplo por ejecución (el del dominio actual).
FEWSHOT_UI = """EJEMPLO (UI/Landing Page):
{{"steps":[
  {{"step":1, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar HTML principal de la landing page", "tool_to_use":"generate_ui_code", "inputs":{{"description":"Landing page moderna con navbar, hero, galería y contacto"}}, "output_variable_name":"html_code", "expected_output_format":"html"}},
  {{"step":2, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar CSS para la landing page", "tool_to_use":"generate_css_code", "inputs":{{"description":"CSS oscuro con acentos dorados"}}, "output_variable_name":"css_code", "expected_output_format":"css"}},
  {{"step":3, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar JS para la landing page", "tool_to_use":"generate_js_code", "inputs":{{"description":"JS para smooth scroll y formulario de contacto"}}, "output_variable_name":"js_code", "expected_output_format":"javascript"}},
  {{"step":4, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Combinar HTML, CSS y JS en un unico index.html", "tool_to_use":"combine_ui_to_html", "inputs":{{"html":"{{{{html_code}}}}", "css":"{{{{css_code}}}}", "js":"{{{{js_code}}}}", "filepath":"index.html"}}, "output_variable_name":"html_final", "expected_output_format":"html"}}
]}}"""

FEWSHOT_SCRIPT = """EJEMPLO (Scripting — cadena genera→escribe→ejecuta→verifica):
{{"steps":[
  {{"step":1, "agent":"DeveloperAgent_CogniTeam", "action_description":"Generar script Python para analizar datos y producir output", "tool_to_use":"generate_textual_artifact", "inputs":{{"description_of_artifact":"Script Python que procesa datos y genera resultados"}}, "output_variable_name":"script_generado", "expected_output_format":"text"}},
  {{"step":2, "agent":"DeveloperAgent_CogniTeam", "action_description":"Escribir el script a disco", "tool_to_use":"write_file_sandboxed", "inputs":{{"relative_filepath":"script.py", "content":"{{{{script_generado}}}}"}}, "output_variable_name":"archivo_escrito", "expected_output_format":"text"}},
  {{"step":3, "agent":"DeveloperAgent_CogniTeam", "action_description":"Ejecutar el script para validar que funciona", "tool_to_use":"execute_terminal_command_safe", "inputs":{{"command":"python3 script.py"}}, "output_variable_name":"ejecucion", "expected_output_format":"text"}},
  {{"step":4, "agent":"DeveloperAgent_CogniTeam", "action_description":"Verificar que los archivos generados existen", "tool_to_use":"list_files_sandboxed", "inputs":{{"relative_dirpath":"."}}, "output_variable_name":"verificacion", "expected_output_format":"text"}}
]}}"""

FEWSHOT_CONTENT = """EJEMPLO (Generación de contenido):
{{"steps":[
  {{"step":1, "agent":"DeveloperAgent_CogniTeam", "action_description":"Generar el contenido solicitado", "tool_to_use":"generate_textual_artifact", "inputs":{{"description_of_artifact":"Contenido/documento según los requisitos de la tarea"}}, "output_variable_name":"contenido", "expected_output_format":"text"}},
  {{"step":2, "agent":"DeveloperAgent_CogniTeam", "action_description":"Persistir el contenido a disco como archivo", "tool_to_use":"write_file_sandboxed", "inputs":{{"relative_filepath":"output.txt", "content":"{{{{contenido}}}}"}}, "output_variable_name":"archivo", "expected_output_format":"text"}}
]}}"""

DOMAIN_FEWSHOT_EXAMPLES: Dict[str, str] = {
    "1_web_development": FEWSHOT_UI,
    "2_software_development": FEWSHOT_SCRIPT,
    "3_education": FEWSHOT_CONTENT,
    "4_graphic_design": FEWSHOT_CONTENT,
    "5_architecture_spatial": FEWSHOT_CONTENT,
    "6_marketing_growth": FEWSHOT_CONTENT,
    "7_audiovisual_management": FEWSHOT_CONTENT,
    "8_game_development": FEWSHOT_SCRIPT,
    "9_data_science": FEWSHOT_SCRIPT,
    "10_content_writing": FEWSHOT_CONTENT,
    "11_devops_infrastructure": FEWSHOT_SCRIPT,
    "12_cybersecurity": FEWSHOT_CONTENT,
    "13_legal_compliance": FEWSHOT_CONTENT,
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
12. IMPORTANTE: Encadena datos entre pasos usando {{output_variable_name}} en los inputs. Si el paso 2 extrae informacion (ej: {{paper_info}}), pasala al paso 3 como "description": "Landing page con: {{paper_info}}". NO generes contenido vacio o placeholder — los datos fluyen mediante referencias.

{fewshot_example}

NO incluyas comentarios ni texto fuera del JSON. Output ÚNICAMENTE el objeto JSON.

Herramientas disponibles:
{tools_description}

Agentes:
{agents_description}"""


class PlannerAgent:
    """Planner agent — generates execution plans via LLM."""
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
    if raw.endswith("```"):
        raw = raw[:-len("```")].strip()
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = raw[first: last + 1]
        try:
            json.loads(candidate)
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


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
    if domain and archetype:
        wm = generate_world_model(requirements, domain, archetype, tools_description=tools_description)
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
        print("  [World Model] No disponible. Procediendo sin simulación prospectiva.")

    plan = await generate_plan(
        planner_agent=planner_agent,
        requirements=requirements,
        tools_description=tools_description,
        agents_description=agents_description,
        replan_context=replan_context,
        domain=domain,
    )

    return {"action": "EXECUTE", "world_model": wm, "plan": plan}
