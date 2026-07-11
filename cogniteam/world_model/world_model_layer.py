from __future__ import annotations 
 
import asyncio 
import json 
import re 
from dataclasses import dataclass, field 
from datetime import datetime 
from typing import Any, Dict, List, Optional, Tuple 
 
import traceforge 
from pydantic import BaseModel, Field, field_validator 
 
 
# 
# 1. MODELOS DE DATOS (Pydantic) 
# 
 
class WorldModelBlock(BaseModel): 
    """Bloque de simulación prospectiva z_t = [z_t^traj     Equivalente textual al world model del paper de Tencent Youtu Lab."""
    trajectory: str = Field( 
        ..., 
        description="Roadmap abstracto de pasos futuros esperados. Sin spoilers concretos." 
    ) 
    gap: str = Field( 
        ..., 
        description="Análisis de qué información falta en el estado actual para ejecutar el plan."
    ) 
    confidence: int = Field( 
        ..., 
        ge=0, 
        le=100, 
        description="Estimación Q-value verbalizada [0-100] de probabilidad de éxito del plan."
    ) 
    keywords: List[str] = Field( 
        default_factory=list, 
        max_length=5, 
        description="Hasta 5 keywords que deberían manifestarse en la ejecución real para confirmar grounding."
    ) 
    analysis: str = Field( 
        ..., 
        description="Evaluación semántica del plan: riesgos, alternativas, supuestos críticos." 
    ) 
 
    @field_validator("keywords") 
    @classmethod 
    def max_five_keywords(cls, v: List[str]) -> List[str]: 
        if len(v) > 5: 
            raise ValueError("El paper recomienda máximo 5 keywords para evitar ruido.") 
        return [k.strip().lower() for k in v if k.strip()] 
 
 
class CalibratedThreshold(BaseModel): 
    """Entrada de la tabla de calibración episódica (emulación de FC-RL Brier score).""" 
    domain: str 
    archetype: str 
    predicted_confidence: float 
    actual_success: bool 
    timestamp: datetime = Field(default_factory=datetime.utcnow) 
 
    @property 
    def brier_error(self) -> float: 
        """Brier score para una observación binaria: (q - outcome)^2""" 
        outcome = 1.0 if self.actual_success else 0.0 
        return (self.predicted_confidence / 100.0 - outcome) ** 2 
 
 
class TaskManifest(BaseModel): 
    """Extensión del TaskManifest existente de CogniTeam.""" 
    task_description: str 
    domain: str 
    archetype: str 
    domain_confidence: float 
    world_model: Optional[WorldModelBlock] = None 
    calibrated_threshold: Optional[float] = None  # Umbral ajustado por historial 
 
 
# 
# 2. PROMPTS FEW-SHOT POR DOMINIO (Format-Eliciting SFT emulado) 
# 
 
# Prompt para el dominio 1_web_development con sus arquetipos. 
# Incluye ejemplos de world_model blocks para landing-page, ecommerce y saas-dashboard. 
WEB_DEV_WORLD_MODEL_PROMPT = """ 
You are the "World Model" layer inside the CogniTeam Planner Agent. 
Your job is NOT to execute the task, but to simulate the future trajectory if the plan is 
followed, 
identify gaps in the current state, and estimate the probability of success. 
 
CRITICAL RULES: 
1. NO SPOILERS: Use abstract placeholders (e.g., "expect to define a color palette"), NOT 
specific values from the task unless they are given. 
2. BE SPECIFIC: In the Trajectory, specify exact milestones/intents, avoiding generic phrases 
like "do the work". 
3. CONFIDENCE CALIBRATION: Be honest. If the task lacks critical inputs, confidence must 
be low. If the task is well-specified, confidence can be high. 
4. KEYWORDS: Extract up to 5 terms that MUST appear in the real execution for the plan to 
be considered grounded. 
 
OUTPUT FORMAT (strict JSON): 
{ 
  "trajectory": "Step-by-step roadmap...", 
  "gap": "Why the current state is insufficient...", 
  "confidence": 85, 
  "keywords": ["keyword1", "keyword2", ...], 
  "analysis": "Your evaluation of the plan..." 
} 
 
--- EXAMPLES --- 
 
Example 1 (Archetype: landing-page) 
Task: "Create a landing page for my renovation business showing services, before/after images, contact form and automatic budget."
{ 
  "trajectory": "1) Scaffold semantic HTML structure (hero, services grid, before/after gallery, 
contact section, footer). 2) Implement responsive CSS with brand variables. 3) Build contact 
form with client-side validation. 4) Add budget calculator logic in JS. 5) Validate accessibility 
and Core Web Vitals.", 
  "gap": "The current state lacks brand color palette, logo assets, and copy text. The budget logic requires pricing rules that have not been provided.",
  "confidence": 70, 
  "keywords": ["html", "responsive", "contact-form", "budget-calculator", "accessibility"], 
  "analysis": "The path is direct but requires clarification on branding and pricing rules. Without them, the landing page will be structurally correct but commercially incomplete."
} 
 
Example 2 (Archetype: ecommerce) 
Task: "Build an online store for handmade candles with Stripe payments." 
{ 
  "trajectory": "1) Define product catalog schema and database models. 2) Build product listing and detail pages. 3) Implement shopping cart session management. 4) Integrate
Stripe checkout with webhook confirmation. 5) Add order dashboard for the admin.", 
  "gap": "No product images, descriptions, or Stripe API keys are present. Tax rules and shipping zones are undefined.",
  "confidence": 55, 
  "keywords": ["stripe", "cart", "webhook", "product-schema", "order-dashboard"], 
  "analysis": "High uncertainty due to missing payment credentials and logistics rules. The technical path is clear, but business logic blocks are unresolved."
} 
 
Example 3 (Archetype: saas-dashboard) 
Task: "Create a SaaS analytics dashboard with user auth and real-time charts." 
{ 
  "trajectory": "1) Set up authentication flow (JWT/OAuth). 2) Design dashboard layout with sidebar navigation. 3) Integrate charting library for real-time data visualization. 4) Build API
endpoints for data aggregation. 5) Implement role-based access control.", 
  "gap": "The current state does not specify the data source, chart types required, or user roles. OAuth provider is not defined.",
  "confidence": 60, 
  "keywords": ["jwt", "dashboard", "charting", "api-endpoints", "rbac"], 
  "analysis": "The architecture is standard, but the absence of data schema and auth provider specifics introduces integration risk."
} 
 
--- TASK TO SIMULATE --- 
Domain: {domain} 
Archetype: {archetype} 
Task Description: {task_description} 
Interaction History: {history} 
 
Generate the world_model JSON now. 
""".strip() 
 
# Prompt para el Scoping Agent (prospective validation antes de clasificar) 
SCOPING_WORLD_MODEL_PROMPT = """ 
You are the "Prospective Scoping" layer inside the CogniTeam Scoping Agent. 
Before finalizing the domain/archetype classification, simulate whether the chosen 
archetype's standard workflow can actually resolve this task. 
CRITICAL RULES: 
1. If the task requires capabilities outside the chosen archetype's harness, flag it immediately.
2. Provide a confidence score that reflects whether the classification is appropriate, NOT just 
the LLM's generic confidence. 3. Identify cross-domain dependencies (e.g., a web task that also needs DevOps or Legal).
 
OUTPUT FORMAT (strict JSON): 
{ 
  "trajectory": "How the chosen archetype workflow would unfold...", 
  "gap": "What is missing for this archetype to succeed...", 
  "confidence": 75, 
  "keywords": ["keyword1", ...], 
  "analysis": "Is this the right archetype? Are there hidden dependencies?" 
} 
 
Chosen Domain: {domain} 
Chosen Archetype: {archetype} 
Task: {task_description} 
 
Generate the prospective validation JSON now. 
""".strip() 
 
# Prompt para el Debugger Agent (verificación de grounding) 
DEBUGGER_GROUNDING_PROMPT = """ 
You are the "Grounding Verifier" inside the CogniTeam Debugger Agent. 
Your job is to check whether the predicted keywords from the Planner's world model actually appeared in the execution artifacts.
 
CRITICAL RULES: 
1. Return a grounding score [0.0, 1.0] based on the fraction of keywords found. 
2. If critical keywords are missing, the plan is considered ungrounded. 
3. Suggest a corrective action if grounding is partial. 
 
OUTPUT FORMAT (strict JSON): 
{ 
  "grounding_score": 0.6, 
  "found_keywords": ["keyword1"], 
  "missing_keywords": ["keyword2", "keyword3"], 
  "corrective_action": "Re-plan step X to include missing Y..." 
} 
 
Predicted Keywords: {predicted_keywords} 
Execution Artifacts Summary: {artifacts_summary} 
 
Generate the grounding verification JSON now. 
""".strip() 
 
# 
# 3. GENERADOR DE WORLD MODEL (Cliente LLM agnóstico) 
# 
 
class LLMClient: 
    """Interfaz abstracta para el cliente LLM de CogniTeam. 
    Adaptar a tu implementación concreta (Groq, Ollama, LiteLLM).""" 
    async def generate(self, prompt: str, temperature: float = 0.6, max_tokens: int = 2048) -> str:
        raise NotImplementedError("Usa tu cliente LLM real (Groq/Ollama).") 
 
 
class WorldModelGenerator: 
    """Genera WorldModelBlocks usando few-shot prompting (emulación de FE-SFT). 
    No requiere entrenamiento del modelo base.""" 
 
    def __init__(self, llm_client: LLMClient): 
        self.llm_client = llm_client 
        self._prompts_by_domain = { 
            "1_web_development": WEB_DEV_WORLD_MODEL_PROMPT, 
            # Añadir más dominios aquí siguiendo el mismo patrón 
        } 
 
    @traceforge.trace(agent="world_model.simulate", tags=["llm", "world_model"]) 
    async def generate( 
        self, 
        task_description: str, 
        domain: str, 
        archetype: str, 
        history: str = "", 
        temperature: float = 0.6, 
    ) -> WorldModelBlock: 
        """Genera el bloque de world model para una tarea dada.""" 
        prompt_template = self._prompts_by_domain.get( 
            domain, 
            # Fallback genérico si el dominio no tiene prompt especializado 
            WEB_DEV_WORLD_MODEL_PROMPT.replace("landing-page", archetype) 
            .replace("ecommerce", archetype) 
            .replace("saas-dashboard", archetype), 
        ) 
 
        prompt = prompt_template.format( 
            domain=domain, 
            archetype=archetype, 
            task_description=task_description, 
            history=history or "No previous interactions.", 
        ) 
 
        raw_response = await self.llm_client.generate( 
            prompt=prompt, temperature=temperature, max_tokens=2048 
        ) 
 
        return self._parse_world_model(raw_response) 
 
    @staticmethod 
    def _parse_world_model(raw: str) -> WorldModelBlock: 
        """Extrae y valida el JSON del world model desde la respuesta del LLM.""" 
        # Buscar bloque JSON 
        match = re.search(r"\{.*\}", raw, re.DOTALL) 
        if not match: 
            raise ValueError(f"No se encontró JSON en la respuesta del world model: {raw[:200]}")
 
        data = json.loads(match.group()) 
        return WorldModelBlock(**data) 
 
 
# 
# 4. AGENTES MODIFICADOS (Integración en el pipeline) 
# 
 
class ProspectivePlannerAgent: 
    """Planner Agent de CogniTeam extendido con capacidad prospectiva. Genera un world_model block ANTES de emitir el plan de ejecución."""

 
    def __init__( 
        self, 
        llm_client: LLMClient, 
        calibration_store: CalibrationStore, 
        default_threshold: float = 0.65, 
    ): 
        self.wm_generator = WorldModelGenerator(llm_client) 
        self.calibration_store = calibration_store 
        self.default_threshold = default_threshold 
 
    async def plan(self, manifest: TaskManifest) -> Tuple[WorldModelBlock, Dict[str, Any]]: 
        """Pipeline de planificación prospectiva: 1. Generar world_model. 
        2. Calibrar contra historial. 
        3. Decidir: ejecutar, reclarificar, o abortar.""" 
        # 1. Generar simulación prospectiva 
        world_model = await self.wm_generator.generate( 
            task_description=manifest.task_description, 
            domain=manifest.domain, 
            archetype=manifest.archetype, 
        ) 
 
        # 2. Obtener umbral calibrado para este dominio/arquetipo 
        threshold = self.calibration_store.get_threshold( 
            manifest.domain, manifest.archetype, default=self.default_threshold 
        ) 
 
        confidence_ratio = world_model.confidence / 100.0 
 
        # 3. Decisión prospectiva 
        if confidence_ratio < threshold: 
            # Señal de retorno al Scoping Agent para clarificación 
            return world_model, { 
                "action": "RECLARIFY", 
                "reason": f"Confidence {world_model.confidence}% < threshold {threshold*100:.0f}%",
                "gaps": [world_model.gap], 
            } 
 
        # 4. Si pasa el umbral, proceder al plan normal (aquí iría tu Planner original) 
        execution_plan = self._build_execution_plan(manifest, world_model) 
        return world_model, { 
            "action": "EXECUTE", 
            "plan": execution_plan, 
            "grounding_keywords": world_model.keywords, 
        } 
 
    def _build_execution_plan(self, manifest: TaskManifest, wm: WorldModelBlock) -> Dict[str, Any]:
        """Construye el plan de ejecución enriquecido con el world model. Integración con tu Planner Agent existente."""

        return { 
            "domain": manifest.domain, 
            "archetype": manifest.archetype, 
            "phases": wm.trajectory.split("\n"), 
            "validation_criteria": wm.keywords, 
            "risk_notes": wm.analysis, 
            "estimated_success_probability": wm.confidence, 
        } 
 
 
class ProspectiveScopingAgent: 
    """Scoping Agent extendido con validación prospectiva. Después de clasificar, simula si el arquetipo elegido puede resolver la tarea."""

 
    def __init__(self, llm_client: LLMClient): 
        self.wm_generator = WorldModelGenerator(llm_client) 
 
    async def validate_classification( 
        self, task_description: str, proposed_domain: str, proposed_archetype: str 
    ) -> Tuple[bool, WorldModelBlock]: 
        """Retorna (is_valid, world_model). 
        is_valid = False si la simulación muestra que el arquetipo es inadecuado.""" 
        prompt = SCOPING_WORLD_MODEL_PROMPT.format( 
            domain=proposed_domain, 
            archetype=proposed_archetype, 
            task_description=task_description, 
        ) 
        raw = await self.wm_generator.llm_client.generate(prompt=prompt, temperature=0.4) 
        wm = WorldModelGenerator._parse_world_model(raw) 
 
        # Si la confianza de adecuación del arquetipo es muy baja, rechazar clasificación 
        is_valid = wm.confidence >= 50 
        return is_valid, wm 
 
 
class ProspectiveDebuggerAgent: 
    """Debugger Agent extendido con verificación de grounding. Compara keywords predichas vs. artefactos reales."""

 
    def __init__(self, llm_client: LLMClient): 
        self.llm_client = llm_client 
 
    async def verify_grounding( 
        self, predicted_keywords: List[str], artifacts_summary: str 
    ) -> Dict[str, Any]: 
        """Retorna score de grounding y acción correctiva.""" 
        prompt = DEBUGGER_GROUNDING_PROMPT.format( 
            predicted_keywords=json.dumps(predicted_keywords), 
            artifacts_summary=artifacts_summary, 
        ) 
        raw = await self.llm_client.generate(prompt=prompt, temperature=0.3, 
max_tokens=1024) 
        match = re.search(r"\{.*\}", raw, re.DOTALL) 
        if not match: 
            return {"grounding_score": 0.0, "corrective_action": "Re-plan from scratch"} 
        return json.loads(match.group()) 
 
 
# 
# 5. MEMORIA DE CALIBRACIÓN (Emulación de FC-RL Brier Score) 
# 
 
class CalibrationStore: 
    """Almacén episódico de calibración de confianza. Reemplaza el FC-RL del paper con una tabla heurística actualizada por experiencia. 
    Puede persistirse en H-MEM / MATM / Fast-Slow de CogniTeam.""" 
 
    def __init__(self): 
        # Estructura: {(domain, archetype): [CalibratedThreshold, ...]} 
        self._history: Dict[Tuple[str, str], List[CalibratedThreshold]] = {} 
 
    def record(self, domain: str, archetype: str, predicted_confidence: int, actual_success: bool):
        """Registrar resultado para actualizar calibración.""" 
        entry = CalibratedThreshold( 
            domain=domain, 
            archetype=archetype, 
            predicted_confidence=predicted_confidence, 
            actual_success=actual_success, 
        ) 
        key = (domain, archetype) 
        self._history.setdefault(key, []).append(entry) 
 
    def get_threshold(self, domain: str, archetype: str, default: float = 0.65) -> float: 
        """Calcula umbral dinámico basado en historial Brier. Si el modelo suele sobrestimar, sube el umbral. Si subestima, lo baja."""

        key = (domain, archetype) 
        entries = self._history.get(key, []) 
        if len(entries) < 5: 
            return default  # Datos insuficientes 
 
        # Calcular Brier score promedio 
        brier_scores = [e.brier_error for e in entries[-20:]]  # Ventana de 20 
        avg_brier = sum(brier_scores) / len(brier_scores) 
 
        # Ajuste heurístico: si Brier es alto (mala calibración), subir umbral para ser más conservador
        adjustment = avg_brier * 0.5  # Factor empírico 
        threshold = min(0.95, default + adjustment) 
        return threshold 
 
    def get_calibration_report(self, domain: str, archetype: str) -> Dict[str, float]: 
        """Reporte de calibración para debugging.""" 
        entries = self._history.get((domain, archetype), []) 
        if not entries: 
            return {} 
        successes = sum(1 for e in entries if e.actual_success) 
        briers = [e.brier_error for e in entries] 
        return { 
            "total_samples": len(entries), 
            "success_rate": successes / len(entries), 
            "mean_brier": sum(briers) / len(briers), 
            "last_threshold": self.get_threshold(domain, archetype), 
        } 
 
 
# 
# 6. ORQUESTADOR PROSPECTIVO (Punto de integración principal) 
# 
 
class ProspectiveOrchestrator: 
    """Wrapper del Orchestrator de CogniTeam que inyecta decisiones prospectivas en los puntos críticos del pipeline."""

 
    def __init__( 
        self, 
        scoping_agent: ProspectiveScopingAgent, 
        planner_agent: ProspectivePlannerAgent, 
        debugger_agent: ProspectiveDebuggerAgent, 
        calibration_store: CalibrationStore, 
    ): 
        self.scoping = scoping_agent 
        self.planner = planner_agent 
        self.debugger = debugger_agent 
        self.calibration = calibration_store 
 
    async def execute_task(self, task_description: str) -> Dict[str, Any]: 
        """Pipeline end-to-end con world model layer."""
        # --- FASE 1: SCOPING + VALIDACIÓN PROSPECTIVA --- 
        # (Aquí iría tu Scoping Agent original; asumimos que devuelve domain/archetype/conf) 
        domain, archetype, domain_conf = await self._original_scoping(task_description) 
 
        is_valid, scoping_wm = await self.scoping.validate_classification( 
            task_description, domain, archetype 
        ) 
 
        if not is_valid: 
            return { 
                "status": "RECLARIFY", 
                "reason": "Scoping validation failed: chosen archetype is likely inadequate.", 
                "world_model": scoping_wm.model_dump(), 
                "suggested_action": "Ask user for clarification or switch archetype.", 
            } 
 
        manifest = TaskManifest( 
            task_description=task_description, 
            domain=domain, 
            archetype=archetype, 
            domain_confidence=domain_conf, 
        ) 
 
        # --- FASE 2: PLANNING PROSPECTIVO --- 
        planner_wm, plan_decision = await self.planner.plan(manifest) 
 
        if plan_decision["action"] == "RECLARIFY": 
            return { 
                "status": "RECLARIFY", 
                "reason": plan_decision["reason"], 
                "gaps": plan_decision["gaps"], 
                "world_model": planner_wm.model_dump(), 
            } 
 
        # --- FASE 3: EJECUCIÓN (Developer Agent, etc.) --- 
        # (Aquí iría tu ejecución real) 
        execution_result = await self._original_execute(plan_decision["plan"]) 
 
        # --- FASE 4: DEBUGGING + GROUNDING (post-ejecución) --- 
        grounding = await self.debugger.verify_grounding( 
            predicted_keywords=planner_wm.keywords, 
            artifacts_summary=execution_result.get("artifacts_summary", ""), 
        ) 
 
        # --- FASE 5: CALIBRACIÓN (actualización de la tabla episódica) --- 
        success = execution_result.get("success", False) and grounding["grounding_score"] > 0.6 
        self.calibration.record( 
            domain=domain, 
            archetype=archetype, 
            predicted_confidence=planner_wm.confidence, 
            actual_success=success, 
        ) 
 
        return { 
            "status": "SUCCESS" if success else "PARTIAL_FAILURE", 
            "world_model": planner_wm.model_dump(), 
            "plan": plan_decision["plan"], 
            "grounding": grounding, 
            "calibration_report": self.calibration.get_calibration_report(domain, archetype), 
        } 
 
    # --- Stubs para integración con tu código existente --- 
    async def _original_scoping(self, task: str) -> Tuple[str, str, float]: 
        # TODO: Reemplazar con tu Scoping Agent real 
        return "1_web_development", "landing-page", 0.85 
 
    async def _original_execute(self, plan: Dict[str, Any]) -> Dict[str, Any]: 
        # TODO: Reemplazar con tu Developer/Orchestrator real 
        return {"success": True, "artifacts_summary": "HTML, CSS, JS generated with contact form."}
 
 
# 
# 7. EJEMPLO DE USO (main.py) 
# 
 
async def main(): 
    """Ejemplo de integración completa.""" 
 
    # Cliente LLM dummy (reemplazar con tu cliente Groq/Ollama) 
    class DummyLLM(LLMClient): 
        async def generate(self, prompt: str, temperature: float = 0.6, max_tokens: int = 2048) -> str:
            """Simula respuesta del modelo.""" 
            return json.dumps({ 
                "trajectory": "1) Scaffold HTML. 2) Style with CSS. 3) Add JS interactivity. 4) Validate.",
                "gap": "Missing brand colors and copy text.", 
                "confidence": 72, 
                "keywords": ["html", "css", "js", "responsive", "validation"], 
                "analysis": "Standard path with moderate risk due to missing assets." 
            }) 
 
    llm = DummyLLM() 
    calibration = CalibrationStore() 
 
    orchestrator = ProspectiveOrchestrator( 
        scoping_agent=ProspectiveScopingAgent(llm), 
        planner_agent=ProspectivePlannerAgent(llm, calibration, default_threshold=0.65), 
        debugger_agent=ProspectiveDebuggerAgent(llm), 
        calibration_store=calibration, 
    ) 
 
    result = await orchestrator.execute_task( 
        "Create a landing page for my renovation business with contact form and budget calculator."
    ) 
    print(json.dumps(result, indent=2, default=str)) 
 
 
if __name__ == "__main__": 
    asyncio.run(main()) 