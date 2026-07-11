from __future__ import annotations 
 
import re 
import yaml 
from pathlib import Path 
from typing import Any, Dict, List, Optional, Tuple 
 
from pydantic import BaseModel, Field 
 
 
class ArchetypeInput(BaseModel): 
    """Definición de un input para un arquetipo.""" 
    type: str 
    default: Any 
    extract_from_task: Optional[List[str]] = None 
    extract_regex: Optional[str] = None 
    min: Optional[int] = None 
    max: Optional[int] = None 
 
 
class ArchetypeStructureStep(BaseModel): 
    """Paso en la estructura de un arquetipo.""" 
    step: int 
    action: str 
    file: str 
    required_tags: Optional[List[str]] = None 
    required_patterns: Optional[List[str]] = None 
    required_sections: Optional[List[str]] = None 
    required_features: Optional[List[str]] = None 
    required_types: Optional[List[str]] = None 
    conditional: Optional[str] = None 
 
 
class ArchetypeVerification(BaseModel): 
    """Reglas de verificación post-generación.""" 
    required_files: Optional[List[str]] = None 
    required_tags: Optional[List[str]] = None 
    required_classes: Optional[List[str]] = None 
    required_patterns: Optional[List[str]] = None 
    forbidden_patterns: Optional[List[str]] = None 
    min_length: Optional[int] = None 
    max_length: Optional[int] = None 
 
 
class ArchetypeThresholds(BaseModel): 
    """Umbrales de decisión.""" 
    min_confidence: int = 60 
    max_re_retries: int = 1 
 
 
class ArchetypeConfig(BaseModel): 
    """Configuración completa de un arquetipo.""" 
    description: str 
    priority: int = 5 
    estimated_success_rate: float = 0.75 
    inputs: Dict[str, ArchetypeInput] = Field(default_factory=dict) 
    structure: List[ArchetypeStructureStep] = Field(default_factory=list) 
    llm_prompt: str 
    verification: ArchetypeVerification = Field(default_factory=ArchetypeVerification) 
    grounding_keywords: List[str] = Field(default_factory=list) 
    thresholds: ArchetypeThresholds = Field(default_factory=ArchetypeThresholds) 
 
 
class DomainConfig(BaseModel): 
    """Configuración de un dominio.""" 
    description: str 
    icon: str = "📦" 
    default_model: str = "llama-3.3-70b-versatile" 
    archetypes: Dict[str, ArchetypeConfig] 
 
 
class CageDefaults(BaseModel): 
    """Configuración por defecto de la jaula.""" 
    llm: Dict[str, Any] 
    cage: Dict[str, Any] 
    thresholds: Dict[str, Any] 
 
 
class CalibrationConfig(BaseModel): 
    """Configuración de calibración.""" 
    initial_thresholds: Dict[str, float] 
    brier_learning_rate: float = 0.1 
    min_samples_for_calibration: int = 5 
    overconfidence_penalty: float = 0.15 
    underconfidence_bonus: float = 0.05 
 
 
class CogniTeamConfig(BaseModel): 
    """Configuración raíz de CogniTeam.""" 
    version: str 
    metadata: Dict[str, Any] 
    defaults: CageDefaults 
    domains: Dict[str, DomainConfig] 
    fallbacks: Dict[str, Any] 
    calibration: CalibrationConfig 
 
 
class YamlArchetypeLoader: 
    """Carga cogniteam_archetypes.yaml y lo convierte en objetos Python. 
    """

    def __init__(self, config_path: str = "cogniteam_archetypes.yaml"): 
        self.config_path = Path(config_path) 
        self._config: Optional[CogniTeamConfig] = None 
        self._load() 
 
    def _load(self): 
        """Carga y valida el YAML.""" 
        if not self.config_path.exists(): 
            raise FileNotFoundError(f"No se encontró {self.config_path}") 
 
        with open(self.config_path, "r", encoding="utf-8") as f: 
            raw = yaml.safe_load(f) 
 
        self._config = CogniTeamConfig(**raw) 
        print(f"[✅] Configuración cargada: {len(self._config.domains)} dominios, " 
              f"{sum(len(d.archetypes) for d in self._config.domains.values())} arquetipos") 
 
    def get_archetype(self, domain: str, archetype: str) -> Optional[ArchetypeConfig]: 
        """Obtiene la configuración de un arquetipo.""" 
        domain_cfg = self._config.domains.get(domain) 
        if not domain_cfg: 
            return None 
        return domain_cfg.archetypes.get(archetype) 
 
    def get_all_archetypes(self) -> List[Tuple[str, str, ArchetypeConfig]]: 
        """Lista todos los arquetipos como (domain, archetype_name, config).""" 
        result = [] 
        for domain_name, domain_cfg in self._config.domains.items(): 
            for archetype_name, archetype_cfg in domain_cfg.archetypes.items(): 
                result.append((domain_name, archetype_name, archetype_cfg)) 
        return result 
 
    def get_llm_prompt(self, domain: str, archetype: str, variables: Dict[str, Any]) -> Optional[str]:
        """Renderiza el prompt LLM con variables Jinja2-style.""" 
        cfg = self.get_archetype(domain, archetype) 
        if not cfg: 
            return None 
 
        prompt = cfg.llm_prompt 
 
        # Renderizar variables simples {{ variable }} 
        for key, value in variables.items(): 
            placeholder = f"{{{{ {key} }}}}" 
            if placeholder in prompt: 
                prompt = prompt.replace(placeholder, str(value)) 
 
        # Renderizar condicionales {% if variable %}...{% endif %} 
        prompt = self._render_conditionals(prompt, variables) 
 
        return prompt 
 
    def _render_conditionals(self, prompt: str, variables: Dict[str, Any]) -> str: 
        """Renderiza bloques condicionales Jinja2-lite.""" 
        # Patrón: {% if condition %}content{% endif %} 
        pattern = r'{%\\s*if\\s+(.+?)\\s*%}(.*?){%\\s*endif\\s*%}' 
 
        def replace_conditional(match): 
            condition = match.group(1).strip() 
            content = match.group(2) 
 
            # Evaluar condición simple 
            # Soporta: variable, not variable, variable == 'value', variable | default('x') 
            if "|" in condition and "default" in condition: 
                # variable | default('x') → siempre verdadero si existe 
                var_name = condition.split("|")[0].strip() 
                return content if var_name in variables and variables[var_name] else "" 
 
            if condition.startswith("not "): 
                var_name = condition[4:].strip() 
                return content if not variables.get(var_name, False) else "" 
 
            if "==" in condition: 
                var_name, expected = condition.split("==", 1) 
                var_name = var_name.strip() 
                expected = expected.strip().strip("'\"").strip()
                return content if str(variables.get(var_name, "")) == expected else ""
 
            # Condición simple: truthy check 
            value = variables.get(condition, False) 
            if isinstance(value, bool): 
                return content if value else "" 
            return content if value else "" 
 
        # Aplicar reemplazo iterativamente hasta que no haya más cambios 
        prev = None 
        while prev != prompt: 
            prev = prompt 
            prompt = re.sub(pattern, replace_conditional, prompt, flags=re.DOTALL) 
 
        return prompt 
 
    def get_verification_rules(self, domain: str, archetype: str) -> ArchetypeVerification: 
        """Obtiene reglas de verificación.""" 
        cfg = self.get_archetype(domain, archetype) 
        if not cfg: 
            return ArchetypeVerification() 
        return cfg.verification 
 
    def get_thresholds(self, domain: str, archetype: str) -> ArchetypeThresholds: 
        """Obtiene umbrales de decisión.""" 
        cfg = self.get_archetype(domain, archetype) 
        if not cfg: 
            return ArchetypeThresholds() 
        return cfg.thresholds 
 
    def get_grounding_keywords(self, domain: str, archetype: str) -> List[str]: 
        """Obtiene keywords para verificación de grounding.""" 
        cfg = self.get_archetype(domain, archetype) 
        if not cfg: 
            return [] 
        return cfg.grounding_keywords 
 
    def get_structure(self, domain: str, archetype: str) -> List[ArchetypeStructureStep]: 
        """Obtiene estructura de pasos.""" 
        cfg = self.get_archetype(domain, archetype) 
        if not cfg: 
            return [] 
        return cfg.structure 
 
    def get_input_definitions(self, domain: str, archetype: str) -> Dict[str, ArchetypeInput]: 
        """Obtiene definiciones de inputs.""" 
        cfg = self.get_archetype(domain, archetype) 
        if not cfg: 
            return {} 
        return cfg.inputs 
 
    def get_default_config(self) -> CageDefaults: 
        """Obtiene configuración por defecto.""" 
        return self._config.defaults 
 
    def get_calibration_config(self) -> CalibrationConfig: 
        """Obtiene configuración de calibración.""" 
        return self._config.calibration 
 
    def get_fallback(self, fallback_name: str = "generic-task") -> Optional[Dict[str, Any]]: 
        """Obtiene configuración de fallback.""" 
        return self._config.fallbacks.get(fallback_name) 
 
    def extract_inputs_from_task(self, task: str, domain: str, archetype: str) -> Dict[str, Any]: 
        """Extrae inputs de la descripción de la tarea usando las reglas del YAML.""" 
        inputs_def = self.get_input_definitions(domain, archetype) 
        extracted = {} 
        task_lower = task.lower() 
 
        for input_name, input_def in inputs_def.items(): 
            value = input_def.default 
 
            # Intentar extraer de lista de keywords 
            if input_def.extract_from_task: 
                if isinstance(input_def.extract_from_task, dict): 
                    # Mapeo de valores (ej: device types) 
                    for mapped_value, keywords in input_def.extract_from_task.items(): 
                        if any(kw.lower() in task_lower for kw in keywords): 
                            if input_def.type == "list": 
                                if not isinstance(value, list): 
                                    value = [] 
                                value.append(mapped_value) 
                            else: 
                                value = mapped_value 
                                break 
                elif isinstance(input_def.extract_from_task, list): 
                    # Lista simple de keywords 
                    if any(kw.lower() in task_lower for kw in input_def.extract_from_task): 
                        if input_def.type == "bool": 
                            value = True 
                        elif input_def.type == "str": 
                            value = input_def.extract_from_task[0] 
 
            # Intentar extraer con regex 
            if input_def.extract_regex: 
                match = re.search(input_def.extract_regex, task, re.I) 
                if match: 
                    raw_value = match.group(1) 
                    if input_def.type == "int": 
                        value = int(raw_value) 
                        if input_def.min is not None: 
                            value = max(value, input_def.min) 
                        if input_def.max is not None: 
                            value = min(value, input_def.max) 
                    elif input_def.type == "str": 
                        value = raw_value.strip() 
 
            extracted[input_name] = value 
 
        return extracted 
 
    def find_archetype_by_keywords(self, task: str) -> Optional[Tuple[str, str]]: 
        """Busca el mejor arquetipo por keywords (clasificación determinista).""" 
        task_lower = task.lower() 
        best_match = None 
        best_score = 0 
 
        for domain_name, domain_cfg in self._config.domains.items(): 
            for archetype_name, archetype_cfg in domain_cfg.archetypes.items(): 
                score = 0 
 
                # Puntuar por keywords de grounding 
                keywords = archetype_cfg.grounding_keywords 
                for kw in keywords: 
                    if kw.lower() in task_lower: 
                        score += 1 
 
                # Puntuar por inputs que matchean 
                for input_def in archetype_cfg.inputs.values(): 
                    if input_def.extract_from_task: 
                        if isinstance(input_def.extract_from_task, list): 
                            if any(kw.lower() in task_lower for kw in input_def.extract_from_task): 
                                score += 2 
                        elif isinstance(input_def.extract_from_task, dict): 
                            for keywords_list in input_def.extract_from_task.values(): 
                                if any(kw.lower() in task_lower for kw in keywords_list): 
                                    score += 2 
 
                # Aplicar prioridad 
                score *= archetype_cfg.priority / 10 
 
                if score > best_score: 
                    best_score = score 
                    best_match = (domain_name, archetype_name) 
 
        return best_match 
 
    def validate_output(self, code: str, domain: str, archetype: str) -> Dict[str, Any]: 
        """Valida el output generado contra las reglas del YAML.""" 
        rules = self.get_verification_rules(domain, archetype) 
        errors = [] 
        warnings = [] 
 
        # Verificar archivos requeridos (si se especifican) 
        if rules.required_files: 
            for fname in rules.required_files: 
                if f"=== {fname} ===" not in code and fname not in code: 
                    warnings.append(f"Posible falta de archivo: {fname}") 
 
        # Verificar tags requeridos 
        if rules.required_tags: 
            for tag in rules.required_tags: 
                if tag.lower() not in code.lower(): 
                    errors.append(f"Falta tag requerido: {tag}") 
 
        # Verificar clases requeridas 
        if rules.required_classes: 
            for cls in rules.required_classes: 
                if f'class="{cls}"' not in code and f"class='{cls}'" not in code: 
                    errors.append(f"Falta clase requerida: {cls}") 
 
        # Verificar patrones requeridos 
        if rules.required_patterns: 
            for pattern in rules.required_patterns: 
                if pattern.lower() not in code.lower(): 
                    warnings.append(f"Posible falta de patrón: {pattern}") 
 
        # Verificar patrones prohibidos 
        if rules.forbidden_patterns: 
            for pattern in rules.forbidden_patterns: 
                if pattern in code: 
                    errors.append(f"[SEGURIDAD] Patrón prohibido: {pattern}") 
 
        # Verificar longitud 
        if rules.min_length and len(code) < rules.min_length: 
            errors.append(f"Código demasiado corto: {len(code)} < {rules.min_length}") 
        if rules.max_length and len(code) > rules.max_length: 
            warnings.append(f"Código muy largo: {len(code)} > {rules.max_length}") 
 
        return { 
            "passed": len(errors) == 0, 
            "errors": errors, 
            "warnings": warnings, 
            "score": max(0, 1.0 - len(errors) * 0.3 - len(warnings) * 0.1), 
        } 
 
 
# 
# INTEGRACIÓN CON DETERMINISTIC_CAGE 
# 
 
def create_cage_from_yaml(config_path: str = "cogniteam_archetypes.yaml"): 
    """Crea una instancia de DeterministicCage configurada desde YAML."""
