from __future__ import annotations 
 
import asyncio 
import json 
import re 
from dataclasses import dataclass, field 
from datetime import datetime 
from enum import Enum, auto 
from typing import Any, Dict, List, Optional, Tuple, Type, Union 
 
from jinja2 import Template, Environment, BaseLoader 
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator 
 
 
# 
# SECCIÓN 1: ESTADOS Y MODELOS BASE 
# 
 
class CageState(Enum): 
    """Estados finitos de la jaula determinista.""" 
    PARSE = auto()           # Extraer entidades de la tarea cruda 
    CLASSIFY = auto()        # Determinar dominio/arquetipo sin LLM 
    VALIDATE = auto()        # Validar que los inputs son suficientes 
    TEMPLATE = auto()        # Renderizar plan/plantilla determinista 
    WORLD_MODEL = auto()     # Validación prospectiva (opcional, con LLM) 
    EXECUTE = auto()         # LLM 70B (único punto costoso) 
    VERIFY = auto()          # Verificación estructural sin LLM 
    CALIBRATE = auto()       # Actualizar tabla de calibración 
    DONE = auto()            # Entrega o reclarificación 
    FAILED = auto()          # Fallo irrecuperable 
 
 
class CageResult(BaseModel): 
    """Resultado de la ejecución de la jaula.""" 
    state: CageState 
    success: bool 
    output: Optional[Dict[str, Any]] = None 
    error: Optional[str] = None 
    llm_calls_used: int = 0 
    reclarification_needed: bool = False 
    reclarification_questions: List[str] = Field(default_factory=list) 
    warnings: List[str] = Field(default_factory=list) 
    execution_time_ms: Optional[int] = None 
 
 
# 
# SECCIÓN 2: SCHEMAS PYDANTIC POR DOMINIO (13 dominios) 
# 
 
class WebTaskInputs(BaseModel): 
    """Schema determinista para el dominio 1_web_development.""" 
    has_brand_assets: bool = Field(default=False, description="Logo, favicon, etc.") 
    has_copy_text: bool = Field(default=False, description="Textos de marketing") 
    has_color_palette: bool = Field(default=False, description="Colores de marca") 
    num_pages: int = Field(default=1, ge=1, le=5, description="Máximo 5 en free tier") 
    needs_interactivity: bool = Field(default=False, description="JS interactivo") 
    needs_backend: bool = Field(default=False, description="API/Backend integrado") 
    target_devices: List[str] = Field(default_factory=lambda: ["desktop", "mobile"]) 
    seo_required: bool = Field(default=False) 
    auth_required: bool = Field(default=False) 
    payment_required: bool = Field(default=False) 
 
    @field_validator("num_pages") 
    @classmethod 
    def max_pages_for_free_tier(cls, v: int) -> int: 
        if v > 5: 
            raise ValueError("Free tier: máximo 5 páginas por tarea") 
        return v 
 
    @field_validator("target_devices") 
    @classmethod 
    def valid_devices(cls, v: List[str]) -> List[str]: 
        allowed = {"desktop", "mobile", "tablet", "tv"} 
        invalid = set(v) - allowed 
        if invalid: 
            raise ValueError(f"Dispositivos no soportados: {invalid}") 
        return v 
 
 
class SoftwareTaskInputs(BaseModel): 
    """Schema para el dominio 2_software_development.""" 
    project_type: str = Field(..., description="api, mobile, desktop, cli") 
    language_preference: Optional[str] = Field(default=None) 
    needs_database: bool = Field(default=False) 
    needs_auth: bool = Field(default=False) 
    needs_tests: bool = Field(default=False) 
    max_endpoints: int = Field(default=10, ge=1, le=20) 
 
    @field_validator("project_type") 
    @classmethod 
    def valid_project_type(cls, v: str) -> str: 
        allowed = {"api", "mobile", "desktop", "cli", "library"} 
        if v not in allowed: 
            raise ValueError(f"Tipo {v} no soportado. Use: {allowed}") 
        return v 
 
 
class Design3DTaskInputs(BaseModel): 
    """Schema para el dominio 3_3d_design.""" 
    output_format: str = Field(default="glb", description="glb, fbx, obj") 
    polygon_budget: int = Field(default=50000, ge=1000, le=100000) 
    texture_resolution: str = Field(default="2k", description="1k, 2k, 4k") 
    animation_required: bool = Field(default=False) 
 
    @field_validator("output_format") 
    @classmethod 
    def valid_format(cls, v: str) -> str: 
        allowed = {"glb", "gltf", "fbx", "obj", "stl"} 
        if v not in allowed: 
            raise ValueError(f"Formato {v} no soportado") 
        return v 
 
 
class GraphicDesignTaskInputs(BaseModel): 
    """Schema para el dominio 4_graphic_design.""" 
    deliverable_type: str = Field(..., description="logo, branding, ui-kit, poster") 
    color_mode: str = Field(default="rgb", description="rgb, cmyk, pantone") 
    dimensions: Optional[str] = Field(default=None, description="1920x1080, A4, etc.") 
    vector_required: bool = Field(default=False) 
 
    @field_validator("deliverable_type") 
    @classmethod 
    def valid_type(cls, v: str) -> str: 
        allowed = {"logo", "branding", "ui-kit", "poster", "flyer", "social-media"} 
        if v not in allowed: 
            raise ValueError(f"Tipo {v} no soportado") 
        return v 
 
 
class VideoTaskInputs(BaseModel): 
    """Schema para el dominio 5_video_production.""" 
    duration_seconds: int = Field(default=60, ge=5, le=300) 
    resolution: str = Field(default="1080p", description="720p, 1080p, 4k") 
    style: str = Field(default="motion-graphics", description="live-action, motion-graphics") 
    has_script: bool = Field(default=False) 
    has_footage: bool = Field(default=False) 
 
    @field_validator("resolution") 
    @classmethod 
    def valid_resolution(cls, v: str) -> str: 
        allowed = {"720p", "1080p", "1440p", "4k"} 
        if v not in allowed: 
            raise ValueError(f"Resolución {v} no soportada en free tier") 
        return v 
 
 
class MarketingTaskInputs(BaseModel): 
    """Schema para el dominio 6_marketing_growth.""" 
    campaign_type: str = Field(..., description="seo, social-media, email, ads") 
    target_audience: Optional[str] = Field(default=None) 
    budget_range: Optional[str] = Field(default=None) 
    timeline_days: int = Field(default=30, ge=1, le=90) 
 
    @field_validator("campaign_type") 
    @classmethod 
    def valid_campaign(cls, v: str) -> str: 
        allowed = {"seo", "social-media", "email", "ads", "content", "influencer"} 
        if v not in allowed: 
            raise ValueError(f"Campaña {v} no soportada") 
        return v 
 
 
class ArchitectureTaskInputs(BaseModel): 
    """Schema para el dominio 7_software_architecture.""" 
    architecture_pattern: str = Field(..., description="microservices, monolith, serverless") 
    scale_users: int = Field(default=1000, ge=10, le=1000000) 
    compliance_required: List[str] = Field(default_factory=list) 
    cloud_provider: Optional[str] = Field(default=None) 
 
    @field_validator("architecture_pattern") 
    @classmethod 
    def valid_pattern(cls, v: str) -> str: 
        allowed = {"microservices", "monolith", "serverless", "event-driven", "layered"} 
        if v not in allowed: 
            raise ValueError(f"Patrón {v} no soportado") 
        return v 
 
 
class GameDevTaskInputs(BaseModel): 
    """Schema para el dominio 8_game_development.""" 
    game_genre: str = Field(..., description="platformer, rpg, puzzle, fps") 
    target_platform: str = Field(default="web", description="web, mobile, desktop") 
    multiplayer: bool = Field(default=False) 
    monetization: Optional[str] = Field(default=None) 
 
    @field_validator("game_genre") 
    @classmethod 
    def valid_genre(cls, v: str) -> str: 
        allowed = {"platformer", "rpg", "puzzle", "fps", "strategy", "idle"} 
        if v not in allowed: 
            raise ValueError(f"Género {v} no soportado en free tier") 
        return v 
 
 
class DataScienceTaskInputs(BaseModel): 
    """Schema para el dominio 9_data_science.""" 
    data_source_type: str = Field(..., description="csv, api, database, synthetic") 
    has_sample_data: bool = Field(default=False) 
    objective: str = Field(..., description="classification, regression, clustering") 
    max_rows: int = Field(default=100000, ge=100, le=1000000) 
    model_type: Optional[str] = Field(default=None) 
 
    @field_validator("objective") 
    @classmethod 
    def valid_objective(cls, v: str) -> str: 
        allowed = {"classification", "regression", "clustering", "anomaly", "forecasting"} 
        if v not in allowed: 
            raise ValueError(f"Objetivo {v} no soportado") 
        return v 
 
    @field_validator("data_source_type") 
    @classmethod 
    def valid_source(cls, v: str) -> str: 
        allowed = {"csv", "api", "database", "synthetic", "json"} 
        if v not in allowed: 
            raise ValueError(f"Fuente {v} no soportada") 
        return v 
 
 
class ContentTaskInputs(BaseModel): 
    """Schema para el dominio 10_content_writing.""" 
    content_type: str = Field(..., description="blog, copywriting, documentation, script") 
    word_count: int = Field(default=500, ge=100, le=5000) 
    tone: str = Field(default="professional", description="professional, casual, technical") 
    seo_keywords: List[str] = Field(default_factory=list) 
    target_language: str = Field(default="es") 
 
    @field_validator("content_type") 
    @classmethod 
    def valid_content(cls, v: str) -> str: 
        allowed = {"blog", "copywriting", "documentation", "script", "email", "whitepaper"} 
        if v not in allowed: 
            raise ValueError(f"Tipo {v} no soportado") 
        return v 
 
 
class DevOpsTaskInputs(BaseModel): 
    """Schema para el dominio 11_devops_infrastructure.""" 
    infrastructure_type: str = Field(..., description="cloud, ci-cd, monitoring, k8s") 
    cloud_provider: Optional[str] = Field(default=None) 
    containerization: bool = Field(default=False) 
    iac_required: bool = Field(default=False) 
 
    @field_validator("infrastructure_type") 
    @classmethod 
    def valid_infra(cls, v: str) -> str: 
        allowed = {"cloud", "ci-cd", "monitoring", "k8s", "serverless"} 
        if v not in allowed: 
            raise ValueError(f"Infraestructura {v} no soportada") 
        return v 
 
 
class SecurityTaskInputs(BaseModel): 
    """Schema para el dominio 12_cybersecurity.""" 
    audit_type: str = Field(..., description="pentest, audit, compliance, training") 
    target_scope: str = Field(..., description="web, network, app, cloud") 
    compliance_framework: Optional[str] = Field(default=None) 
 
    @field_validator("audit_type") 
    @classmethod 
    def valid_audit(cls, v: str) -> str: 
        allowed = {"pentest", "audit", "compliance", "training", "incident-response"} 
        if v not in allowed: 
            raise ValueError(f"Auditoría {v} no soportada") 
        return v 
 
 
class LegalTaskInputs(BaseModel): 
    """Schema para el dominio 13_legal_compliance.""" 
    document_type: str = Field(..., description="contract, privacy-policy, terms, compliance") 
    jurisdiction: Optional[str] = Field(default=None) 
    industry: Optional[str] = Field(default=None) 
 
    @field_validator("document_type") 
    @classmethod 
    def valid_doc(cls, v: str) -> str: 
        allowed = {"contract", "privacy-policy", "terms", "compliance", "gdpr", "nda"} 
        if v not in allowed: 
            raise ValueError(f"Documento {v} no soportado") 
        return v 
 
 
# Registro global de schemas 
DOMAIN_SCHEMAS: Dict[str, Type[BaseModel]] = { 
    "1_web_development": WebTaskInputs, 
    "2_software_development": SoftwareTaskInputs, 
    "3_3d_design": Design3DTaskInputs, 
    "4_graphic_design": GraphicDesignTaskInputs, 
    "5_video_production": VideoTaskInputs, 
    "6_marketing_growth": MarketingTaskInputs, 
    "7_software_architecture": ArchitectureTaskInputs, 
    "8_game_development": GameDevTaskInputs, 
    "9_data_science": DataScienceTaskInputs, 
    "10_content_writing": ContentTaskInputs, 
    "11_devops_infrastructure": DevOpsTaskInputs, 
    "12_cybersecurity": SecurityTaskInputs, 
    "13_legal_compliance": LegalTaskInputs, 
} 
 
 
# 
# SECCIÓN 3: REGLAS DE CLASIFICACIÓN DETERMINISTA 
# 
 
@dataclass 
class ClassificationRule: 
    """Regla de clasificación basada en regex.""" 
    patterns: List[re.Pattern] 
    domain: str 
    archetype: str 
    confidence: float = 1.0 
    required_keywords: List[str] = field(default_factory=list) 
    excluded_keywords: List[str] = field(default_factory=list) 
    priority: int = 0  # Mayor número = mayor prioridad 
 
 
# Reglas de clasificación para TODOS los dominios 
CLASSIFICATION_RULES = [ 
    # === DOMINIO 1: WEB DEVELOPMENT === 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(landing page|página de (inicio|aterrizaje)|home page|one.page)\b", 
re.I), 
            re.compile(r"\b(página web simple|web simple|sitio web básico)\b", re.I), 
        ], 
        domain="1_web_development", 
        archetype="landing-page", 
        required_keywords=["hero", "cta", "services", "contact"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(ecommerce|tienda online|shop|online store|venta online)\b", re.I), 
            re.compile(r"\b(carrito|checkout|productos|catálogo|catalog)\b", re.I), 
        ], 
        domain="1_web_development", 
        archetype="ecommerce", 
        required_keywords=["product", "cart", "checkout", "catalog"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(dashboard|panel de control|admin panel|analytics|métricas)\b", re.I), 
            re.compile(r"\b(gráficos|charts|tablas de datos|kpi|indicadores)\b", re.I), 
        ], 
        domain="1_web_development", 
        archetype="saas-dashboard", 
        required_keywords=["chart", "table", "filter", "auth", "dashboard"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(blog|artículos|noticias|revista|magazine|cms)\b", re.I), 
        ], 
        domain="1_web_development", 
        archetype="cms-blog", 
        required_keywords=["post", "category", "tag", "author"], 
        priority=8, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(portfolio|galería|galeria|showcase|trabajos)\b", re.I), 
        ], 
        domain="1_web_development", 
        archetype="portfolio", 
        required_keywords=["gallery", "project", "image", "filter"], 
        priority=8, 
    ), 
 
    # === DOMINIO 2: SOFTWARE DEVELOPMENT === 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(api|endpoint|rest api|graphql|backend|servidor)\b", re.I), 
            re.compile(r"\b(crud|microservicio|microservice|webhook)\b", re.I), 
        ], 
        domain="2_software_development", 
        archetype="api-development", 
        required_keywords=["endpoint", "auth", "schema", "crud"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(app móvil|mobile app|aplicación móvil|ios|android|flutter|react native)\b", re.I), 
        ], 
        domain="2_software_development", 
        archetype="mobile-apps", 
        required_keywords=["screen", "navigation", "api", "storage"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(app de escritorio|desktop app|aplicación de escritorio|electron|tauri)\b", re.I),
        ], 
        domain="2_software_development", 
        archetype="desktop-apps", 
        required_keywords=["window", "menu", "file-system", "ipc"], 
        priority=8, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(cli|command line|terminal|script|automatización|automation)\b", 
re.I), 
        ], 
        domain="2_software_development", 
        archetype="cli-tools", 
        required_keywords=["command", "flag", "output", "config"], 
        priority=7, 
    ), 
 
    # === DOMINIO 3: 3D DESIGN === 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(3d|modelado 3d|three.js|blender|arquitectura 3d|3d model)\b", re.I), 
            re.compile(r"\b(render|renderizado|visualización 3d|3d visualization)\b", re.I), 
        ], 
        domain="3_3d_design", 
        archetype="3d-archviz", 
        required_keywords=["mesh", "material", "lighting", "camera"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(producto 3d|3d product|modelo de producto|product visualization)\b", re.I),
        ], 
        domain="3_3d_design", 
        archetype="3d-product", 
        required_keywords=["model", "texture", "rotation", "viewer"], 
        priority=9, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(animación 3d|3d animation|rigging|character animation)\b", re.I), 
        ], 
        domain="3_3d_design", 
        archetype="3d-animation", 
        required_keywords=["rig", "bone", "timeline", "keyframe"], 
        priority=8, 
    ), 
 
    # === DOMINIO 4: GRAPHIC DESIGN === 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(logo|identidad visual|brand identity|logotipo|marca)\b", re.I), 
        ], 
        domain="4_graphic_design", 
        archetype="branding", 
        required_keywords=["logo", "color", "typography", "guideline"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(ui|ux|interfaz|interface|diseño de interfaz|figma|sketch)\b", re.I), 
            re.compile(r"\b(diseño de app|app design|web design|diseño web)\b", re.I), 
        ], 
        domain="4_graphic_design", 
        archetype="ui-ux", 
        required_keywords=["component", "screen", "prototype", "wireframe"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(póster|poster|flyer|folleto|cartel|banner|social media)\b", re.I), 
        ], 
        domain="4_graphic_design", 
        archetype="marketing-materials", 
        required_keywords=["layout", "image", "text", "cta"], 
        priority=8, 
    ), 
 
    # === DOMINIO 5: VIDEO PRODUCTION === 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(video|vídeo|motion graphics|animación|after effects|premiere)\b", 
re.I), 
            re.compile(r"\b(edición de video|video editing|postproducción|post-production)\b", 
re.I), 
        ], 
        domain="5_video_production", 
        archetype="video-production", 
        required_keywords=["clip", "timeline", "transition", "export"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(motion graphics|motion design|animación 2d|2d animation|kinetic)\b", re.I),
        ], 
        domain="5_video_production", 
        archetype="motion-graphics", 
        required_keywords=["shape", "text-animation", "easing", "composition"], 
        priority=9, 
    ), 
 
    # === DOMINIO 6: MARKETING & GROWTH === 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(seo|posicionamiento|search engine|google ranking|keywords)\b", 
re.I), 
        ], 
        domain="6_marketing_growth", 
        archetype="seo-sem", 
        required_keywords=["keyword", "meta", "backlink", "content"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(redes sociales|social media|instagram|tiktok|linkedin|community manager)\b", re.I),
            re.compile(r"\b(publicaciones|posts|engagement|followers|seguidores)\b", re.I), 
        ], 
        domain="6_marketing_growth", 
        archetype="social-media", 
        required_keywords=["post", "hashtag", "engagement", "calendar"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(campaña|campaign|ads|publicidad|google ads|facebook ads|ppc)\b", 
re.I), 
        ], 
        domain="6_marketing_growth", 
        archetype="campaign-launch", 
        required_keywords=["audience", "budget", "creative", "conversion"], 
        priority=9, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(email marketing|newsletter|mailing|drip campaign)\b", re.I), 
        ], 
        domain="6_marketing_growth", 
        archetype="email-marketing", 
        required_keywords=["template", "segment", "open-rate", "automation"], 
        priority=8, 
    ), 
 
    # === DOMINIO 7: SOFTWARE ARCHITECTURE === 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(arquitectura de software|software architecture|system design|diseño de sistema)\b", re.I),
            re.compile(r"\b(diseño de arquitectura|architecture design|patrón de diseño)\b", 
re.I), 
        ], 
        domain="7_software_architecture", 
        archetype="system-design", 
        required_keywords=["component", "interface", "data-flow", "scalability"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            
re.compile(r"\b(microservicios|microservices|docker|kubernetes|k8s|orquestación)\b", re.I), 
        ], 
        domain="7_software_architecture", 
        archetype="microservices", 
        required_keywords=["service", "gateway", "discovery", "container"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            
re.compile(r"\b(seguridad|security|autenticación|authentication|autorización|authorization) b", re.I),
        ], 
        domain="7_software_architecture", 
        archetype="security", 
        required_keywords=["auth", "encryption", "token", "policy"], 
        priority=9, 
    ), 
 
    # === DOMINIO 8: GAME DEVELOPMENT === 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(juego|game|videojuego|videogame|unity|unreal|godot)\b", re.I), 
            re.compile(r"\b(plataformas|platformer|rpg|fps|puzzle|strategy)\b", re.I), 
        ], 
        domain="8_game_development", 
        archetype="game-development", 
        required_keywords=["level", "sprite", "physics", "input"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(assets de juego|game assets|sprites|texturas|models|sound effects)\b", re.I),
        ], 
        domain="8_game_development", 
        archetype="game-assets", 
        required_keywords=["sprite", "texture", "audio", "animation"], 
        priority=8, 
    ), 
 
    # === DOMINIO 9: DATA SCIENCE === 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(machine learning|ml|deep learning|ia|inteligencia artificial|ai model)\b", re.I),
            re.compile(r"\b(modelo predictivo|predictive model|entrenar modelo|train model)\b", re.I),
        ], 
        domain="9_data_science", 
        archetype="ml-model", 
        required_keywords=["feature", "train", "test", "accuracy"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(pipeline de datos|data pipeline|etl|extracción|transformación|carga)\b", re.I),
        ], 
        domain="9_data_science", 
        archetype="data-pipeline", 
        required_keywords=["extract", "transform", "load", "schedule"], 
        priority=9, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(análisis de datos|data analysis|analytics|dashboard|bi|business intelligence)\b", re.I),
        ], 
        domain="9_data_science", 
        archetype="analytics", 
        required_keywords=["metric", "visualization", "insight", "report"], 
        priority=9, 
    ), 
 
    # === DOMINIO 10: CONTENT WRITING === 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(blog|artículo|article|post|entrada|redacción)\b", re.I), 
        ], 
        domain="10_content_writing", 
        archetype="blog-article", 
        required_keywords=["heading", "paragraph", "seo", "cta"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(copywriting|copy|texto publicitario|sales copy|landing copy)\b", re.I), 
        ], 
        domain="10_content_writing", 
        archetype="copywriting", 
        required_keywords=["headline", "benefit", "objection", "cta"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(documentación|documentation|docs|readme|wiki|manual)\b", re.I), 
        ], 
        domain="10_content_writing", 
        archetype="documentation", 
        required_keywords=["section", "code-block", "example", "api-ref"], 
        priority=8, 
    ), 
 
    # === DOMINIO 11: DEVOPS === 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(devops|ci/cd|pipeline|github actions|gitlab ci|jenkins)\b", re.I), 
            re.compile(r"\b(automatización de despliegue|deploy automation|cd pipeline)\b", 
re.I), 
        ], 
        domain="11_devops_infrastructure", 
        archetype="ci-cd", 
        required_keywords=["build", "test", "deploy", "stage"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(cloud|aws|azure|gcp|infraestructura|infrastructure|terraform)\b", re.I), 
        ], 
        domain="11_devops_infrastructure", 
        archetype="cloud-infrastructure", 
        required_keywords=["resource", "network", "security-group", "scaling"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            
re.compile(r"\b(monitorización|monitoring|observability|logs|métricas|prometheus|grafana) b", re.I),
        ], 
        domain="11_devops_infrastructure", 
        archetype="monitoring", 
        required_keywords=["metric", "alert", "dashboard", "log"], 
        priority=9, 
    ), 
 
    # === DOMINIO 12: CYBERSECURITY === 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(pentest|penetration test|prueba de penetración|ethical hacking)\b", 
re.I), 
            re.compile(r"\b(vulnerabilidad|vulnerability|exploit|cve|bug bounty)\b", re.I), 
        ], 
        domain="12_cybersecurity", 
        archetype="pentesting", 
        required_keywords=["scan", "exploit", "report", "cvss"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(auditoría de seguridad|security audit|assessment|evaluación)\b", 
re.I), 
        ], 
        domain="12_cybersecurity", 
        archetype="security-audit", 
        required_keywords=["checklist", "control", "finding", "remediation"], 
        priority=9, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(compliance|normativa|regulación|gdpr|iso 27001|soc2)\b", re.I), 
        ], 
        domain="12_cybersecurity", 
        archetype="compliance", 
        required_keywords=["requirement", "evidence", "gap", "remediation"], 
        priority=8, 
    ), 
 
    # === DOMINIO 13: LEGAL COMPLIANCE === 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(contrato|contract|agreement|acuerdo|legal document)\b", re.I), 
        ], 
        domain="13_legal_compliance", 
        archetype="contracts", 
        required_keywords=["clause", "party", "term", "obligation"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(privacidad|privacy|política de privacidad|privacy policy|gdpr)\b", re.I), 
        ], 
        domain="13_legal_compliance", 
        archetype="privacy", 
        required_keywords=["data", "consent", "right", "processor"], 
        priority=10, 
    ), 
    ClassificationRule( 
        patterns=[ 
            re.compile(r"\b(términos|terms|condiciones|conditions|terms of service|tos)\b", re.I), 
        ], 
        domain="13_legal_compliance", 
        archetype="terms", 
        required_keywords=["user", "service", "liability", "termination"], 
        priority=9, 
    ), 
] 
 
 
def classify_without_llm(task_description: str) -> Optional[Tuple[str, str, float, List[str]]]: 
    """Clasifica la tarea usando regex y árbol de decisión. 
    Retorna (domain, archetype, confidence, matched_keywords) o None.""" 
    task_lower = task_description.lower() 
 
    # Ordenar por prioridad descendente 
    sorted_rules = sorted(CLASSIFICATION_RULES, key=lambda r: r.priority, reverse=True) 
 
    for rule in sorted_rules: 
        # Verificar que TODOS los patrones coincidan (AND lógico) 
        if all(p.search(task_description) for p in rule.patterns): 
            # Verificar que NO haya keywords excluidas 
            if any(excl in task_lower for excl in rule.excluded_keywords): 
                continue 
 
            # Verificar que haya suficientes keywords requeridas presentes 
            matched = [kw for kw in rule.required_keywords if kw in task_lower] 
            coverage = (len(matched) / len(rule.required_keywords)) if rule.required_keywords else 1.0 
 
            if coverage >= 0.3:  # Al menos 30% de keywords requeridas 
                adjusted_confidence = rule.confidence * coverage 
                return rule.domain, rule.archetype, adjusted_confidence, matched 
 
    return None 
 
 
# 
# SECCIÓN 4: PLANTILLAS DETERMINISTAS POR ARQUETIPO (Jinja2) 
# 
 
# Plantillas de prompt para LLM - estructura rígida, solo rellenar huecos 
ARCHETYPE_PROMPT_TEMPLATES = { 
    # === WEB DEVELOPMENT === 
    "landing-page": { 
        "structure": [ 
            {"step": 1, "action": "html_structure", "file": "index.html", "template": 
"semantic_html_hero"}, 
            {"step": 2, "action": "css_styling", "file": "styles.css", "template": "responsive_css"}, 
            {"step": 3, "action": "js_interactivity", "file": "script.js", "template": "vanilla_js", 
"conditional": "needs_interactivity"}, 
            {"step": 4, "action": "validation", "tool": "html_validator"}, 
        ], 
        "required_inputs": [], 
        "llm_prompt": """ 
Crea una landing page con esta ESTRUCTURA OBLIGATORIA. NO añadas secciones extra. 
 
SECCIONES REQUERIDAS: 
1. <header> con navegación (mínimo 3 links) 
2. <section class="hero"> con: 
   - H1: "{{ title }}" 
   - Subtítulo: "{{ subtitle }}" 
   - CTA button: "{{ cta_text }}" 
3. <section class="services"> con 3 cards de servicio 
4. <section class="contact"> con formulario (nombre, email, mensaje) 
5. <footer> con copyright y links sociales 
 
REGLAS DURAS: 
- HTML5 semántico (header, main, section, footer) 
- CSS responsive: mobile-first, breakpoint 768px 
- Si NO hay paleta de colores: usa #1a1a2e (fondo oscuro), #e94560 (acento), #f5f5f5 
(texto) 
- Si NO hay copy: usa Lorem ipsum de 20 palabras máximo 
- NO uses frameworks (Bootstrap, Tailwind). CSS puro. 
- NO añadas: blog, testimonios, pricing, equipo, FAQ 
 
OUTPUT: Solo código HTML completo en un solo archivo. Sin explicaciones. Sin markdown. 
""".strip(), 
        "extractors": { 
            "title": r"(?:para|de|mi|un|una)\s+([^,.]+?)(?:\s+(?:con|que|y)|$)", 
            "subtitle": r"(?:que|muestre|showing|displaying)\s+([^,.]+)", 
            "cta_text": r'(?:cta|botón|button|llamada|action)\s*(?::|de|con)?\s*[\'"]?([^\'".,]+)[\'"]?' , 
        }, 
    }, 
 
    "ecommerce": { 
        "structure": [ 
            {"step": 1, "action": "product_schema", "file": "products.json", "template": 
"json_schema"}, 
            {"step": 2, "action": "product_listing", "file": "index.html", "template": "product_grid"}, 
            {"step": 3, "action": "product_detail", "file": "product.html", "template": 
"detail_page"}, 
            {"step": 4, "action": "cart_logic", "file": "cart.js", "template": "session_cart"}, 
            {"step": 5, "action": "checkout", "file": "checkout.html", "template": "stripe_checkout", 
"conditional": "payment_required"}, 
        ], 
        "required_inputs": [], 
        "llm_prompt": """ 
Crea un ecommerce MÍNIMO VIABLE con esta estructura exacta: 
 
ARCHIVOS A GENERAR: 
1. index.html - Grid de productos (máximo 6 productos de ejemplo) 
2. product.html - Página de producto individual 
3. cart.js - Carrito con localStorage 
4. styles.css - CSS puro, responsive 
5. (Opcional) checkout.html - Si se menciona pago 
 
ESTRUCTURA HTML index.html: 
- Header con logo y icono de carrito (contador) 
- Grid de productos: imagen placeholder, nombre, precio, botón "Añadir" 
- Cada producto es un <article class="product-card"> 
 
ESTRUCTURA product.html: 
- Imagen grande placeholder 
- Nombre, descripción (50 palabras), precio 
- Selector de cantidad 
- Botón "Añadir al carrito" 
 
REGLAS DURAS: 
- Máximo 3 páginas HTML 
- Vanilla JS only. NO React, Vue, Angular 
- CSS inline o en <style> para reducir requests 
- Productos de ejemplo: usa nombres genéricos (Producto A, B, C) 
- Precios: números enteros, moneda € 
- Imágenes: usar placeholder.com o data URI SVG 
 
OUTPUT: Código de cada archivo separado por === FILENAME === 
""".strip(), 
        "extractors": {}, 
    }, 
 
    "saas-dashboard": { 
        "structure": [ 
            {"step": 1, "action": "auth_flow", "file": "login.html", "template": "jwt_auth", 
"conditional": "auth_required"}, 
            {"step": 2, "action": "dashboard_layout", "file": "dashboard.html", "template": 
"admin_layout"}, 
            {"step": 3, "action": "charts", "file": "charts.js", "template": "chartjs_integration"}, 
            {"step": 4, "action": "data_table", "file": "table.js", "template": "sortable_table"}, 
            {"step": 5, "action": "sidebar", "file": "sidebar.js", "template": "collapsible_nav"}, 
        ], 
        "required_inputs": [], 
        "llm_prompt": """ 
Crea un dashboard de administración con esta estructura: 
 
LAYOUT: 
- Sidebar izquierda colapsable (iconos + texto) 
- Header superior con search, notificaciones, perfil 
- Main content area con grid de widgets 
- Footer minimalista 
 
WIDGETS REQUERIDOS (mínimo 4): 
1. Stat cards: 4 métricas con icono, valor, cambio % 
2. Gráfico de líneas (últimos 30 días) 
3. Gráfico circular (distribución) 
4. Tabla de datos con paginación (10 filas) 
 
REGLAS DURAS: 
- CSS Grid para layout principal 
- Chart.js para gráficos (CDN) 
- Datos de ejemplo en arrays JS 
- Responsive: sidebar se convierte en bottom nav en móvil 
- Colores: usa sistema de CSS variables (--primary, --secondary, etc.) 
- NO backend real. Todo es mock data. 
 
OUTPUT: HTML completo + CSS + JS en un solo archivo. 
""".strip(), 
        "extractors": {}, 
    }, 
 
    # === SOFTWARE DEVELOPMENT === 
    "api-development": { 
        "structure": [ 
            {"step": 1, "action": "schema_definition", "file": "openapi.yaml", "template": 
"openapi_spec"}, 
            {"step": 2, "action": "server_setup", "file": "server.py", "template": "fastapi_express"}, 
            {"step": 3, "action": "endpoints", "file": "routes.py", "template": "crud_routes"}, 
            {"step": 4, "action": "tests", "file": "test_api.py", "template": "pytest_jest", 
"conditional": "needs_tests"}, 
        ], 
        "required_inputs": ["project_type"], 
        "llm_prompt": """ 
Crea una API REST con esta estructura exacta: 
 
TECNOLOGÍA: {{ language_preference | default('Python/FastAPI') }} 
ENDPOINTS REQUERIDOS (máximo {{ max_endpoints | default(5) }}): 
- GET /items - Listar con paginación (limit, offset) 
- GET /items/{id} - Obtener uno 
- POST /items - Crear (validar campos requeridos) 
- PUT /items/{id} - Actualizar completo 
- DELETE /items/{id} - Eliminar 
 
REGLAS DURAS: 
- Validación de entrada en TODOS los endpoints 
- Manejo de errores 400, 404, 500 con JSON consistente 
- {{ "Autenticación JWT" if needs_auth else "Sin auth (comentar dónde iría)" }} 
- {{ "Tests unitarios" if needs_tests else "Sin tests" }} 
- Datos en memoria (lista/dict), NO base de datos real 
- Documentación OpenAPI/Swagger automática 
 
OUTPUT: Código completo listo para ejecutar. 
""".strip(), 
        "extractors": { 
            "language_preference": r"(python|node|go|ruby|rust|java)", 
        }, 
    }, 
 
    "mobile-apps": { 
        "structure": [ 
            {"step": 1, "action": "project_setup", "file": "App.js", "template": "react_native"}, 
            {"step": 2, "action": "navigation", "file": "navigation.js", "template": "stack_nav"}, 
            {"step": 3, "action": "screens", "file": "screens/", "template": "screen_components"}, 
            {"step": 4, "action": "api_integration", "file": "api.js", "template": "axios_fetch"}, 
        ], 
        "required_inputs": [], 
        "llm_prompt": """ 
Crea una app móvil con React Native (o Flutter si se especifica) con esta estructura: 
 
PANTALLAS REQUERIDAS (mínimo 3): 
1. Home - Lista de items con scroll infinito 
2. Detail - Vista detalle con imagen, descripción, acciones 
3. Profile - Datos de usuario, settings, logout 
 
NAVEGACIÓN: 
- Bottom tabs: Home, Search, Profile 
- Stack dentro de Home: List → Detail 
 
REGLAS DURAS: 
- Expo SDK (sin eject) 
- Styled Components o StyleSheet 
- Datos mock en archivo separado 
- Sin backend real 
- Iconos: @expo/vector-icons 
- Máximo 5 pantallas 
 
OUTPUT: Código de App.js + componentes principales. 
""".strip(), 
        "extractors": {}, 
    }, 
 
    # === DATA SCIENCE === 
    "ml-model": { 
        "structure": [ 
            {"step": 1, "action": "data_loading", "file": "load_data.py", "template": 
"pandas_loader"}, 
            {"step": 2, "action": "preprocessing", "file": "preprocess.py", "template": 
"sklearn_pipeline"}, 
            {"step": 3, "action": "model_training", "file": "train.py", "template": "sklearn_model"}, 
            {"step": 4, "action": "evaluation", "file": "evaluate.py", "template": "metrics_report"}, 
            {"step": 5, "action": "inference", "file": "predict.py", "template": "inference_script"}, 
        ], 
        "required_inputs": ["objective", "data_source_type"], 
        "llm_prompt": """ 
Crea un pipeline de Machine Learning completo: 
 
OBJETIVO: {{ objective }} 
DATOS: {{ data_source_type }} 
 
ARCHIVOS REQUERIDOS: 
1. load_data.py - Carga desde {{ data_source_type }} 
2. preprocess.py - Limpieza, encoding, scaling 
3. train.py - Entrenamiento con validación cruzada 
4. evaluate.py - Métricas y reporte 
5. predict.py - Inferencia sobre nuevos datos 
 
REGLAS DURAS: 
- scikit-learn obligatorio 
- {{ "Clasificación: accuracy, precision, recall, F1" if objective == 'classification' else 
"Regresión: MSE, RMSE, R², MAE" if objective == 'regression' else "Clustering: silhouette, 
inertia, davies-bouldin" }} - Guardar modelo con joblib
- Manejo de valores nulos y outliers 
- Split train/test 80/20 
- Sin deep learning (scikit-learn only) 
 
OUTPUT: Código Python completo, listo para ejecutar. 
""".strip(), 
        "extractors": { 
            "objective": r"(classification|regression|clustering|anomaly|forecasting)", 
            "data_source_type": r"(csv|api|database|synthetic|json)", 
        }, 
    }, 
 
    "data-pipeline": { 
        "structure": [ 
            {"step": 1, "action": "extract", "file": "extract.py", "template": "data_extractor"}, 
            {"step": 2, "action": "transform", "file": "transform.py", "template": 
"pandas_transform"}, 
            {"step": 3, "action": "load", "file": "load.py", "template": "data_loader"}, 
            {"step": 4, "action": "orchestrate", "file": "pipeline.py", "template": "airflow_dag"}, 
        ], 
        "required_inputs": ["data_source_type"], 
        "llm_prompt": """ 
Crea un pipeline ETL completo: 
 
EXTRACCIÓN desde: {{ data_source_type }} 
 
ARCHIVOS: 
1. extract.py - Extraer datos (con manejo de errores y retries) 
2. transform.py - Limpiar, normalizar, enriquecer 
3. load.py - Cargar a destino (CSV/JSON/DB) 
4. pipeline.py - Orquestación con logging 
 
REGLAS DURAS: 
- pandas para transformación 
- logging en cada etapa 
- Manejo de errores: si falla extract, no continuar 
- Validación de schema antes de load 
- Sin Airflow (script Python puro) 
- Configuración en config.yaml 
 
OUTPUT: Código Python + config YAML. 
""".strip(), 
        "extractors": {}, 
    }, 
 
    # === CONTENT WRITING === 
    "blog-article": { 
        "structure": [ 
            {"step": 1, "action": "outline", "file": "outline.md", "template": "article_outline"}, 
            {"step": 2, "action": "draft", "file": "article.md", "template": "markdown_article"}, 
            {"step": 3, "action": "seo_meta", "file": "meta.json", "template": "seo_metadata"}, 
        ], 
        "required_inputs": [], 
        "llm_prompt": """ 
Escribe un artículo de blog con esta estructura exacta: 
 
FORMATO: Markdown 
LONGITUD: {{ word_count | default(800) }} palabras 
TONO: {{ tone | default('professional') }} 
 
ESTRUCTURA OBLIGATORIA: 
1. H1: Título atractivo (máximo 60 caracteres) 
2. Introducción (100 palabras): hook + problema + promesa 
3. H2: "¿Qué es [tema]?" (150 palabras) 
4. H2: "Beneficios de [tema]" (200 palabras, lista de 3-5 items) 
5. H2: "Cómo implementar [tema]" (200 palabras, pasos numerados) 
6. H2: "Conclusión" (100 palabras, resumen + CTA) 
 
REGLAS DURAS: 
- Párrafos de máximo 3-4 líneas 
- Subtítulos descriptivos (H2/H3) 
- {{ "Keywords SEO: " + ", ".join(seo_keywords) if seo_keywords else "Sin keywords específicas" }}
- Lenguaje {{ target_language | default('es') }} 
- NO usar: "En conclusión", "En resumen", "Como hemos visto" 
- CTA final: pregunta al lector o sugerencia de siguiente paso 
 
OUTPUT: Solo markdown, sin explicaciones. 
""".strip(), 
        "extractors": { 
            "word_count": r"(\d+)\s*(?:palabras|words)", 
            "tone": r"(professional|casual|technical|formal|friendly)", 
        }, 
    }, 
 
    "copywriting": { 
        "structure": [ 
            {"step": 1, "action": "research", "file": "brief.md", "template": "copy_brief"}, 
            {"step": 2, "action": "headlines", "file": "headlines.txt", "template": 
"headline_variants"}, 
            {"step": 3, "action": "body_copy", "file": "copy.md", "template": "sales_copy"}, 
            {"step": 4, "action": "cta", "file": "cta.txt", "template": "cta_variants"}, 
        ], 
        "required_inputs": [], 
        "llm_prompt": """ 
Escribe copy de ventas con esta estructura: 
 
FORMATO: Texto plano con secciones marcadas 
 
ESTRUCTURA: 
=== HEADLINE === 
- Variante 1: Directa (menciona beneficio principal) 
- Variante 2: Curiosidad (pregunta o stat sorprendente) 
- Variante 3: Urgencia (limitado en tiempo/cantidad) 
 
=== BODY === 
- Hook (2-3 líneas): problema agudo del target 
- Story/Proof: testimonio o dato de credibilidad 
- Benefits: 3 bullets con formato "Beneficio + Cómo" 
- Objection handling: 1 objeción común + refutación 
 
=== CTA === 
- Principal: acción clara + beneficio inmediato 
- Secundaria: opción de bajo compromiso 
 
REGLAS DURAS: 
- Máximo 300 palabras total 
- Segunda persona ("tú", "tu") 
- Verbos de acción al inicio de bullets 
- Sin jerga técnica innecesaria 
- 1 emoji máximo por sección 
 
OUTPUT: Texto formateado con las secciones marcadas. 
""".strip(), 
        "extractors": {}, 
    }, 
 
    # === DEVOPS === 
    "ci-cd": { 
        "structure": [ 
            {"step": 1, "action": "pipeline_config", "file": ".github/workflows/ci.yml", "template": 
"github_actions"}, 
            {"step": 2, "action": "dockerfile", "file": "Dockerfile", "template": "docker_python"}, 
            {"step": 3, "action": "deploy_script", "file": "deploy.sh", "template": "bash_deploy"}, 
        ], 
        "required_inputs": [], 
        "llm_prompt": """ 
Crea un pipeline CI/CD completo: 
 
ARCHIVOS: 
1. .github/workflows/ci.yml - GitHub Actions 
2. Dockerfile - Containerización 
3. deploy.sh - Script de despliegue 
 
PIPELINE REQUERIDA: 
- Trigger: push a main, pull_request 
- Jobs: lint → test → build → deploy 
- Test: pytest con coverage mínimo 80% 
- Build: Docker build + push a registry 
- Deploy: SSH a servidor (simulado) 
 
REGLAS DURAS: 
- Usar actions oficiales (actions/checkout, actions/setup-python) 
- Caché de dependencias 
- Secrets: usar variables de entorno 
- Dockerfile multi-stage (build → runtime) 
- deploy.sh con rollback automático si falla health check 
 
OUTPUT: Archivos listos para usar. 
""".strip(), 
        "extractors": {}, 
    }, 
 
    "cloud-infrastructure": { 
        "structure": [ 
            {"step": 1, "action": "terraform_main", "file": "main.tf", "template": "terraform_aws"}, 
            {"step": 2, "action": "variables", "file": "variables.tf", "template": "tf_variables"}, 
            {"step": 3, "action": "outputs", "file": "outputs.tf", "template": "tf_outputs"}, 
            {"step": 4, "action": "modules", "file": "modules/", "template": "tf_modules"}, 
        ], 
        "required_inputs": ["cloud_provider"], 
        "llm_prompt": """ 
Crea infraestructura cloud con Terraform: 
 
PROVEEDOR: {{ cloud_provider | default('AWS') }} 
 
RECURSOS REQUERIDOS: 
- VPC con 2 subnets (pública/privada) 
- Security groups (HTTP, HTTPS, SSH) 
- EC2/Compute instance (t3.micro o equivalente) 
- S3/Bucket para assets 
- RDS/Database (opcional, comentado) 
 
REGLAS DURAS: 
- Variables parametrizadas (region, instance_type) 
- Outputs: IP pública, DNS, endpoint DB 
- Tags en todos los recursos 
- Estado remoto en S3 (comentado) 
- Sin hardcoded values 
 
OUTPUT: Archivos .tf completos. 
""".strip(), 
        "extractors": { 
            "cloud_provider": r"(aws|azure|gcp|digitalocean|linode)", 
        }, 
    }, 
 
    # === CYBERSECURITY === 
    "pentesting": { 
        "structure": [ 
            {"step": 1, "action": "recon", "file": "recon.sh", "template": "nmap_recon"}, 
            {"step": 2, "action": "scan", "file": "scan.py", "template": "vulnerability_scan"}, 
            {"step": 3, "action": "exploit", "file": "exploit.md", "template": "exploit_guide"}, 
            {"step": 4, "action": "report", "file": "report.md", "template": "pentest_report"}, 
        ], 
        "required_inputs": ["target_scope"], 
        "llm_prompt": """ 
Crea un framework de pentesting ético: 
 
ALCANCE: {{ target_scope }} 
 
ARCHIVOS: 
1. recon.sh - Reconocimiento con nmap, whois, dns 
2. scan.py - Escaneo de vulnerabilidades automatizado 
3. exploit.md - Guía de explotación (solo teórica) 
4. report.md - Plantilla de informe ejecutivo 
 
REGLAS DURAS: 
- Solo técnicas de caja gris/blanca 
- Disclaimer legal en cada script 
- Sin exploits automáticos (solo detección) 
- CVSS scoring para hallazgos 
- Remediación incluida para cada vulnerabilidad 
- Formato: Markdown con tablas 
 
OUTPUT: Scripts + documentación. 
""".strip(), 
        "extractors": { 
            "target_scope": r"(web|network|app|cloud|api)", 
        }, 
    }, 
 
    # === LEGAL === 
    "privacy-policy": { 
        "structure": [ 
            {"step": 1, "action": "template", "file": "privacy-policy.md", "template": 
"gdpr_template"}, 
            {"step": 2, "action": "customize", "file": "privacy-policy-custom.md", "template": 
"custom_legal"}, 
        ], 
        "required_inputs": [], 
        "llm_prompt": """ 
Genera una Política de Privacidad con esta estructura: 
 
SECCIONES OBLIGATORIAS (GDPR/CCPA): 
1. Identidad del Responsable (placeholder) 
2. Datos recopilados y finalidad 
3. Base legal del tratamiento 
4. Destinatarios de los datos 
5. Transferencias internacionales 
6. Plazo de conservación 
7. Derechos del interesado (acceso, rectificación, supresión...) 
8. Derecho a reclamar ante AEPD/authority 
9. Decisiones automatizadas 
10. Cambios en la política 
 
REGLAS DURAS: 
- Lenguaje claro y comprensible (NO legalese excesivo) 
- Placeholders marcados con [EMPRESA], [EMAIL], [PAÍS] 
- Mencionar cookies si aplica 
- Incluir contacto DPO (Data Protection Officer) 
- Fecha de última actualización 
 
OUTPUT: Markdown completo, listo para personalizar. 
""".strip(), 
        "extractors": {}, 
    }, 
} 
 
 
# 
# SECCIÓN 5: EXTRACTOR DE VARIABLES 
# 
 
def extract_variables(task_description: str, archetype: str) -> Dict[str, Any]: 
    """Extrae variables del texto de la tarea para rellenar plantillas. Usa regex definidas en cada arquetipo."""

    template_data = ARCHETYPE_PROMPT_TEMPLATES.get(archetype) 
    if not template_data: 
        return {} 
 
    extractors = template_data.get("extractors", {}) 
    variables = {} 
 
    for var_name, pattern in extractors.items(): 
        match = re.search(pattern, task_description, re.I) 
        if match: 
            variables[var_name] = match.group(1).strip() 
 
    # Extracciones genéricas comunes 
    variables["title"] = _extract_title(task_description) 
    variables["subtitle"] = _extract_subtitle(task_description) 
    variables["cta_text"] = _extract_cta(task_description) 
 
    return variables 
 
 
def _extract_title(task: str) -> str: 
    """Extrae título probable de la tarea.""" 
    # Patrones comunes: "para mi negocio de X", "de X", "sobre X" 
    patterns = [ 
        r"(?:para|de|sobre|mi|un|una)\s+([^,.;]{3,60}?)(?:\s+(?:con|que|y|para)|$)", 
        r"(?:crear|hacer|desarrollar|diseñar)\s+(?:una?\s+)?([^,.;]{3,60}?)(?:\s+(?:con|que|y)|$)", 
    ] 
    for pat in patterns: 
        match = re.search(pat, task, re.I) 
        if match: 
            return match.group(1).strip().title() 
    return "Mi Proyecto" 
 
 
def _extract_subtitle(task: str) -> str: 
    """Extrae subtítulo o descripción.""" 
    match = re.search(r"(?:que|muestre|showing|displaying|con)\s+([^,.;]{10,100})", task, re.I) 
    if match: 
        return match.group(1).strip().capitalize() 
    return "Solución profesional a medida" 
 
 
def _extract_cta(task: str) -> str: 
    """Extrae CTA probable.""" 
    ctas = { 
        "contacto": "Contáctanos", 
        "presupuesto": "Solicitar Presupuesto", 
        "demo": "Ver Demo", 
        "prueba": "Probar Gratis", 
        "comprar": "Comprar Ahora", 
        "registro": "Registrarme", 
        "descargar": "Descargar", 
    } 
    task_lower = task.lower() 
    for keyword, cta in ctas.items(): 
        if keyword in task_lower: 
            return cta 
    return "Saber Más" 
 
 
# 
# SECCIÓN 6: JAULA PRINCIPAL 
# 
 
class DeterministicCage: 
    """Jaula de estados finitos que guía al LLM por un camino determinista. Maximiza éxito en free tiers y hardware limitado."""

 
    def __init__(self, llm_client, world_model_layer=None, calibration_store=None): 
        self.llm = llm_client 
        self.wm = world_model_layer 
        self.calibration = calibration_store 
        self.llm_calls_used = 0 
        self.rejection_memory: Dict[str, List[str]] = {} 
 
    async def run(self, task_description: str) -> CageResult: 
        """Ejecuta la jaula completa con todos los estados.""" 
        start_time = datetime.utcnow() 
 
        try: 
            # --- ESTADO 1: PARSE --- 
            parsed = self._parse_task(task_description) 
 
            # --- ESTADO 2: CLASSIFY (sin LLM) --- 
            classification = classify_without_llm(task_description) 
 
            if classification is None: 
                # Fallback: usar LLM para clasificación (último recurso) 
                classification = await self._classify_with_llm(task_description) 
 
            domain, archetype, conf, matched_keywords = classification 
 
            # --- ESTADO 3: VALIDATE (schema Pydantic) --- 
            schema_class = DOMAIN_SCHEMAS.get(domain) 
            inputs = {} 
            if schema_class: 
                try: 
                    inputs = self._extract_inputs(task_description, schema_class, archetype) 
                except (ValidationError, ValueError) as e: 
                    return CageResult( 
                        state=CageState.VALIDATE, 
                        success=False, 
                        reclarification_needed=True, 
                        reclarification_questions=[f"Input inválido: {e}"], 
                        llm_calls_used=self.llm_calls_used, 
                    ) 
 
            # --- ESTADO 4: TEMPLATE (plan determinista) --- 
            template_data = ARCHETYPE_PROMPT_TEMPLATES.get(archetype) 
            if not template_data: 
                return CageResult( 
                    state=CageState.TEMPLATE, 
                    success=False, 
                    error=f"Arquetipo '{archetype}' no tiene template determinista", 
                    llm_calls_used=self.llm_calls_used, 
                ) 
 
            # Verificar inputs requeridos 
            missing = [ 
                inp for inp in template_data.get("required_inputs", []) 
                if not inputs.get(inp, False) 
            ] 
            if missing: 
                return CageResult( 
                    state=CageState.TEMPLATE, 
                    success=False, 
                    reclarification_needed=True, 
                    reclarification_questions=[ 
                        f"Falta información requerida para '{archetype}': {m}. " 
                        f"Por favor especifica este dato para continuar." 
                        for m in missing 
                    ], 
                    llm_calls_used=self.llm_calls_used, 
                ) 
 
            # Extraer variables para el prompt 
            variables = extract_variables(task_description, archetype) 
            variables.update(inputs) 
 
            # Renderizar prompt 
            prompt_template = template_data["llm_prompt"] 
            prompt = self._render_prompt(prompt_template, variables) 
 
            # Añadir adaptación de memoria de rechazo 
            adaptation = self._get_rejection_adaptation(archetype) 
            if adaptation: 
                prompt = adaptation + "\n\n" + prompt 
 
            # --- ESTADO 5: WORLD MODEL (opcional, con LLM) --- 
            if self.wm: 
                wm_result = await self._check_world_model(prompt, domain, archetype, 
task_description) 
                if wm_result.get("action") == "RECLARIFY": 
                    return CageResult( 
                        state=CageState.WORLD_MODEL, 
                        success=False, 
                        reclarification_needed=True, 
                        reclarification_questions=wm_result.get("gaps", ["Plan incierto"]), 
                        llm_calls_used=self.llm_calls_used, 
                    ) 
 
            # --- ESTADO 6: EXECUTE (LLM 70B, única llamada costosa) --- 
            code_output = await self._call_llm(prompt, temperature=0.2, max_tokens=4000) 
 
            # --- ESTADO 7: VERIFY (sin LLM) --- 
            verification = self._verify_output(code_output, template_data.get("structure", []), 
archetype) 
 
            if not verification["passed"]: 
                # Reintento con prompt corregido (máximo 1 reintento) 
                if self.llm_calls_used < 5:  # Límite de seguridad 
                    corrected_prompt = ( 
                        prompt +  
                        f"\n\n[ERROR PREVIO - CORREGIR]: {verification['errors']}\n" 
                        f"Corrige SOLO estos errores específicos. Mantén el resto igual." 
                    ) 
                    code_output = await self._call_llm(corrected_prompt, temperature=0.1, 
max_tokens=4000) 
                    verification = self._verify_output(code_output, template_data.get("structure", []), 
archetype) 
 
                if not verification["passed"]: 
                    self._record_rejection(archetype, verification["errors"]) 
                    return CageResult( 
                        state=CageState.VERIFY, 
                        success=False, 
                        error=f"Verificación fallida: {verification['errors']}", 
                        llm_calls_used=self.llm_calls_used, 
                    ) 
 
            # --- ESTADO 8: CALIBRATE --- 
            if self.calibration: 
                # La calibración real ocurre post-ejecución, aquí solo preparamos 
                pass 
 
            elapsed = int((datetime.utcnow() - start_time).total_seconds() * 1000) 
 
            return CageResult( 
                state=CageState.DONE, 
                success=True, 
                output={ 
                    "code": code_output, 
                    "domain": domain, 
                    "archetype": archetype, 
                    "structure": template_data.get("structure", []), 
                    "classification_confidence": conf, 
                    "matched_keywords": matched_keywords, 
                    "inputs_used": variables, 
                }, 
                llm_calls_used=self.llm_calls_used, 
                execution_time_ms=elapsed, 
            ) 
 
        except Exception as e: 
            elapsed = int((datetime.utcnow() - start_time).total_seconds() * 1000) 
            return CageResult( 
                state=CageState.FAILED, 
                success=False, 
                error=f"Error crítico en jaula: {str(e)}", 
                llm_calls_used=self.llm_calls_used, 
                execution_time_ms=elapsed, 
            ) 
 
    # --- Métodos internos --- 
 
    def _parse_task(self, task: str) -> Dict[str, Any]: 
        """Extracción de entidades mediante regex (sin LLM).""" 
        return { 
            "emails": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', task), 
            "urls": re.findall(r'https?://\S+', task), 
            "numbers": [int(n) for n in re.findall(r'\b\d+\b', task)], 
            "has_budget": bool(re.search(r'\b(presupuesto|budget|precio|coste|pricing)\b', task, 
re.I)), 
            "has_contact": bool(re.search(r'\b(contacto|contact|formulario|form|email)\b', task, 
re.I)), 
            "has_payment": bool(re.search(r'\b(pago|payment|stripe|paypal|tarjeta|card)\b', task, 
re.I)), 
            "has_auth": 
bool(re.search(r'\b(login|auth|usuario|user|password|contraseña|oauth)\b', task, re.I)), 
            "word_count_hint": self._extract_word_count(task), 
        } 
 
    def _extract_word_count(self, task: str) -> Optional[int]: 
        """Extrae indicación de longitud de texto.""" 
        match = re.search(r'(\d+)\s*(?:palabras|words|palabras|caracteres)', task, re.I) 
        return int(match.group(1)) if match else None 
 
    async def _classify_with_llm(self, task: str) -> Tuple[str, str, float, List[str]]: 
        """Último recurso: clasificar con LLM. Temperatura 0 para determinismo.""" 
        domains_list = "\n".join([f"- {k}: {v.__name__}" for k, v in DOMAIN_SCHEMAS.items()]) 
        prompt = f"""Clasifica esta tarea en UN dominio y UN arquetipo. 
DOMINIOS DISPONIBLES: 
{domains_list} 
 
Responde EXACTAMENTE en este formato (sin markdown, sin explicaciones): 
DOMINIO: <dominio> 
ARQUETIPO: <arquetipo> 
 
Tarea: {task}""" 
 
        response = await self._call_llm(prompt, temperature=0.0, max_tokens=100) 
 
        # Parseo robusto 
        domain_match = re.search(r'DOMINIO:\s*(\S+)', response, re.I) 
        archetype_match = re.search(r'ARQUETIPO:\s*(\S+)', response, re.I) 
 
        domain = domain_match.group(1) if domain_match else "1_web_development" 
        archetype = archetype_match.group(1) if archetype_match else "landing-page" 
 
        return domain, archetype, 0.6, ["llm_fallback"] 
 
    def _extract_inputs(self, task: str, schema_class: Type[BaseModel], archetype: str) -> Dict[str, Any]:
        """Extrae inputs del texto para validar contra schema.""" 
        task_lower = task.lower() 
 
        # Mapeo de extracciones según el schema 
        data = {} 
 
        if schema_class == WebTaskInputs: 
            data = { 
                "has_brand_assets": any(k in task_lower for k in ["logo", "marca", "brand", 
"favicon", "icono"]), 
                "has_copy_text": any(k in task_lower for k in ["texto", "copy", "contenido", 
"content", "escribir"]), 
                "has_color_palette": any(k in task_lower for k in ["color", "paleta", "palette", 
"tema", "theme"]), 
                "num_pages": self._extract_num_pages(task), 
                "needs_interactivity": any(k in task_lower for k in ["interactivo", "interactive", 
"calculadora", "calculator", "formulario dinámico"]), 
                "needs_backend": any(k in task_lower for k in ["backend", "api", "base de datos", 
"database", "servidor", "server"]), 
                "target_devices": self._extract_devices(task), 
                "seo_required": any(k in task_lower for k in ["seo", "posicionamiento", "google", 
"ranking"]), 
                "auth_required": any(k in task_lower for k in ["login", "auth", "usuario", "registro", 
"oauth"]), 
                "payment_required": any(k in task_lower for k in ["pago", "payment", "stripe", 
"checkout", "comprar"]), 
            } 
        elif schema_class == DataScienceTaskInputs: 
            data = { 
                "data_source_type": self._extract_data_source(task), 
                "has_sample_data": "muestra" in task_lower or "sample" in task_lower or 
"ejemplo" in task_lower, 
                "objective": self._extract_ml_objective(task), 
                "max_rows": 100000, 
                "model_type": self._extract_model_type(task), 
            } 
        elif schema_class == ContentTaskInputs: 
            data = { 
                "content_type": self._extract_content_type(task), 
                "word_count": self._extract_word_count(task) or 500, 
                "tone": self._extract_tone(task), 
                "seo_keywords": self._extract_seo_keywords(task), 
                "target_language": "es" if any(k in task_lower for k in ["español", "castellano"]) else 
"en", 
            } 
        elif schema_class == DevOpsTaskInputs: 
            data = { 
                "infrastructure_type": self._extract_infra_type(task), 
                "cloud_provider": self._extract_cloud_provider(task), 
                "containerization": any(k in task_lower for k in ["docker", "container", 
"contenedor"]), 
                "iac_required": any(k in task_lower for k in ["terraform", "cloudformation", 
"pulumi", "ansible"]), 
            } 
        elif schema_class == SecurityTaskInputs: 
            data = { 
                "audit_type": self._extract_audit_type(task), 
                "target_scope": self._extract_security_scope(task), 
                "compliance_framework": self._extract_compliance(task), 
            } 
        else: 
            # Fallback genérico 
            data = {"detected_from_task": task_lower[:100]} 
 
        return schema_class(**data).model_dump() 
 
    def _extract_num_pages(self, task: str) -> int: 
        match = re.search(r'(\d+)\s*(?:páginas|pages|pantallas|screens)', task, re.I) 
        return min(int(match.group(1)), 5) if match else 1 
 
    def _extract_devices(self, task: str) -> List[str]: 
        devices = [] 
        task_lower = task.lower() 
        if any(k in task_lower for k in ["móvil", "mobile", "responsive", "phone"]): 
            devices.append("mobile") 
        if any(k in task_lower for k in ["tablet", "ipad"]): 
            devices.append("tablet") 
        if not devices or any(k in task_lower for k in ["desktop", "pc", "ordenador", "web"]): 
            devices.append("desktop") 
        return devices 
 
    def _extract_data_source(self, task: str) -> str: 
        if re.search(r'\b(csv|excel|spreadsheet)\b', task, re.I): 
            return "csv" 
        if re.search(r'\b(api|endpoint|rest)\b', task, re.I): 
            return "api" 
        if re.search(r'\b(database|db|sql|postgres|mysql)\b', task, re.I): 
            return "database" 
        return "csv" 
 
    def _extract_ml_objective(self, task: str) -> str: 
        if re.search(r'\b(clasificar|classification|clasificación|categorizar)\b', task, re.I): 
            return "classification" 
        if re.search(r'\b(regresión|regression|predecir|predict)\b', task, re.I): 
            return "regression" 
        if re.search(r'\b(cluster|agrupar|clustering|segmentar)\b', task, re.I): 
            return "clustering" 
        return "classification" 
 
    def _extract_model_type(self, task: str) -> Optional[str]: 
        if re.search(r'\b(random forest|rf|árbol|tree)\b', task, re.I): 
            return "random_forest" 
        if re.search(r'\b(svm|vector|machine)\b', task, re.I): 
            return "svm" 
        if re.search(r'\b(red neuronal|neural|deep|nn)\b', task, re.I): 
            return "neural_network" 
        return None 
 
    def _extract_content_type(self, task: str) -> str: 
        if re.search(r'\b(blog|artículo|article|post|entrada)\b', task, re.I): 
            return "blog" 
        if re.search(r'\b(copy|copywriting|ventas|sales|landing)\b', task, re.I): 
            return "copywriting" 
        if re.search(r'\b(doc|documentación|documentation|wiki|manual)\b', task, re.I): 
            return "documentation" 
        return "blog" 
 
    def _extract_tone(self, task: str) -> str: 
        if re.search(r'\b(formal|técnico|technical|académico)\b', task, re.I): 
            return "technical" 
        if re.search(r'\b(casual|informal|amigable|friendly|divertido)\b', task, re.I): 
            return "casual" 
        return "professional" 
 
    def _extract_seo_keywords(self, task: str) -> List[str]: 
        match = re.search(r'(?:seo|keywords|palabras clave):?\s*([^,.;]+)', task, re.I) 
        if match: 
            return [k.strip() for k in match.group(1).split(",") if k.strip()] 
        return [] 
 
    def _extract_infra_type(self, task: str) -> str: 
        if re.search(r'\b(ci/cd|pipeline|github actions|gitlab|jenkins)\b', task, re.I): 
            return "ci-cd" 
        if re.search(r'\b(cloud|aws|azure|gcp|infraestructura)\b', task, re.I): 
            return "cloud" 
        if re.search(r'\b(monitor|observability|prometheus|grafana|logs)\b', task, re.I): 
            return "monitoring" 
        if re.search(r'\b(kubernetes|k8s|docker swarm|orquestación)\b', task, re.I): 
            return "k8s" 
        return "cloud" 
 
    def _extract_cloud_provider(self, task: str) -> Optional[str]: 
        if re.search(r'\baws\b', task, re.I): 
            return "aws" 
        if re.search(r'\bazure\b', task, re.I): 
            return "azure" 
        if re.search(r'\b(gcp|google cloud)\b', task, re.I): 
            return "gcp" 
        return None 
 
    def _extract_audit_type(self, task: str) -> str: 
        if re.search(r'\b(pentest|penetration|ethical hacking|exploit)\b', task, re.I): 
            return "pentest" 
        if re.search(r'\b(audit|auditoría|assessment|evaluación)\b', task, re.I): 
            return "audit" 
        if re.search(r'\b(compliance|normativa|gdpr|iso|soc2)\b', task, re.I): 
            return "compliance" 
        return "audit" 
 
    def _extract_security_scope(self, task: str) -> str: 
        if re.search(r'\b(web|api|aplicación|app)\b', task, re.I): 
            return "web" 
        if re.search(r'\b(network|red|infraestructura|infra)\b', task, re.I): 
            return "network" 
        if re.search(r'\b(cloud|aws|azure|gcp)\b', task, re.I): 
            return "cloud" 
        return "web" 
 
    def _extract_compliance(self, task: str) -> Optional[str]: 
        if re.search(r'\bgdpr\b', task, re.I): 
            return "GDPR" 
        if re.search(r'\biso\s*27001\b', task, re.I): 
            return "ISO27001" 
        if re.search(r'\bsoc2\b', task, re.I): 
            return "SOC2" 
        return None 
 
    def _render_prompt(self, template_str: str, variables: Dict[str, Any]) -> str: 
        """Renderiza template Jinja2 con variables extraídas.""" 
        env = Environment(loader=BaseLoader()) 
        template = env.from_string(template_str) 
        return template.render(**variables) 
 
    async def _check_world_model(self, prompt: str, domain: str, archetype: str, task: str) -> Dict[str, Any]:
        """Usa world model layer para validar antes de ejecutar."""
        if not self.wm:
            return {"action": "EXECUTE"} 
 
        # Importar aquí para evitar dependencia circular 
        try: 
            from world_model_layer import TaskManifest 
            manifest = TaskManifest( 
                task_description=task, 
                domain=domain, 
                archetype=archetype, 
                domain_confidence=0.9, 
            ) 
            wm, decision = await self.wm.plan(manifest) 
            return decision 
        except Exception: 
            return {"action": "EXECUTE"} 
 
    async def _call_llm(self, prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        """Wrapper de llamada al LLM con conteo y hardening.""" 
        self.llm_calls_used += 1 
 
        # Hard limit para free tier 
        if self.llm_calls_used > 5: 
            raise RuntimeError("Límite de llamadas LLM excedido (5). Abortando para preservar quota.")
 
        # Truncar prompt si es excesivamente largo (modelos locales limitados) 
        max_prompt_chars = 6000 
        if len(prompt) > max_prompt_chars: 
            prompt = prompt[:max_prompt_chars] + "\n[...truncado por límite de contexto]" 
 
        return await self.llm.generate(prompt, temperature=temperature, 
max_tokens=max_tokens) 
 
    def _verify_output(self, code: str, expected_structure: List[Dict], archetype: str) -> Dict[str, 
Any]: 
        """Verificación estructural sin LLM.""" 
        errors = [] 
        warnings = [] 
 
        # Verificaciones universales 
        if not code or len(code) < 50: 
            errors.append("Output vacío o demasiado corto") 
 
        # Verificaciones por tipo de arquetipo 
        if archetype in ["landing-page", "ecommerce", "saas-dashboard"]: 
            if "<html" not in code.lower(): 
                errors.append("Falta etiqueta <html>") 
            if "<!DOCTYPE" not in code.upper(): 
                errors.append("Falta DOCTYPE") 
            if "<head>" not in code.lower(): 
                errors.append("Falta <head>") 
            if "<body>" not in code.lower(): 
                errors.append("Falta <body>") 
 
            # Verificar estructura esperada 
            for step in expected_structure: 
                if step["action"] == "html_structure": 
                    if "<header" not in code.lower(): 
                        errors.append("Falta <header>") 
                    if "<footer" not in code.lower(): 
                        warnings.append("Falta <footer>") 
                elif step["action"] == "css_styling": 
                    if "{" not in code or "}" not in code: 
                        errors.append("Falta CSS válido (sin llaves)") 
                    if "@media" not in code.lower(): 
                        warnings.append("Falta media query responsive") 
 
        elif archetype in ["api-development", "cli-tools"]: 
            if "def " not in code and "function " not in code and "=>" not in code: 
                errors.append("No se detectan funciones/métodos") 
            if "return" not in code.lower(): 
                warnings.append("No se detectan sentencias return") 
 
        elif archetype in ["ml-model", "data-pipeline"]: 
            if "import" not in code.lower(): 
                errors.append("Faltan imports") 
            if "def " not in code: 
                errors.append("No se detectan funciones") 
 
        elif archetype in ["blog-article", "copywriting", "documentation"]: 
            if len(code.split()) < 50: 
                errors.append("Texto demasiado corto (< 50 palabras)") 
 
        # Patrones prohibidos (seguridad) 
        forbidden_patterns = [ 
            ("rm -rf /", "Comando destructivo detectado"), 
            ("DROP TABLE", "SQL destructivo detectado"), 
            ("DELETE FROM", "SQL DELETE sin WHERE"), 
            ("format c:", "Comando de formateo"), 
            ("os.system", "Llamada a sistema potencialmente peligrosa"), 
            ("subprocess.call", "Subproceso no controlado"), 
            ("eval(", "Uso de eval()"), 
            ("exec(", "Uso de exec()"), 
        ] 
 
        for pattern, msg in forbidden_patterns: 
            if pattern in code: 
                errors.append(f"[SEGURIDAD] {msg}: {pattern}") 
 
        # Verificar que no hay alucinaciones de estructura 
        if "<!-- Explicación -->" in code or "<!-- Explanation -->" in code: 
            warnings.append("Posible alucinación: comentarios de explicación en output") 
 
        return { 
            "passed": len(errors) == 0, 
            "errors": "; ".join(errors) if errors else "OK", 
            "warnings": warnings, 
        } 
 
    def _get_rejection_adaptation(self, archetype: str) -> str: 
        """Genera instrucción adicional basada en fallos previos.""" 
        failures = self.rejection_memory.get(archetype, []) 
        if not failures: 
            return "" 
 
        # Si falla mucho por formato, añadir recordatorio 
        json_failures = sum(1 for f in failures if "json" in f.lower()) 
        if json_failures > 2: 
            return "CRÍTICO: Tu output anterior falló al parsear. Genera SOLO código válido, sin markdown, sin explicaciones."
 
        # Si falla por overconfidence 
        conf_failures = sum(1 for f in failures if "confidence" in f.lower()) 
        if conf_failures > 2: 
            return "CRÍTICO: Sé conservador. Si hay ambigüedad, genera código mínimo viable, no features extra." 
 
        return "" 
 
    def _record_rejection(self, archetype: str, error: str): 
        """Registra fallo para adaptación futura.""" 
        self.rejection_memory.setdefault(archetype, []).append(error) 
        # Mantener máximo 10 fallos por arquetipo 
        self.rejection_memory[archetype] = self.rejection_memory[archetype][-10:] 
 
 
# 
# SECCIÓN 7: INTEGRACIÓN CON WORLD MODEL LAYER 
# 
 
async def run_with_world_model( 
    task_description: str, 
    llm_client, 
    world_model_layer=None, 
    calibration_store=None, 
) -> CageResult: 
    """Función de conveniencia para ejecutar la jaula con world model.""" 
    cage = DeterministicCage( 
        llm_client=llm_client, 
        world_model_layer=world_model_layer, 
        calibration_store=calibration_store, 
    ) 
    return await cage.run(task_description) 
 
 
# 
# SECCIÓN 8: EJEMPLOS DE USO 
# 
 
async def demo(): 
    """Demo de uso con cliente LLM dummy.""" 
 
    class DummyLLM: 
        async def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
            """Simula respuesta del modelo.""" 
            return """<!DOCTYPE html> <html lang="es">
<head> 
    <meta charset="UTF-8"> 
    <meta name="viewport" content="width=device-width, initial-scale=1.0"> 
    <title>Mi Landing Page</title> 
    <style> 
        * { margin: 0; padding: 0; box-sizing: border-box; } 
        body { font-family: Arial, sans-serif; background: #1a1a2e; color: #f5f5f5; } 
        header { padding: 1rem; display: flex; justify-content: space-between; } 
        .hero { text-align: center; padding: 4rem 2rem; } 
        .hero h1 { font-size: 3rem; margin-bottom: 1rem; } 
        .hero button { background: #e94560; color: white; padding: 1rem 2rem; border: none; 
cursor: pointer; } 
        .services { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; padding: 
2rem; } 
        .card { background: #16213e; padding: 2rem; border-radius: 8px; } 
        footer { text-align: center; padding: 2rem; } 
        @media (max-width: 768px) { 
            .services { grid-template-columns: 1fr; } 
            .hero h1 { font-size: 2rem; } 
        } 
    </style> 
</head> 
<body> 
    <header> 
        <div>Logo</div> 
        <nav><a href="#">Inicio</a> <a href="#servicios">Servicios</a> <a 
href="#contacto">Contacto</a></nav> 
    </header> 
    <main> 
        <section class="hero"> 
            <h1>Mi Negocio de Reformas</h1> 
            <p>Solución profesional a medida</p> 
            <button>Saber Más</button> 
        </section> 
        <section class="services" id="servicios"> 
            <div class="card"><h3>Reformas Integrales</h3><p>Descripción del 
servicio</p></div> 
            <div class="card"><h3>Interiorismo</h3><p>Descripción del servicio</p></div> 
            <div class="card"><h3>Rehabilitación</h3><p>Descripción del 
servicio</p></div> 
        </section> 
        <section id="contacto"> 
            <h2>Contacto</h2> 
            <form><input type="text" placeholder="Nombre"><input type="email" 
placeholder="Email"><textarea placeholder="Mensaje"></textarea><button 
type="submit">Enviar</button></form> 
        </section> 
    </main> 
    <footer><p>© 2024 Mi Negocio</p></footer> 
</body> 
</html>""" 
 
    llm = DummyLLM() 
    result = await run_with_world_model( 
        task_description="Crea una landing page para mi negocio de reformas con contacto", 
        llm_client=llm, 
    ) 
 
    print(f"Estado: {result.state.name}") 
    print(f"Éxito: {result.success}") 
    print(f"LLM calls: {result.llm_calls_used}") 
    print(f"Tiempo: {result.execution_time_ms}ms") 
    if result.output: 
        print(f"Dominio: {result.output['domain']}") 
        print(f"Arquetipo: {result.output['archetype']}") 
        print(f"Confianza clasificación: {result.output['classification_confidence']:.2f}") 
    if result.error: 
        print(f"Error: {result.error}") 
    if result.reclarification_needed: 
        print(f"Reclarificación: {result.reclarification_questions}") 
 
 
if __name__ == "__main__": 
    asyncio.run(demo()) 