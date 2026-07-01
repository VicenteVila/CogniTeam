"""Tests for the Scoping Agent module."""
import json
from unittest.mock import patch

import pytest

from cogniteam.scoping.loader import (
    BlueprintLoader,
    get_blueprint_for_task,
    get_blueprints_for_task_multi,
    classify_task,
    classify_task_multi,
    merge_blueprints,
)
from cogniteam.scoping.manifest import TaskManifest, ClassificationInfo
from cogniteam.scoping.agent import (
    _param_name_to_human,
    _collect_known_params,
    _generate_questions,
    _generate_clarified_task,
    _build_manifest,
    clarify_task,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def lading_bp():
    return {
        "domain_key": "1_web_development",
        "domain_name": "Web Development",
        "archetype_key": "landing-page",
        "archetype_name": "Landing Page",
        "priority": "conversion_and_speed",
        "domain_rules": ["Mobile-First obligatorio.", "Lighthouse > 95"],
        "stack": {"framework": "Next.js", "language": "TypeScript"},
        "required_parameters": [
            {"name": "project_name", "question": "¿Nombre del proyecto?", "required": True, "type": "string"},
            {"name": "business_type", "question": "¿Tipo de negocio?", "required": True, "type": "string"},
        ],
        "optional_parameters": [
            {"name": "custom_domain", "question": "¿Dominio personalizado?", "required": False, "type": "string"},
        ],
        "_classification": {
            "domain_key": "1_web_development",
            "archetype_key": "landing-page",
            "confidence": 0.92,
            "reasoning": "Landing page",
            "is_primary": True,
        },
    }


@pytest.fixture
def mobile_bp():
    return {
        "domain_key": "2_software_development",
        "domain_name": "Software Development",
        "archetype_key": "mobile-apps",
        "archetype_name": "Mobile Apps",
        "priority": "cross_platform_consistency",
        "domain_rules": ["Inmutabilidad de datos.", "Manejo estricto de excepciones."],
        "stack": {"framework": "Flutter", "local_db": "Hive"},
        "required_parameters": [
            {"name": "project_name", "question": "¿Nombre del proyecto?", "required": True, "type": "string"},
            {"name": "target_platforms", "question": "¿Plataformas?", "required": True, "type": "string"},
            {"name": "offline_features", "question": "¿Funcionalidades offline?", "required": True, "type": "string"},
        ],
        "optional_parameters": [
            {"name": "push_notifications", "question": "¿Notificaciones push?", "required": False, "type": "string"},
        ],
        "_classification": {
            "domain_key": "2_software_development",
            "archetype_key": "mobile-apps",
            "confidence": 0.78,
            "reasoning": "App móvil",
            "is_primary": False,
        },
    }


@pytest.fixture
def multi_blueprints(lading_bp, mobile_bp):
    return [lading_bp, mobile_bp]


# ── Tests: manifest.py ────────────────────────────────────────────────

class TestTaskManifest:
    def test_minimal_manifest(self):
        m = TaskManifest()
        assert m.status == "READY_TO_EXECUTE"
        assert m.manifest_version == "1.0"
        assert m.classification.domain_key == ""

    def test_secondary_classifications(self, multi_blueprints):
        primary = ClassificationInfo(**multi_blueprints[0]["_classification"], domain_name="Web", archetype_name="Landing Page", domain_rules=[], stack={})
        secondary = ClassificationInfo(**multi_blueprints[1]["_classification"], domain_name="Software", archetype_name="Mobile Apps", domain_rules=[], stack={})
        m = TaskManifest(
            classification=primary,
            secondary_classifications=[secondary],
            parameters={"project_name": "Test"},
            clarified_task="Tarea clarificada",
            original_task="Hazme algo",
        )
        assert m.classification.is_primary is True
        assert len(m.secondary_classifications) == 1
        assert m.secondary_classifications[0].archetype_key == "mobile-apps"
        assert m.secondary_classifications[0].is_primary is False

    def test_all_classifications(self, multi_blueprints):
        primary = ClassificationInfo(domain_key="web", domain_rules=[], stack={})
        secondary = ClassificationInfo(domain_key="mobile", domain_rules=[], stack={})
        m = TaskManifest(classification=primary, secondary_classifications=[secondary])
        all_c = m.all_classifications()
        assert len(all_c) == 2
        assert all_c[0].domain_key == "web"
        assert all_c[1].domain_key == "mobile"

    def test_to_dict(self):
        m = TaskManifest(original_task="test")
        d = m.to_dict()
        assert d["original_task"] == "test"

    def test_to_json(self):
        m = TaskManifest(original_task="test")
        j = m.to_json()
        parsed = json.loads(j)
        assert parsed["original_task"] == "test"

    def test_classification_info_defaults(self):
        c = ClassificationInfo()
        assert c.domain_key == ""
        assert c.confidence == 0.0
        assert c.is_primary is True


# ── Tests: loader.py ──────────────────────────────────────────────────

class TestBlueprintLoader:
    def test_singleton(self):
        a = BlueprintLoader()
        b = BlueprintLoader()
        assert a is b

    def test_loads_all_blueprints(self):
        loader = BlueprintLoader()
        bps = loader.get_all_blueprints()
        assert len(bps) >= 26

    def test_generic_blueprint_present(self):
        loader = BlueprintLoader()
        generic = loader.get_generic_blueprint()
        assert generic["archetype_key"] == "generic-task"
        assert len(generic["required_parameters"]) == 3

    def test_domain_list(self):
        loader = BlueprintLoader()
        domains = loader.get_domain_list()
        assert len(domains) >= 7

    def test_get_specific_blueprint(self):
        loader = BlueprintLoader()
        bp = loader.get_blueprint("1_web_development", "landing-page")
        assert bp is not None
        assert bp["archetype_key"] == "landing-page"

    def test_get_nonexistent_blueprint(self):
        loader = BlueprintLoader()
        bp = loader.get_blueprint("nonexistent", "nothing")
        assert bp is None


class TestMergeBlueprints:
    def test_merge_deduplicates_params(self, lading_bp, mobile_bp):
        merged = merge_blueprints([lading_bp, mobile_bp])
        # project_name appears in both, should appear once
        param_names = [p["name"] for p in merged["required_parameters"]]
        assert param_names.count("project_name") == 1
        assert "target_platforms" in param_names
        assert "offline_features" in param_names
        assert "business_type" in param_names

    def test_merge_combines_rules(self, lading_bp, mobile_bp):
        merged = merge_blueprints([lading_bp, mobile_bp])
        assert "Mobile-First obligatorio." in merged["domain_rules"]
        assert "Inmutabilidad de datos." in merged["domain_rules"]

    def test_merge_single_blueprint(self, lading_bp):
        merged = merge_blueprints([lading_bp])
        assert merged["archetype_key"] == "landing-page"

    def test_merge_empty_returns_generic(self):
        merged = merge_blueprints([])
        assert merged["archetype_key"] == "generic-task"


# ── Tests: agent.py helpers ───────────────────────────────────────────

class TestAgentHelpers:
    def test_param_name_to_human(self):
        assert _param_name_to_human("project_name") == "Project name"
        assert _param_name_to_human("") == ""
        assert _param_name_to_human("simple") == "Simple"

    def test_collect_known_params_empty(self):
        bps = [{"required_parameters": [{"name": "project_name"}], "optional_parameters": []}]
        result = _collect_known_params("Hazme algo", bps)
        assert result == {}

    def test_generate_questions_fallback_on_no_llm(self, multi_blueprints):
        with patch("cogniteam.scoping.agent.llm_complete", return_value=None):
            questions = _generate_questions(multi_blueprints, "Hazme algo", {})
            # 3 unique required params across both blueprints (project_name deduped by LLM but not by fallback)
            # Actually the fallback deduplicates too since we check `p not in required`
            assert len(questions) == 4  # project_name, business_type, target_platforms, offline_features

    def test_generate_questions_parses_llm_response(self, multi_blueprints):
        fake_json = json.dumps([
            {"parameter": "project_name", "question": "¿Nombre?", "options": None, "archetype": "Landing Page"},
            {"parameter": "target_platforms", "question": "¿Plataformas?", "options": ["iOS", "Android"], "archetype": "Mobile Apps"},
        ])
        with patch("cogniteam.scoping.agent.llm_complete", return_value=fake_json):
            questions = _generate_questions(multi_blueprints, "Hazme algo", {})
            assert len(questions) == 2
            assert questions[0]["parameter"] == "project_name"
            assert questions[1]["archetype"] == "Mobile Apps"

    def test_generate_questions_skips_known(self, multi_blueprints):
        known = {"project_name": "MiApp"}
        with patch("cogniteam.scoping.agent.llm_complete", return_value=None):
            questions = _generate_questions(multi_blueprints, "Hazme algo", known)
            assert len(questions) == 3  # business_type, target_platforms, offline_features

    def test_generate_clarified_task_with_multi(self, multi_blueprints):
        with patch("cogniteam.scoping.agent.llm_complete", return_value="Texto clarificado con múltiples arquetipos"):
            result = _generate_clarified_task(multi_blueprints, "Hazme algo", {"project_name": "Test"})
            assert "Texto clarificado" in result

    def test_generate_clarified_task_fallback(self, multi_blueprints):
        with patch("cogniteam.scoping.agent.llm_complete", return_value=None):
            result = _generate_clarified_task(multi_blueprints, "Hazme algo", {})
            assert "Tarea:" in result
            assert "Arquetipos:" in result

    def test_build_manifest_single(self, lading_bp):
        manifest = _build_manifest(
            [lading_bp],
            "Hazme una landing",
            {"project_name": "Test"},
            "Tarea clarificada",
        )
        assert manifest.status == "READY_TO_EXECUTE"
        assert manifest.classification.archetype_key == "landing-page"
        assert manifest.classification.is_primary is True
        assert len(manifest.secondary_classifications) == 0

    def test_build_manifest_multi(self, multi_blueprints):
        manifest = _build_manifest(
            multi_blueprints,
            "Hazme una app con landing",
            {"project_name": "Test", "target_platforms": "iOS"},
            "Tarea clarificada multi",
        )
        assert len(manifest.secondary_classifications) == 1
        assert manifest.secondary_classifications[0].archetype_key == "mobile-apps"
        assert manifest.secondary_classifications[0].is_primary is False
        assert len(manifest.constraints) == 4  # 2 from landing + 2 from mobile

    def test_build_manifest_merges_constraints(self, multi_blueprints):
        manifest = _build_manifest(multi_blueprints, "test", {}, "test")
        assert "Mobile-First obligatorio." in manifest.constraints
        assert "Inmutabilidad de datos." in manifest.constraints


# ── Integration: Full clarify_task with mocks ─────────────────────────

class TestClarifyTask:
    def test_clarify_task_single_archetype(self, lading_bp):
        with (
            patch("cogniteam.scoping.agent.get_blueprints_for_task_multi", return_value=[lading_bp]),
            patch("cogniteam.scoping.agent._generate_questions", return_value=[]),
            patch("cogniteam.scoping.agent._generate_clarified_task", return_value="Tarea clarificada."),
        ):
            manifest = clarify_task("Hazme una landing")
            assert manifest.classification.archetype_key == "landing-page"
            assert len(manifest.secondary_classifications) == 0

    def test_clarify_task_multi_archetype(self, multi_blueprints):
        with (
            patch("cogniteam.scoping.agent.get_blueprints_for_task_multi", return_value=multi_blueprints),
            patch("cogniteam.scoping.agent._generate_questions", return_value=[]),
            patch("cogniteam.scoping.agent._generate_clarified_task", return_value="Tarea clarificada multi."),
            patch("cogniteam.scoping.agent._confirm_classification", return_value=True),
        ):
            manifest = clarify_task("Hazme una app con landing")
            assert manifest.classification.archetype_key == "landing-page"
            assert len(manifest.secondary_classifications) == 1
            assert manifest.secondary_classifications[0].archetype_key == "mobile-apps"

    def test_clarify_task_with_questions(self, multi_blueprints):
        questions = [
            {"parameter": "project_name", "question": "¿Nombre?", "options": None, "archetype": "Landing Page"},
        ]
        with (
            patch("cogniteam.scoping.agent.get_blueprints_for_task_multi", return_value=multi_blueprints),
            patch("cogniteam.scoping.agent._generate_questions", return_value=questions),
            patch("cogniteam.scoping.agent._generate_clarified_task", return_value="Tarea clarificada."),
            patch("cogniteam.scoping.agent._ask_user", return_value="MiProyecto"),
            patch("cogniteam.scoping.agent._confirm_classification", return_value=True),
        ):
            manifest = clarify_task("Hazme app y landing")
            assert manifest.parameters["project_name"] == "MiProyecto"

    def test_clarify_task_low_confidence_confirmation(self, lading_bp):
        low_conf = dict(lading_bp)
        low_conf["_classification"] = {
            "domain_key": "1_web_development",
            "archetype_key": "landing-page",
            "confidence": 0.3,
            "reasoning": "Baja",
            "is_primary": True,
        }
        with (
            patch("cogniteam.scoping.agent.get_blueprints_for_task_multi", return_value=[low_conf]),
            patch("cogniteam.scoping.agent._generate_questions", return_value=[]),
            patch("cogniteam.scoping.agent._generate_clarified_task", return_value="Tarea clarificada."),
            patch("cogniteam.scoping.agent._confirm_classification", return_value=True),
        ):
            manifest = clarify_task("Hazme algo")
            assert manifest.status == "READY_TO_EXECUTE"

    def test_clarify_task_generic_fallback(self):
        generic_bp = {
            "domain_key": "generic",
            "domain_name": "Generic",
            "archetype_key": "generic-task",
            "archetype_name": "Tarea genérica",
            "priority": "clarification",
            "domain_rules": [],
            "stack": {},
            "required_parameters": [{"name": "task_goal", "question": "¿Objetivo?", "required": True, "type": "string"}],
            "optional_parameters": [],
            "_classification": {
                "domain_key": "generic",
                "archetype_key": "generic-task",
                "confidence": 0.0,
                "reasoning": "Fallo",
                "is_primary": True,
            },
        }
        with (
            patch("cogniteam.scoping.agent.get_blueprints_for_task_multi", return_value=[generic_bp]),
            patch("cogniteam.scoping.agent._generate_questions", return_value=[]),
            patch("cogniteam.scoping.agent._generate_clarified_task", return_value="Tarea genérica clarificada."),
            patch("cogniteam.scoping.agent._confirm_classification", return_value=True),
        ):
            manifest = clarify_task("Hazme algo raro")
            assert manifest.classification.archetype_key == "generic-task"
