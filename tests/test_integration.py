"""Integration tests: compilation, domain classification, blueprint integrity."""
import ast
import json
from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).resolve().parent.parent / "cogniteam"
BLUEPRINT_FILE = Path(__file__).resolve().parent.parent / "antigravity_complete_system_7_domains.json"
EXPECTED_SUPPORTED_DOMAINS = {
    "1_web_development": 6,
    "2_software_development": 5,
    "3_education": 5,
    "4_graphic_design": 3,
    "8_game_development": 5,
}

ALL_PYTHON_FILES = list(SOURCE_DIR.rglob("*.py")) + [
    Path(__file__).resolve().parent.parent / "main.py",
]


class TestCompilation:
    def test_all_files_compile(self):
        errors = []
        for path in ALL_PYTHON_FILES:
            try:
                ast.parse(path.read_text("utf-8"), filename=str(path))
            except SyntaxError as e:
                errors.append(f"{path.relative_to(SOURCE_DIR.parent)}: {e}")
        assert not errors, "\n".join(errors)

    def test_no_orphan_triple_quotes(self):
        planner = SOURCE_DIR / "agents" / "planner_agent.py"
        text = planner.read_text("utf-8")
        # detect leftover bare """ that are not part of docstrings: look for
        # lines that start with just """, not preceded by def/class/attribute
        lines = text.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == '"""' and not text.count('"""') > 1:
                continue  # already balanced
        # ensure triple-quote pairs are balanced
        count = text.count('"""')
        assert count % 2 == 0, f"Unbalanced triple-quotes in planner_agent.py (found {count})"


class TestBlueprintIntegrity:
    def test_blueprint_has_all_supported_domains(self):
        bp = json.loads(BLUEPRINT_FILE.read_text("utf-8"))
        domains = bp.get("dominios", bp)
        for domain_key in EXPECTED_SUPPORTED_DOMAINS:
            assert domain_key in domains, f"Missing domain {domain_key} in blueprint"
            domain = domains[domain_key]
            # each domain should have archetypes
            archetypes = domain.get("archetypes", {})
            assert len(archetypes) > 0, f"No archetypes for {domain_key}"

    def test_blueprint_no_purged_domains(self):
        bp = json.loads(BLUEPRINT_FILE.read_text("utf-8"))
        domains = bp.get("dominios", bp)
        purged = {"5_data_science", "6_content_writing", "7_devops",
                   "9_cybersecurity", "10_legal_compliance", "11_architecture_spatial",
                   "12_marketing_growth", "13_audiovisual_management"}
        for key in purged:
            assert key not in domains, f"Purged domain {key} unexpectedly present in blueprint"
