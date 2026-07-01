"""Tests for Skills/Policy module (NanoResearch + SAGE)."""
import json
import os
import random
import tempfile
import time

import pytest
from cogniteam.memory.skills.skills import SkillPolicyManager, Skill, SkillPreference, GraphMemoryNode, get_skills


@pytest.fixture
def mgr():
    with tempfile.TemporaryDirectory() as td:
        m = SkillPolicyManager(namespace="test")
        m._persist_dir = os.path.join(td, "skills_test")
        m.load()
        yield m


def test_register_skill(mgr):
    sid = mgr.register_skill("CodeReview", "Review code for bugs", category="development")
    assert sid in mgr.skills
    assert mgr.skills[sid].name == "CodeReview"
    assert mgr.skills[sid].category == "development"


def test_register_skill_creates_sage_node(mgr):
    sid = mgr.register_skill("Debug", "Find and fix errors")
    assert len(mgr.nodes) == 1
    assert mgr.graph.number_of_nodes() == 1


def test_get_skills_by_category(mgr):
    mgr.register_skill("S1", "Desc1", category="dev")
    mgr.register_skill("S2", "Desc2", category="dev")
    mgr.register_skill("S3", "Desc3", category="ops")
    dev_skills = mgr.get_skills_by_category("dev")
    assert len(dev_skills) == 2
    ops_skills = mgr.get_skills_by_category("ops")
    assert len(ops_skills) == 1


def test_get_top_skills(mgr):
    for i in range(5):
        sid = mgr.register_skill(f"S{i}", f"Skill {i}")
        mgr.record_skill_usage(sid, success=i > 2, score=i / 5.0)
    tops = mgr.get_top_skills(top_k=3)
    assert len(tops) == 3
    assert all(s.usage_count > 0 for s in tops)


def test_record_skill_usage(mgr):
    sid = mgr.register_skill("Test", "A test skill")
    mgr.record_skill_usage(sid, success=True, score=0.8)
    assert mgr.skills[sid].usage_count == 1
    assert mgr.skills[sid].success_rate == 1.0
    assert mgr.skills[sid].avg_score == 0.8

    mgr.record_skill_usage(sid, success=True, score=0.4)
    assert mgr.skills[sid].usage_count == 2
    assert mgr.skills[sid].success_rate == 1.0


def test_record_skill_usage_multiple(mgr):
    sid = mgr.register_skill("Test", "A test")
    mgr.record_skill_usage(sid, success=True, score=1.0)
    mgr.record_skill_usage(sid, success=False, score=0.0)
    assert mgr.skills[sid].usage_count == 2
    assert mgr.skills[sid].success_rate == 0.5
    assert mgr.skills[sid].avg_score == 0.5


def test_sdpo_preference(mgr):
    a = mgr.register_skill("SkillA", "First skill")
    b = mgr.register_skill("SkillB", "Second skill")
    mgr.add_preference(a, b, preferred=a, context="testing")
    assert len(mgr.preferences) == 1
    assert mgr.skills[a].avg_score > mgr.skills[b].avg_score


def test_get_preferred_skill(mgr):
    a = mgr.register_skill("SkillA", "Best skill")
    b = mgr.register_skill("SkillB", "Worse skill")
    mgr.add_preference(a, b, a, "general")
    preferred = mgr.get_preferred_skill("general", [a, b])
    assert preferred == a


def test_get_preferred_skill_no_candidates(mgr):
    assert mgr.get_preferred_skill("anything", []) is None


def test_sage_write_creates_node(mgr):
    nid = mgr.sage_write("Python is a programming language", node_type="fact")
    assert nid in mgr.nodes
    assert mgr.graph.has_node(nid)


def test_sage_write_connects_related(mgr):
    nid1 = mgr.sage_write("Python is a great language for web development")
    nid2 = mgr.sage_write("Django is a Python web framework")
    nid3 = mgr.sage_write("JavaScript runs in the browser")
    # Python and Django should be related
    edges = [(e.source, e.target) for e in mgr.edges]
    assert any(nid1 in e for e in edges) or any(nid2 in e for e in edges)


def test_sage_read_returns_relevant(mgr):
    mgr.sage_write("Python is for programming")
    mgr.sage_write("Cats are animals")
    results = mgr.sage_read("Python", top_k=5)
    contents = [r["content"] for r in results]
    assert any("Python" in c for c in contents)


def test_sage_read_empty_graph(mgr):
    results = mgr.sage_read("anything")
    assert results == []


def test_sage_consolidate_removes_low_importance(mgr):
    for i in range(20):
        nid = mgr._add_sage_node(f"Node {i}", "observation", confidence=0.1,
                                 metadata={"i": i})
        if i < 5:
            mgr.nodes[nid].importance = 1.0
        else:
            mgr.nodes[nid].importance = 0.01
    mgr._sage_consolidate()
    # Should have removed ~5 low-importance nodes
    assert len(mgr.nodes) < 20


def test_retrieve_with_memory_basic(mgr):
    mgr.register_skill("PythonDev", "Python development skill", category="dev")
    mgr.sage_write("Python is a programming language", "fact")
    result = mgr.retrieve_with_memory("Python")
    assert "query" in result
    assert result["query"] == "Python"
    assert len(result["graph_memory"]) >= 1 or len(result["relevant_skills"]) >= 1


def test_co_evolve_basic(mgr):
    for i in range(3):
        mgr.register_skill(f"S{i}", f"Skill {i}")

    def eval_fn(skill_id):
        return random.random()

    result = mgr.co_evolve(eval_fn, num_rounds=2, verbose=False)
    assert result["rounds"] == 2
    assert len(result["history"]) == 2


def test_persistence(mgr):
    sid = mgr.register_skill("PersistSkill", "Skill for persistence test")
    mgr.sage_write("Test knowledge", "fact")
    mgr.add_preference(sid, sid, sid, "context")
    mgr.save()

    m2 = SkillPolicyManager(namespace="test")
    m2._persist_dir = mgr._persist_dir
    m2.load()
    assert len(m2.skills) >= 1
    assert "PersistSkill" in [s.name for s in m2.skills.values()]
    assert len(m2.nodes) >= 1
