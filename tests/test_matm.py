"""Tests for MATM memory module."""
import json
import os
import tempfile
import time

import pytest
from cogniteam.memory.matm.matm import MATM, ExpertiseEntry, MemoryChunk, DelegationRecord, get_matm


@pytest.fixture
def matm():
    with tempfile.TemporaryDirectory() as td:
        m = MATM(namespace="test")
        m._persist_dir = os.path.join(td, "matm_test")
        m.load()
        yield m


def test_register_agent(matm):
    matm.register_agent("AgentA", topics=["Python", "Django"])
    assert "AgentA" in matm.agent_expertise
    assert "Python" in matm.agent_expertise["AgentA"]
    assert "Django" in matm.agent_expertise["AgentA"]


def test_declare_expertise(matm):
    matm.declare_expertise("AgentA", "AgentB", "Python", keywords=["django", "flask"])
    entries = matm.expertise_directory["AgentB"]
    assert len(entries) == 1
    assert entries[0].agent_a == "AgentA"
    assert entries[0].topic == "Python"
    assert "django" in entries[0].keywords


def test_declare_expertise_updates_existing(matm):
    matm.declare_expertise("AgentA", "AgentB", "Python", keywords=["v1"])
    time.sleep(0.01)
    matm.declare_expertise("AgentA", "AgentB", "Python", keywords=["v2"])
    entries = matm.expertise_directory["AgentB"]
    assert len(entries) == 1
    assert entries[0].keywords == ["v2"]


def test_update_expertise_performance_ema(matm):
    matm.declare_expertise("AgentA", "AgentB", "Python")
    matm.update_expertise_performance("AgentB", "AgentA", "Python", success=True)
    entry = matm.expertise_directory["AgentB"][0]
    assert entry.interaction_count == 1
    assert entry.performance == 1.0
    assert entry.confidence == 0.55

    matm.update_expertise_performance("AgentB", "AgentA", "Python", success=False)
    assert entry.interaction_count == 2
    assert 0.0 < entry.performance < 1.0
    assert entry.confidence == 0.5


def test_find_expert_by_topic(matm):
    matm.declare_expertise("AgentA", "AgentB", "Python", initial_confidence=0.9)
    matm.declare_expertise("AgentC", "AgentB", "JavaScript", initial_confidence=0.8)
    experts = matm.find_expert("AgentB", "Python", top_k=3)
    assert len(experts) >= 1
    assert any(e.agent_a == "AgentA" for e in experts)


def test_find_expert_returns_empty(matm):
    experts = matm.find_expert("UnknownAgent", "anything")
    assert experts == []


def test_store_memory(matm):
    chunk_id = matm.store_memory("AgentA", "Python is great", "Python",
                                 tags=["lang"], importance=0.8)
    assert chunk_id in matm.memories
    mem = matm.memories[chunk_id]
    assert mem.agent_id == "AgentA"
    assert mem.content == "Python is great"


def test_store_memory_with_sharing_updates_expertise(matm):
    matm.store_memory("AgentA", "Data about X", "DataScience",
                      tags=["ml", "ai"], shared_with=["AgentB", "AgentC"])
    assert "AgentB" in matm.expertise_directory
    assert "AgentC" in matm.expertise_directory
    assert any(e.agent_a == "AgentA" for e in matm.expertise_directory["AgentB"])


def test_retrieve_memories_by_topic(matm):
    matm.store_memory("AgentA", "Content about Python", "Python")
    matm.store_memory("AgentA", "Content about JS", "JavaScript")
    matm.store_memory("AgentB", "More Python", "Python")
    results = matm.retrieve_memories("AgentA", "Python", top_k=10)
    topics = {m.topic for m in results}
    assert "Python" in topics
    # Python memory should be ranked first (higher relevance score)
    assert results[0].topic == "Python"


def test_retrieve_memories_includes_shared(matm):
    matm.store_memory("AgentA", "Shared secret", "Secret", shared_with=["AgentB"])
    results = matm.retrieve_memories("AgentB", "secret", top_k=10)
    contents = [m.content for m in results]
    assert "Shared secret" in contents


def test_transactive_retrieve_delegates_when_confident(matm):
    matm.declare_expertise("AgentA", "AgentB", "Python", initial_confidence=0.9)
    matm.store_memory("AgentA", "Python answer 42", "Python")
    matm.update_expertise_performance("AgentB", "AgentA", "Python", success=True)

    result = matm.transactive_retrieve("AgentB", "Python")
    assert result["method"] == "delegation"
    assert result["delegate"] == "AgentA"
    assert len(result["memories"]) > 0


def test_transactive_retrieve_fallback(matm):
    result = matm.transactive_retrieve("AgentB", "unknown topic")
    assert result["method"] == "direct"


def test_synthesize_knowledge_fallback(matm):
    matm.store_memory("AgentA", "Alice uses Python", "Python")
    result = matm.synthesize_knowledge("AgentA", "Python")
    assert isinstance(result, str)
    assert len(result) > 0


def test_synthesize_knowledge_no_memories(matm):
    result = matm.synthesize_knowledge("AgentA", "nonexistent")
    assert result == "No relevant memories found."


def test_persistence(matm):
    matm.store_memory("AgentA", "Persist test", "Testing")
    matm.declare_expertise("AgentA", "AgentB", "Testing")
    matm.save()

    m2 = MATM(namespace="test")
    m2._persist_dir = matm._persist_dir
    m2.load()
    assert len(m2.memories) == 1
    assert len(m2.expertise_directory) >= 1
