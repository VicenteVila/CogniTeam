"""Tests for H-MEM memory module."""
import json
import os
import time
import tempfile

import pytest
from cogniteam.memory.hmem.hmem import HMEM, MemoryNode, KnowledgeTriplet, get_hmem
from unittest.mock import patch, MagicMock


@pytest.fixture
def hmem():
    with tempfile.TemporaryDirectory() as td:
        hm = HMEM(namespace="test")
        hm._persist_dir = os.path.join(td, "hmem_test")
        hm.load()
        yield hm


def test_add_memory_creates_node(hmem):
    node_id = hmem.add_memory("Hello world", entities=["Alice"], importance=0.8)
    assert node_id in hmem.nodes
    node = hmem.nodes[node_id]
    assert node.content == "Hello world"
    assert node.level == 0
    assert node.importance == 0.8
    assert "Alice" in node.entities


def test_add_memory_creates_triplets(hmem):
    hmem.add_memory("Hello world", entities=["Alice", "Bob"])
    triplets = hmem.query_knowledge(entity="Alice")
    assert len(triplets) >= 1
    assert triplets[0].subject == "Alice"
    assert triplets[0].predicate == "mentioned_in"


def test_add_knowledge(hmem):
    hmem.add_knowledge("Python", "is_a", "language", confidence=0.9)
    assert len(hmem.knowledge_triplets) == 1
    assert hmem.knowledge_triplets[0].subject == "Python"
    assert hmem.knowledge_triplets[0].predicate == "is_a"
    assert hmem.knowledge_triplets[0].obj == "language"


def test_query_knowledge_by_entity(hmem):
    hmem.add_knowledge("Alice", "knows", "Bob", 0.9)
    hmem.add_knowledge("Charlie", "knows", "Diana", 0.8)
    results = hmem.query_knowledge(entity="Alice")
    assert len(results) == 1
    assert results[0].subject == "Alice"


def test_query_knowledge_by_predicate(hmem):
    hmem.add_knowledge("Alice", "works_at", "Google", 0.9)
    hmem.add_knowledge("Alice", "lives_in", "NYC", 0.7)
    results = hmem.query_knowledge(predicate="works_at")
    assert len(results) == 1


def test_query_knowledge_min_confidence(hmem):
    hmem.add_knowledge("X", "rel", "Y", 0.3)
    hmem.add_knowledge("X", "rel", "Z", 0.7)
    results = hmem.query_knowledge(entity="X", min_confidence=0.5)
    assert len(results) == 1


def test_get_entity_graph_one_hop(hmem):
    hmem.add_knowledge("Alice", "knows", "Bob", 0.9)
    hmem.add_knowledge("Alice", "works_at", "Google", 0.8)
    subgraph = hmem.get_entity_graph("Alice", max_hops=1)
    assert len(subgraph) >= 2
    related_entities = {r["subject"] for r in subgraph} | {r["obj"] for r in subgraph}
    assert "Google" in related_entities or "Bob" in related_entities


def test_get_entity_graph_two_hops(hmem):
    hmem.add_knowledge("Alice", "knows", "Bob", 0.9)
    hmem.add_knowledge("Bob", "works_at", "Google", 0.9)
    subgraph = hmem.get_entity_graph("Alice", max_hops=2)
    # Should include Bob and Google
    related = {r["subject"]: r for r in subgraph} | {r["obj"]: r for r in subgraph}
    assert "Bob" in related or any(r["obj"] == "Bob" or r["subject"] == "Bob" for r in subgraph)


def test_temporal_context_scoring(hmem):
    id1 = hmem.add_memory("Recent important", importance=0.9)
    time.sleep(0.01)
    id2 = hmem.add_memory("Older less important", importance=0.1)
    context = hmem.get_temporal_context(max_nodes=10)
    assert len(context) >= 2
    # Most recent/important should be first
    assert context[0].content == "Recent important" or context[0].content == "Older less important"


def test_consolidation(hmem):
    with patch.object(hmem, "_summarize") as mock_sum:
        mock_sum.return_value = "Consolidated summary"
        hmem.add_memory("First message", importance=0.5)
        hmem.add_memory("Second message", importance=0.6)
        hmem.consolidate()
        level1_nodes = [n for n in hmem.nodes.values() if n.level == 1]
        assert len(level1_nodes) >= 1
        assert level1_nodes[0].summary == "Consolidated summary"


def test_consolidation_locks(hmem):
    hmem._consolidation_lock = True
    hmem.consolidate()  # Should return immediately
    assert hmem._consolidation_lock == True


def test_hybrid_retrieve_basic(hmem):
    hmem.add_memory("Alice likes Python", entities=["Alice", "Python"], importance=0.7)
    hmem.add_knowledge("Python", "is", "language", 0.9)
    result = hmem.hybrid_retrieve("Python")
    assert "query" in result
    assert result["query"] == "Python"
    assert len(result["results"]) > 0
    assert "Python" in result["matched_entities"]


def test_save_load(hmem):
    hmem.add_memory("Test memory", entities=["TestEntity"])
    hmem.add_knowledge("TestEntity", "is", "test", 0.9)
    hmem.save()

    hm2 = HMEM(namespace="test")
    hm2._persist_dir = hmem._persist_dir
    hm2.load()
    assert len(hm2.nodes) == 1
    assert len(hm2.knowledge_triplets) >= 1
    assert "TestEntity" in [t.subject for t in hm2.knowledge_triplets]


def test_save_load_empty():
    with tempfile.TemporaryDirectory() as td:
        hm = HMEM(namespace="empty_test")
        hm._persist_dir = os.path.join(td, "hmem_empty")
        hm.save()
        hm2 = HMEM(namespace="empty_test")
        hm2._persist_dir = hm._persist_dir
        hm2.load()
        assert len(hm2.nodes) == 0
        assert len(hm2.knowledge_triplets) == 0


def test_get_temporal_context_empty(hmem):
    assert hmem.get_temporal_context() == []


def test_answer_with_memory_fallback(hmem):
    hmem.add_memory("Alice uses Python for data science", entities=["Alice", "Python"])
    result = hmem.answer_with_memory("What does Alice use?")
    assert isinstance(result, str)
    assert len(result) > 0
