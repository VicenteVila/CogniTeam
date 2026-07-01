"""Tests for GraphRAG memory module (without LLM)."""
import json
import os
import tempfile

import pytest
from unittest.mock import patch, MagicMock

from cogniteam.memory.graphrag import GraphRAGMemory, get_graphrag


@pytest.fixture
def graphrag():
    with tempfile.TemporaryDirectory() as td:
        persist_dir = os.path.join(td, "graphrag_test")
        gm = GraphRAGMemory(namespace="test")
        gm._persist_dir = persist_dir
        gm.load()
        yield gm


def test_extract_entities_fallback(graphrag):
    text = "Alice and Bob went to Paris. Microsoft announced new AI features."
    entities = graphrag._extract_entities_fallback(text, "chunk_1")
    names = [e["name"] for e in entities]
    assert "Alice" in names or "Bob" in names or "Paris" in names
    assert "Microsoft" in names or "AI" in names
    assert len(entities) >= 2


def test_add_to_graph(graphrag):
    entities = [
        {"name": "Alice", "type": "person", "description": "A person"},
        {"name": "Bob", "type": "person", "description": "Another person"},
    ]
    rels = [
        {"source": "Alice", "target": "Bob", "relation": "knows"},
    ]
    graphrag._add_to_graph(entities, rels, "chunk_1")
    assert graphrag.graph.number_of_nodes() == 2
    assert graphrag.graph.has_edge("Alice", "Bob")


def test_entity_relationship_missing_nodes_created(graphrag):
    entities = [{"name": "Alice", "type": "person", "description": "A person"}]
    rels = [
        {"source": "Alice", "target": "Microsoft", "relation": "works_at"},
    ]
    graphrag._add_to_graph(entities, rels, "chunk_1")
    assert graphrag.graph.has_node("Microsoft")
    assert graphrag.graph.has_edge("Alice", "Microsoft")


def test_detect_communities_small_graph(graphrag):
    graphrag._add_to_graph(
        [{"name": n, "type": "concept", "description": n} for n in ["A", "B"]],
        [{"source": "A", "target": "B", "relation": "connects"}],
        "chunk_1",
    )
    comms = graphrag.detect_communities()
    assert isinstance(comms, dict)
    assert len(comms) >= 1
    all_nodes = set()
    for members in comms.values():
        all_nodes.update(members)
    assert "A" in all_nodes


def test_global_search_basic(graphrag):
    nodes = ["Python", "Django", "Flask", "JavaScript", "React"]
    graphrag._add_to_graph(
        [{"name": n, "type": "language" if n in ("Python", "JavaScript") else "framework", "description": f"{n} is a tool"}
         for n in nodes],
        [{"source": "Python", "target": "Django", "relation": "has_framework"},
         {"source": "Python", "target": "Flask", "relation": "has_framework"},
         {"source": "JavaScript", "target": "React", "relation": "has_framework"}],
        "chunk_1",
    )
    graphrag.detect_communities()
    graphrag.summarize_communities()
    results = graphrag.global_search("Python", top_k=5)
    assert len(results) >= 1
    # Fallback summary includes member count; check query returned relevant results
    top_result = results[0]
    assert top_result["score"] > 0
    assert "Python" in str(top_result["members"])


def test_local_search_finds_seeds(graphrag):
    graphrag._add_to_graph(
        [{"name": "Python", "type": "language", "description": "Programming language"},
         {"name": "Django", "type": "framework", "description": "Python web framework"}],
        [{"source": "Python", "target": "Django", "relation": "has_framework"}],
        "chunk_1",
    )
    results = graphrag.local_search("Python", max_hops=2)
    assert len(results) >= 1
    entities = [r["entity"] for r in results]
    assert "Python" in entities


def test_hybrid_search(graphrag):
    graphrag._add_to_graph(
        [{"name": "Python", "type": "language", "description": "A language"},
         {"name": "Django", "type": "framework", "description": "A framework"}],
        [{"source": "Python", "target": "Django", "relation": "has_framework"}],
        "chunk_1",
    )
    graphrag.detect_communities()
    graphrag.summarize_communities()
    result = graphrag.hybrid_search("Python", top_k_global=1, top_k_local=5)
    assert "query" in result
    assert "global_communities" in result
    assert "local_entities" in result


def test_persistence(graphrag):
    graphrag._add_to_graph(
        [{"name": "Alice", "type": "person", "description": "A person"}],
        [],
        "chunk_1",
    )
    graphrag.save()
    gm2 = GraphRAGMemory(namespace="test")
    gm2._persist_dir = graphrag._persist_dir
    gm2.load()
    assert gm2.graph.has_node("Alice")
    assert gm2.graph.number_of_nodes() == 1


def test_add_text_triggers_extraction(graphrag):
    with patch.object(graphrag, "extract_entities_from_text") as mock_extract:
        mock_extract.return_value = []
        graphrag.add_text("Some text here", source="test_source")
        mock_extract.assert_called_once_with("Some text here", "test_source")


def test_save_load_communities(graphrag):
    graphrag._add_to_graph(
        [{"name": n, "type": "concept", "description": n} for n in ["A", "B", "C"]],
        [{"source": "A", "target": "B", "relation": "x"},
         {"source": "B", "target": "C", "relation": "y"}],
        "chunk_1",
    )
    graphrag.detect_communities()
    graphrag.community_summaries = {0: "Community zero summary"}
    graphrag.save()
    gm2 = GraphRAGMemory(namespace="test")
    gm2._persist_dir = graphrag._persist_dir
    gm2.load()
    assert len(gm2.communities) > 0
    assert 0 in gm2.community_summaries


def test_global_registry():
    from cogniteam.config.settings import settings as cfg
    with tempfile.TemporaryDirectory() as td:
        orig = cfg.project_root
        cfg.project_root = td
        try:
            inst = get_graphrag("registry_test")
            assert inst is get_graphrag("registry_test")
            assert inst is not get_graphrag("other_ns")
        finally:
            cfg.project_root = orig
