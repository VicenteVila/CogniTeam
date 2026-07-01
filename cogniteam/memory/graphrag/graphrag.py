"""
GraphRAG (Microsoft) - Knowledge Graph with community detection and summarization.

Reference: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
(https://arxiv.org/abs/2404.16130)

Key algorithms:
1. Entity extraction from text chunks via LLM
2. Knowledge graph construction with entity resolution
3. Leiden community detection on the graph
4. Community summarization (generating summaries per community)
5. Global search: answer query by combining community summaries
6. Local search: traverse graph from query-relevant entities
"""

import json
import math
import os
import re
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from cogniteam.config.settings import settings


class GraphRAGMemory:
    """GraphRAG implementation with community detection and summarization."""

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.graph: nx.Graph = nx.Graph()
        self.entity_descriptions: Dict[str, str] = {}
        self.source_chunks: Dict[str, str] = {}
        self.communities: Dict[int, List[str]] = {}
        self.community_summaries: Dict[int, str] = {}
        self._persist_dir = os.path.join(
            settings.project_root, ".cogniteam", "graphrag", namespace
        )

    # ── Entity Extraction ──────────────────────────────────────────────

    def extract_entities_from_text(
        self, text: str, chunk_id: str
    ) -> List[Dict[str, Any]]:
        """Extract entities and relationships from text using LLM."""
        try:
            from cogniteam.tools.utils.llm import llm_complete

            prompt = (
                "Extract all named entities and their relationships from the text below.\n"
                "Return a JSON object with keys 'entities' (list of {name, type, description}) "
                "and 'relationships' (list of {source, target, relation}).\n"
                "Types can be: person, organization, location, technology, concept, tool, code, file, other.\n\n"
                f"Text:\n{text[:8000]}\n\nJSON:"
            )
            raw = llm_complete(prompt, task="extract", max_tokens=4096, timeout_seconds=60)
            if raw:
                result = json.loads(raw)
                entities = result.get("entities", [])
                rels = result.get("relationships", [])
                self._add_to_graph(entities, rels, chunk_id)
                return entities
        except Exception:
            pass

        return self._extract_entities_fallback(text, chunk_id)

    def _extract_entities_fallback(
        self, text: str, chunk_id: str
    ) -> List[Dict[str, Any]]:
        """Regex-based fallback for entity extraction."""
        entities = []
        rels = []
        seen: Set[str] = set()

        # Simple noun phrase extraction
        for match in re.finditer(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text):
            name = match.group(1).strip()
            if len(name) > 2 and name not in seen:
                seen.add(name)
                entities.append(
                    {"name": name, "type": "concept", "description": name}
                )
        self._add_to_graph(entities, rels, chunk_id)
        return entities

    def _add_to_graph(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        source_chunk: str,
    ):
        for ent in entities:
            name = ent["name"].strip()
            if not name:
                continue
            self.graph.add_node(
                name,
                type=ent.get("type", "concept"),
                description=ent.get("description", ""),
            )
            if name not in self.entity_descriptions:
                self.entity_descriptions[name] = ent.get("description", "")
        for rel in relationships:
            s, t = rel.get("source", "").strip(), rel.get("target", "").strip()
            if s and t:
                if not self.graph.has_node(s):
                    self.graph.add_node(s, type="concept", description=s)
                if not self.graph.has_node(t):
                    self.graph.add_node(t, type="concept", description=t)
                self.graph.add_edge(
                    s, t, relation=rel.get("relation", "related_to")
                )
        self.source_chunks[source_chunk] = json.dumps(
            {"entities": entities, "relationships": relationships}
        )

    # ── Community Detection (Leiden Algorithm) ─────────────────────────

    def detect_communities(self) -> Dict[int, List[str]]:
        """Run Leiden community detection on the graph."""
        if self.graph.number_of_nodes() < 2:
            return {0: list(self.graph.nodes())}

        try:
            from graspologic.partition import leiden

            partition_map = leiden(
                self.graph,
                random_seed=42,
                trials=10,
            )
        except ImportError:
            # Fallback: greedy modularity
            try:
                from networkx.algorithms.community import greedy_modularity_communities

                communities = list(greedy_modularity_communities(self.graph))
                partition_map = {}
                for cid, comm in enumerate(communities):
                    for node in comm:
                        partition_map[node] = cid
            except Exception:
                partition_map = {n: 0 for n in self.graph.nodes()}

        communities: Dict[int, List[str]] = defaultdict(list)
        for node, cid in partition_map.items():
            communities[cid].append(node)
        self.communities = dict(communities)
        return self.communities

    def summarize_communities(self) -> Dict[int, str]:
        """Generate summaries for each detected community using LLM."""
        if not self.communities:
            self.detect_communities()

        summaries = {}
        for cid, members in self.communities.items():
            if not members:
                summaries[cid] = f"Community {cid}: empty"
                continue

            descriptions = []
            for m in members:
                desc = self.entity_descriptions.get(m, "")
                neighbors = list(self.graph.neighbors(m))
                neighbor_str = ", ".join(neighbors[:10])
                descriptions.append(
                    f"- {m}: {desc[:200]} (connected to: {neighbor_str})"
                )
            community_text = "\n".join(descriptions)

            try:
                from cogniteam.tools.utils.llm import llm_complete

                prompt = (
                    "Summarize the following community of entities and their relationships "
                    "in one paragraph. Focus on the main theme and how entities connect.\n\n"
                    f"{community_text}\n\nSummary:"
                )
                result = llm_complete(prompt, task="fast", max_tokens=512, timeout_seconds=60)
                if result:
                    summaries[cid] = result
                else:
                    summaries[cid] = f"Community {cid}: {len(members)} members."
            except Exception:
                summaries[cid] = f"Community {cid}: {len(members)} members."

        self.community_summaries = summaries
        return summaries

    # ── Search ─────────────────────────────────────────────────────────

    def global_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Global search: rank community summaries against query."""
        if not self.community_summaries:
            self.summarize_communities()

        scored = []
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        for cid, summary in self.community_summaries.items():
            summary_lower = summary.lower()
            # Token overlap score
            token_overlap = len(query_tokens & set(summary_lower.split()))
            # Entity overlap score
            members = self.communities.get(cid, [])
            entity_overlap = sum(
                1 for m in members if m.lower() in query_lower
            )
            score = token_overlap * 0.5 + entity_overlap * 2.0
            scored.append((score, cid, summary))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, cid, summary in scored[:top_k]:
            results.append(
                {
                    "community_id": cid,
                    "score": score,
                    "summary": summary,
                    "members": self.communities.get(cid, []),
                }
            )
        return results

    def local_search(self, query: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """Local search: find entities matching query, then traverse neighbors."""
        query_lower = query.lower()

        # Find seed entities
        seeds = []
        for node in self.graph.nodes():
            if node.lower() in query_lower or query_lower in node.lower():
                desc = self.entity_descriptions.get(node, "")
                if desc and (
                    query_lower in desc.lower()
                    or any(t in desc.lower() for t in query_lower.split())
                ):
                    seeds.append(node)

        if not seeds:
            # Try partial matching
            for node in self.graph.nodes():
                node_lower = node.lower()
                for qt in query_lower.split():
                    if len(qt) > 3 and qt in node_lower:
                        seeds.append(node)
                        break

        # BFS from seeds
        visited: Set[str] = set()
        results = []
        for seed in seeds[:5]:
            for node, dist in nx.single_source_shortest_path_length(
                self.graph, seed, cutoff=max_hops
            ).items():
                if node not in visited:
                    visited.add(node)
                    edge_data = []
                    for neighbor in self.graph.neighbors(node):
                        edge = self.graph.get_edge_data(node, neighbor)
                        if edge and edge.get("relation"):
                            edge_data.append(
                                f"->{neighbor} ({edge['relation']})"
                            )
                    results.append(
                        {
                            "entity": node,
                            "type": self.graph.nodes[node].get("type", "concept"),
                            "description": self.entity_descriptions.get(node, ""),
                            "distance_from_seed": dist,
                            "connections": edge_data[:5],
                        }
                    )

        results.sort(key=lambda x: x["distance_from_seed"])
        return results[:10]

    def hybrid_search(
        self, query: str, top_k_global: int = 3, top_k_local: int = 5
    ) -> Dict[str, Any]:
        """Combine global and local search results."""
        global_results = self.global_search(query, top_k=top_k_global)
        local_results = self.local_search(query, max_hops=2)

        return {
            "query": query,
            "global_communities": global_results,
            "local_entities": local_results,
            "summary": self._generate_hybrid_summary(
                query, global_results, local_results
            ),
        }

    def _generate_hybrid_summary(
        self,
        query: str,
        global_results: List[Dict[str, Any]],
        local_results: List[Dict[str, Any]],
    ) -> str:
        context_parts = []
        for gr in global_results:
            context_parts.append(f"Community: {gr['summary']}")
        for lr in local_results[:3]:
            context_parts.append(
                f"Entity '{lr['entity']}' ({lr['type']}): {lr['description']}"
            )
        context = "\n".join(context_parts)

        if not context:
            return "No relevant context found."

        try:
            from cogniteam.tools.utils.llm import llm_complete

            prompt = (
                f"Answer the query based solely on the provided context.\n\n"
                f"Query: {query}\n\nContext:\n{context}\n\nAnswer:"
            )
            result = llm_complete(prompt, task="fast", max_tokens=1024, timeout_seconds=60)
            if result:
                return result
        except Exception:
            pass
        return context

    def add_text(self, text: str, source: str = ""):
        """Add text to the GraphRAG memory, extracting entities and relationships."""
        chunk_id = source or f"chunk_{len(self.source_chunks)}"
        self.extract_entities_from_text(text, chunk_id)

    # ── Persistence ────────────────────────────────────────────────────

    def save(self):
        """Persist GraphRAG state to disk."""
        os.makedirs(self._persist_dir, exist_ok=True)
        # Graph
        data = nx.node_link_data(self.graph)
        with open(os.path.join(self._persist_dir, "graph.json"), "w") as f:
            json.dump(data, f, indent=2)
        # Communities
        with open(
            os.path.join(self._persist_dir, "communities.json"), "w"
        ) as f:
            json.dump(
                {
                    "communities": self.communities,
                    "summaries": self.community_summaries,
                },
                f,
                indent=2,
            )
        # Entities
        with open(
            os.path.join(self._persist_dir, "entities.json"), "w"
        ) as f:
            json.dump(self.entity_descriptions, f, indent=2)

    def load(self):
        """Load GraphRAG state from disk."""
        try:
            with open(os.path.join(self._persist_dir, "graph.json")) as f:
                data = json.load(f)
                self.graph = nx.node_link_graph(data)
            with open(
                os.path.join(self._persist_dir, "communities.json")
            ) as f:
                data = json.load(f)
                self.communities = {
                    int(k): v for k, v in data.get("communities", {}).items()
                }
                self.community_summaries = {
                    int(k): v
                    for k, v in data.get("summaries", {}).items()
                }
            with open(
                os.path.join(self._persist_dir, "entities.json")
            ) as f:
                self.entity_descriptions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass


# Global registry for easy access
_instances: Dict[str, GraphRAGMemory] = {}


def get_graphrag(namespace: str = "default") -> GraphRAGMemory:
    if namespace not in _instances:
        _instances[namespace] = GraphRAGMemory(namespace)
        _instances[namespace].load()
    return _instances[namespace]
