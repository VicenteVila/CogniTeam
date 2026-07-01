"""
H-MEM: Hybrid Memory with Temporal Memory Tree + Knowledge Graph.

Reference: "Elevating Lifelong Personalized Dialogue Systems through
Hybrid Retrieval-Augmented Generations" (CUHK + Huawei)

Key algorithms:
1. Temporal Memory Tree (TMT): hierarchical tree with temporal dimensions
   - Level 0: Most recent turns (session-level, seconds-minutes)
   - Level 1: Focused episodes (minutes-hours)
   - Level 2: Daily summaries (hours-days)
   - Level 3: Weekly/Monthly patterns (weeks-months)
   - Level 4: Long-term knowledge (months-years)
2. Knowledge Graph (KG): entities and relationships from conversations
3. Memory consolidation: recent -> working -> long-term via summarization
4. Hybrid retrieval: traverse TMT + query KG, then merge and rank
"""

import json
import math
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from cogniteam.config.settings import settings


@dataclass
class MemoryNode:
    """A node in the Temporal Memory Tree."""
    id: str
    content: str
    timestamp: float
    level: int  # 0=session, 1=episode, 2=daily, 3=weekly, 4=longterm
    importance: float = 0.0  # [0,1] how important/novel this memory is
    access_count: int = 0
    last_accessed: float = 0.0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    embedding: Optional[List[float]] = None


@dataclass
class KnowledgeTriplet:
    """A fact in the knowledge graph."""
    subject: str
    predicate: str
    obj: str
    confidence: float  # [0,1]
    timestamp: float
    source_memory_id: Optional[str] = None


class HMEM:
    """Hybrid Memory: Temporal Memory Tree + Knowledge Graph."""

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.nodes: Dict[str, MemoryNode] = {}
        self.knowledge_triplets: List[KnowledgeTriplet] = []
        self.root_id: Optional[str] = None
        self._persist_dir = os.path.join(
            settings.project_root, ".cogniteam", "hmem", namespace
        )
        self._consolidation_lock = False

    # ── Temporal Memory Tree Operations ────────────────────────────────

    def add_memory(
        self,
        content: str,
        entities: Optional[List[str]] = None,
        importance: float = 0.5,
    ) -> str:
        """Add a new memory at level 0 (session)."""
        now = time.time()
        node_id = f"mem_{now}_{hash(content) % 10000}"

        node = MemoryNode(
            id=node_id,
            content=content,
            timestamp=now,
            level=0,
            importance=min(1.0, max(0.0, importance)),
            entities=entities or [],
        )

        # Attach to parent at current highest level
        if self.root_id is None:
            self.root_id = node_id
        else:
            # Find the most recent level-0 node as parent
            recent_l0 = self._find_most_recent(level=0)
            if recent_l0:
                node.parent_id = recent_l0.id
                recent_l0.children_ids.append(node_id)
            else:
                node.parent_id = self.root_id

        self.nodes[node_id] = node

        # Extract and store entities as knowledge triplets
        if entities:
            for entity in entities:
                self.knowledge_triplets.append(
                    KnowledgeTriplet(
                        subject=entity,
                        predicate="mentioned_in",
                        obj=content[:100],
                        confidence=0.5,
                        timestamp=now,
                        source_memory_id=node_id,
                    )
                )

        return node_id

    def _find_most_recent(self, level: int) -> Optional[MemoryNode]:
        candidates = [
            n for n in self.nodes.values() if n.level == level
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda n: n.timestamp)

    def get_temporal_context(
        self, max_nodes: int = 20
    ) -> List[MemoryNode]:
        """Retrieve the most recent/important memories across levels."""
        # From root, traverse children collecting nodes weighted by
        # recency and importance
        if not self.root_id or self.root_id not in self.nodes:
            return []

        scored: List[Tuple[float, MemoryNode]] = []
        now = time.time()

        def traverse(node_id: str, depth: int = 0):
            if node_id not in self.nodes:
                return
            node = self.nodes[node_id]
            if node.level < 0:
                return
            recency = math.exp(-(now - node.timestamp) / 86400)
            level_bonus = 1.0 / (node.level + 1)
            score = (
                node.importance * 0.4
                + recency * 0.3
                + min(node.access_count, 10) / 10.0 * 0.2
                + level_bonus * 0.1
            )
            scored.append((score, node))

            for child_id in node.children_ids[:5]:
                traverse(child_id, depth + 1)

        traverse(self.root_id)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:max_nodes]]

    # ── Knowledge Graph Operations ─────────────────────────────────────

    def add_knowledge(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.8,
    ):
        self.knowledge_triplets.append(
            KnowledgeTriplet(
                subject=subject,
                predicate=predicate,
                obj=obj,
                confidence=min(1.0, max(0.0, confidence)),
                timestamp=time.time(),
            )
        )

    def query_knowledge(
        self,
        entity: Optional[str] = None,
        predicate: Optional[str] = None,
        min_confidence: float = 0.0,
        top_k: int = 10,
    ) -> List[KnowledgeTriplet]:
        """Query knowledge triplets by entity and/or predicate."""
        results = []
        for t in self.knowledge_triplets:
            if t.confidence < min_confidence:
                continue
            if entity and entity.lower() not in (
                t.subject.lower(),
                t.obj.lower(),
            ):
                continue
            if predicate and predicate.lower() != t.predicate.lower():
                continue
            results.append(t)
        return results[:top_k]

    def get_entity_graph(
        self, seed_entity: str, max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """Get subgraph around a seed entity."""
        results = []
        visited_entities: Set[str] = {seed_entity.lower()}
        current_level = [(seed_entity, 0)]

        while current_level:
            entity, depth = current_level.pop(0)
            for t in self.knowledge_triplets:
                subj_lower = t.subject.lower()
                obj_lower = t.obj.lower()

                if subj_lower == entity.lower():
                    results.append(
                        {
                            "subject": t.subject,
                            "predicate": t.predicate,
                            "obj": t.obj,
                            "confidence": t.confidence,
                            "distance": depth,
                        }
                    )
                    if obj_lower not in visited_entities and depth < max_hops:
                        visited_entities.add(obj_lower)
                        current_level.append((t.obj, depth + 1))

                elif obj_lower == entity.lower():
                    results.append(
                        {
                            "subject": t.subject,
                            "predicate": t.predicate,
                            "obj": t.obj,
                            "confidence": t.confidence,
                            "distance": depth,
                        }
                    )
                    if subj_lower not in visited_entities and depth < max_hops:
                        visited_entities.add(subj_lower)
                        current_level.append((t.subject, depth + 1))

        return results

    # ── Memory Consolidation ───────────────────────────────────────────

    def consolidate(self):
        """Consolidate recent memories into higher-level abstractions."""
        if self._consolidation_lock:
            return
        self._consolidation_lock = True

        try:
            now = time.time()
            # Group level-0 nodes by time windows
            l0_nodes = [
                n for n in self.nodes.values() if n.level == 0
            ]

            # Level 0 -> 1: group by 10 min windows
            windows: Dict[str, List[MemoryNode]] = defaultdict(list)
            for node in l0_nodes:
                window_key = datetime.fromtimestamp(
                    node.timestamp
                ).strftime("%Y-%m-%d_%H:%M")
                minute = int(
                    (datetime.fromtimestamp(node.timestamp).minute) / 10
                )
                window_key = f"{window_key[:14]}_{minute*10:02d}"
                windows[window_key].append(node)

            for window_key, group in windows.items():
                if len(group) < 2:
                    continue
                # Check if already consolidated
                existing = [
                    n
                    for n in self.nodes.values()
                    if n.level == 1 and n.parent_id
                    and any(
                        c == n.id
                        for c in self.nodes[n.parent_id].children_ids
                    )
                ]
                group_ids = {n.id for n in group}
                already_has_summary = any(
                    n.parent_id in group_ids for n in existing
                )
                if already_has_summary:
                    continue

                combined = " | ".join(n.content for n in group)
                avg_importance = sum(n.importance for n in group) / len(
                    group
                )
                entities = list(
                    set(
                        e
                        for n in group
                        for e in (n.entities or [])
                    )
                )

                summary = self._summarize(combined)
                node_id = f"cons_{window_key}_{int(now)}"
                summary_node = MemoryNode(
                    id=node_id,
                    content=summary or combined[:500],
                    timestamp=now,
                    level=1,
                    importance=avg_importance * 1.1,
                    entities=entities,
                    summary=summary,
                )
                for gn in group:
                    gn.level = -1  # Mark as archived
                    summary_node.children_ids.append(gn.id)
                    gn.parent_id = node_id
                self.nodes[node_id] = summary_node

        finally:
            self._consolidation_lock = False

    def _summarize(self, text: str) -> Optional[str]:
        try:
            from cogniteam.tools.utils.llm import llm_complete

            prompt = (
                "Summarize the following conversation/notes concisely.\n\n"
                f"{text[:4000]}\n\nConcise summary:"
            )
            return llm_complete(prompt, task="fast", max_tokens=512, timeout_seconds=60)
        except Exception:
            pass
        return None

    # ── Hybrid Retrieval ───────────────────────────────────────────────

    def hybrid_retrieve(
        self, query: str, top_k_temporal: int = 10, top_k_knowledge: int = 10
    ) -> Dict[str, Any]:
        """Retrieve from both TMT and KG, merge and rank results."""
        # Temporal retrieval: get recent/important context
        temporal = self.get_temporal_context(max_nodes=top_k_temporal)

        # Knowledge retrieval: find matching entities
        query_lower = query.lower()
        matched_entities: Set[str] = set()
        for t in self.knowledge_triplets:
            subj = (t.subject or "").lower()
            obj = (t.obj or "").lower()
            if query_lower in subj or query_lower in obj:
                matched_entities.add(t.subject)
                matched_entities.add(t.obj)

        knowledge_results = []
        for entity in matched_entities:
            subgraph = self.get_entity_graph(entity, max_hops=1)
            knowledge_results.extend(subgraph)

        # Score and rank combined results
        all_content = []
        for node in temporal:
            all_content.append(
                {
                    "type": "temporal",
                    "content": node.content,
                    "level": node.level,
                    "importance": node.importance,
                    "timestamp": node.timestamp,
                }
            )
        for kr in knowledge_results:
            all_content.append(
                {
                    "type": "knowledge",
                    "content": f"{kr['subject']} {kr['predicate']} {kr['obj']}",
                    "confidence": kr["confidence"],
                }
            )

        # Sort by recency/importance for temporal, confidence for knowledge
        def sort_key(item):
            if item["type"] == "temporal":
                return item.get("importance", 0) * 0.6 + 0.4 * (
                    1.0
                    - math.exp(
                        -(time.time() - item.get("timestamp", 0)) / 86400
                    )
                )
            return item.get("confidence", 0)

        all_content.sort(key=sort_key, reverse=True)

        return {
            "query": query,
            "results": all_content[:20],
            "matched_entities": list(matched_entities),
            "temporal_nodes_count": len(temporal),
            "knowledge_triplets_count": len(knowledge_results),
        }

    def answer_with_memory(self, query: str) -> str:
        """Answer a query using hybrid retrieval + LLM."""
        context = self.hybrid_retrieve(query)
        combined = "\n".join(
            r["content"] for r in context["results"][:10]
        )

        if not combined:
            return "No relevant context found."

        try:
            from cogniteam.tools.utils.llm import llm_complete

            prompt = (
                f"Answer the query based on the provided memory context.\n"
                f"If the context doesn't contain relevant info, say so.\n\n"
                f"Query: {query}\n\n"
                f"Memory Context:\n{combined[:5000]}\n\n"
                f"Answer:"
            )
            result = llm_complete(prompt, task="fast", max_tokens=1024, timeout_seconds=60)
            if result:
                return result
        except Exception as e:
            print(f"  H-MEM answer error: {e}")
        return combined[:500]

    # ── Persistence ────────────────────────────────────────────────────

    def save(self):
        os.makedirs(self._persist_dir, exist_ok=True)

        nodes_data = {
            nid: asdict(node) for nid, node in self.nodes.items()
        }
        triplets_data = [asdict(t) for t in self.knowledge_triplets]

        with open(os.path.join(self._persist_dir, "nodes.json"), "w") as f:
            json.dump(nodes_data, f, indent=2, default=str)
        with open(
            os.path.join(self._persist_dir, "triplets.json"), "w"
        ) as f:
            json.dump(triplets_data, f, indent=2, default=str)
        with open(
            os.path.join(self._persist_dir, "root.txt"), "w"
        ) as f:
            f.write(self.root_id or "")

    def load(self):
        try:
            with open(os.path.join(self._persist_dir, "nodes.json")) as f:
                nodes_data = json.load(f)
                for nid, ndata in nodes_data.items():
                    self.nodes[nid] = MemoryNode(**ndata)
            with open(
                os.path.join(self._persist_dir, "triplets.json")
            ) as f:
                triplets_data = json.load(f)
                self.knowledge_triplets = [
                    KnowledgeTriplet(**t) for t in triplets_data
                ]
            with open(os.path.join(self._persist_dir, "root.txt")) as f:
                self.root_id = f.read().strip() or None
        except (FileNotFoundError, json.JSONDecodeError):
            pass


_instances: Dict[str, HMEM] = {}


def get_hmem(namespace: str = "default") -> HMEM:
    if namespace not in _instances:
        _instances[namespace] = HMEM(namespace)
        _instances[namespace].load()
    return _instances[namespace]
