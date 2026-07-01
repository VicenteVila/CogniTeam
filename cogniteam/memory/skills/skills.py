"""
Skills & Policy Module: NanoResearch + SAGE integration.

References:
- NanoResearch (Shanghai AI Lab): Skill bank + SDPO (Skill Direct Preference Optimization)
- SAGE (Peking): Self-Evolving Agentic Graph-Memory (Writer-Reader architecture)

Key algorithms:
1. Skill Bank: Repository of reusable agent skills/behaviors
2. SDPO: Learn skill preferences from pairwise comparisons
3. Co-evolution: Multiple skills evolve simultaneously
4. SAGE Writer: Updates graph memory with new knowledge
5. SAGE Reader: Queries graph memory for relevant context
6. Self-Evolution: Graph structure + skill policies improve over time
"""

import json
import math
import os
import random
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx

from cogniteam.config.settings import settings


# ── Skill Data Structures ──────────────────────────────────────────────

@dataclass
class Skill:
    """A reusable capability/skill."""
    id: str
    name: str
    description: str
    code_template: Optional[str] = None
    prompt_template: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    category: str = "general"
    version: int = 1
    usage_count: int = 0
    success_rate: float = 0.0
    avg_score: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class SkillPreference:
    """A pairwise preference for SDPO."""
    skill_a_id: str
    skill_b_id: str
    preferred: str  # which skill was preferred (a_id or b_id)
    context: str
    timestamp: float
    score_a: float = 0.0
    score_b: float = 0.0


# ── SAGE Data Structures ──────────────────────────────────────────────

@dataclass
class GraphMemoryNode:
    """A node in the SAGE graph memory."""
    id: str
    content: str
    node_type: str  # "concept", "fact", "skill", "experience", "observation"
    confidence: float = 0.5
    importance: float = 0.5
    timestamp: float = 0.0
    access_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphMemoryEdge:
    """An edge in the SAGE graph memory."""
    source: str
    target: str
    relation: str  # "supports", "contradicts", "extends", "generalizes", "specializes", "related"
    weight: float = 1.0
    timestamp: float = 0.0


class SkillPolicyManager:
    """Combined Skills (NanoResearch) and SAGE (Self-Evolving Graph Memory)."""

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace

        # Skills
        self.skills: Dict[str, Skill] = {}
        self.preferences: List[SkillPreference] = []

        # SAGE Graph
        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes: Dict[str, GraphMemoryNode] = {}
        self.edges: List[GraphMemoryEdge] = []

        self._persist_dir = os.path.join(
            settings.project_root, ".cogniteam", "skills", namespace
        )

    # ── Skill Bank ─────────────────────────────────────────────────────

    def register_skill(
        self,
        name: str,
        description: str,
        category: str = "general",
        code_template: Optional[str] = None,
        prompt_template: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        prerequisites: Optional[List[str]] = None,
    ) -> str:
        """Register a new skill."""
        skill_id = f"skill_{name.lower().replace(' ', '_')}_{int(time.time())}"
        skill = Skill(
            id=skill_id,
            name=name,
            description=description,
            code_template=code_template,
            prompt_template=prompt_template,
            parameters=parameters or {},
            prerequisites=prerequisites or [],
            category=category,
            created_at=time.time(),
            updated_at=time.time(),
        )
        self.skills[skill_id] = skill

        # Also add to SAGE graph as a skill node
        self._add_sage_node(
            content=f"Skill: {name} - {description}",
            node_type="skill",
            metadata={"skill_id": skill_id, "category": category},
        )

        return skill_id

    def get_skills_by_category(self, category: str) -> List[Skill]:
        return [s for s in self.skills.values() if s.category == category]

    def get_top_skills(self, top_k: int = 10) -> List[Skill]:
        scored = sorted(
            self.skills.values(),
            key=lambda s: s.success_rate * 0.5 + s.avg_score * 0.3 + s.usage_count * 0.2,
            reverse=True,
        )
        return scored[:top_k]

    def record_skill_usage(
        self, skill_id: str, success: bool, score: float = 0.5
    ):
        """Update skill metrics after usage."""
        if skill_id not in self.skills:
            return
        skill = self.skills[skill_id]
        skill.usage_count += 1
        n = skill.usage_count
        skill.success_rate = skill.success_rate + (1.0 / n) * (
            (1.0 if success else 0.0) - skill.success_rate
        )
        skill.avg_score = skill.avg_score + (1.0 / n) * (score - skill.avg_score)
        skill.updated_at = time.time()

    # ── SDPO: Skill Direct Preference Optimization ─────────────────────

    def add_preference(
        self,
        skill_a_id: str,
        skill_b_id: str,
        preferred: str,
        context: str,
    ):
        """Record a pairwise skill preference."""
        pref = SkillPreference(
            skill_a_id=skill_a_id,
            skill_b_id=skill_b_id,
            preferred=preferred,
            context=context,
            timestamp=time.time(),
        )
        self.preferences.append(pref)

        # Update skill scores based on preference
        if preferred == skill_a_id:
            self._update_skill_score(skill_a_id, +0.05)
            self._update_skill_score(skill_b_id, -0.02)
        else:
            self._update_skill_score(skill_b_id, +0.05)
            self._update_skill_score(skill_a_id, -0.02)

    def _update_skill_score(self, skill_id: str, delta: float):
        if skill_id in self.skills:
            self.skills[skill_id].avg_score = max(
                0.0, min(1.0, self.skills[skill_id].avg_score + delta)
            )

    def get_preferred_skill(
        self, context: str, candidates: List[str]
    ) -> Optional[str]:
        """Given candidates and context, find the preferred skill using SDPO."""
        context_lower = context.lower()

        # Aggregate preferences involving these candidates
        pref_scores: Dict[str, float] = defaultdict(float)
        for p in self.preferences:
            if p.skill_a_id in candidates or p.skill_b_id in candidates:
                if p.preferred == p.skill_a_id:
                    pref_scores[p.skill_a_id] += 0.1
                else:
                    pref_scores[p.skill_b_id] += 0.1

        # Add base scores
        for c in candidates:
            if c in self.skills:
                skill = self.skills[c]
                pref_scores[c] += skill.avg_score * 0.3 + skill.success_rate * 0.3

        if not pref_scores:
            return candidates[0] if candidates else None

        return max(pref_scores, key=pref_scores.get)

    # ── SAGE: Self-Evolving Graph Memory ───────────────────────────────

    def _add_sage_node(
        self,
        content: str,
        node_type: str = "observation",
        confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        node_id = f"sage_{node_type}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        node = GraphMemoryNode(
            id=node_id,
            content=content,
            node_type=node_type,
            confidence=confidence,
            importance=0.5,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self.nodes[node_id] = node
        self.graph.add_node(
            node_id,
            type=node_type,
            content=content[:100],
        )
        return node_id

    def _add_sage_edge(
        self,
        source: str,
        target: str,
        relation: str = "related",
        weight: float = 1.0,
    ):
        if source not in self.graph or target not in self.graph:
            return
        edge = GraphMemoryEdge(
            source=source,
            target=target,
            relation=relation,
            weight=weight,
            timestamp=time.time(),
        )
        self.edges.append(edge)
        self.graph.add_edge(source, target, relation=relation, weight=weight)

    def sage_write(self, content: str, node_type: str = "observation"):
        """SAGE Writer: add new observation to graph memory and connect to existing nodes."""
        node_id = self._add_sage_node(content, node_type)

        # Find and connect to related existing nodes
        content_lower = content.lower()
        content_tokens = set(content_lower.split())

        for existing_id, existing_node in self.nodes.items():
            if existing_id == node_id:
                continue

            existing_lower = existing_node.content.lower()
            overlap = len(content_tokens & set(existing_lower.split()))
            if overlap > 2:
                relation = "related"
                if overlap > 5:
                    relation = "supports"
                self._add_sage_edge(
                    node_id, existing_id, relation=relation, weight=overlap / 10.0
                )

        # Consolidate: if too many nodes, prune least important
        if len(self.nodes) > 1000:
            self._sage_consolidate()

        return node_id

    def sage_read(
        self, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """SAGE Reader: query graph memory for relevant context."""
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        # Score nodes by content overlap and graph centrality
        if self.graph.number_of_nodes() == 0:
            return []
        centralities = nx.pagerank(self.graph, alpha=0.85)
        scored: List[tuple[float, GraphMemoryNode]] = []

        for node_id, node in self.nodes.items():
            content_lower = node.content.lower()
            token_overlap = len(query_tokens & set(content_lower.split())) / max(
                len(query_tokens), 1
            )
            centrality = centralities.get(node_id, 0.0)
            score = (
                token_overlap * 0.5
                + centrality * 0.2
                + node.importance * 0.2
                + node.confidence * 0.1
            )
            scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, node in scored[:top_k]:
            # Get connected context
            neighbors = []
            for neighbor_id in self.graph.neighbors(node.id):
                if neighbor_id in self.nodes:
                    edge_data = self.graph.get_edge_data(node.id, neighbor_id)
                    neighbors.append(
                        {
                            "content": self.nodes[neighbor_id].content[:100],
                            "relation": edge_data.get("relation", "related") if edge_data else "related",
                        }
                    )

            results.append(
                {
                    "node_id": node.id,
                    "content": node.content,
                    "type": node.node_type,
                    "relevance_score": score,
                    "confidence": node.confidence,
                    "connections": neighbors[:5],
                }
            )
        return results

    def _sage_consolidate(self):
        """Remove low-importance, low-access nodes. Merge similar ones."""
        scores = []
        for node_id, node in self.nodes.items():
            degree = self.graph.degree(node_id)
            access_bonus = min(node.access_count / 10, 1)
            score = node.importance * 0.4 + node.confidence * 0.2 + degree * 0.2 + access_bonus * 0.2
            scores.append((score, node_id, node))

        scores.sort(key=lambda x: x[0])
        to_remove = set()
        for score, node_id, node in scores[: len(self.nodes) // 4]:
            to_remove.add(node_id)

        for node_id in to_remove:
            if node_id in self.nodes:
                del self.nodes[node_id]
                if node_id in self.graph:
                    self.graph.remove_node(node_id)

        self.edges = [
            e for e in self.edges if e.source not in to_remove and e.target not in to_remove
        ]

    # ── Co-evolution ───────────────────────────────────────────────────

    def co_evolve(
        self,
        task_eval_fn: Callable[[str], float],
        num_rounds: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Co-evolve skills and graph memory together."""
        history = []

        for round_num in range(num_rounds):
            # 1. Select top skills for this round
            top_skills = self.get_top_skills(top_k=3)

            if verbose:
                print(f"  Co-evolution round {round_num + 1}: {len(top_skills)} top skills")

            # 2. Evaluate skills on the task
            for skill in top_skills:
                score = task_eval_fn(skill.id)
                success = score > 0.5
                self.record_skill_usage(skill.id, success, score)

                # Record in SAGE graph
                self.sage_write(
                    f"Task evaluation with skill '{skill.name}': score={score:.3f}",
                    node_type="experience",
                )

            # 3. Generate new skills based on what's learned
            if round_num > 0 and random.random() < 0.3:
                self._generate_new_skill(top_skills)

            history.append(
                {
                    "round": round_num + 1,
                    "top_skills": [s.name for s in top_skills],
                    "skill_count": len(self.skills),
                    "graph_nodes": len(self.nodes),
                }
            )

        return {"rounds": num_rounds, "history": history}

    def _generate_new_skill(self, inspiration_skills: List[Skill]):
        """Generate a new skill by combining/evolving existing ones."""
        if not inspiration_skills:
            return

        try:
            from cogniteam.tools.utils.llm import llm_complete

            existing = "\n".join(
                f"- {s.name}: {s.description}" for s in inspiration_skills
            )
            prompt = (
                "Based on the following existing skills, propose a NEW skill "
                "that would be useful for an AI agent. Return a JSON object with "
                "'name' and 'description' keys.\n\n"
                f"Existing skills:\n{existing}\n\nNew skill (JSON):"
            )
            raw = llm_complete(prompt, task="fast", max_tokens=512, timeout_seconds=60)
            if raw:
                result = json.loads(raw)
                self.register_skill(
                    name=result.get("name", "new_skill"),
                    description=result.get("description", ""),
                    category="evolved",
                )
        except Exception:
            pass

    # ── SAGE-Enhanced Retrieval ────────────────────────────────────────

    def retrieve_with_memory(
        self, query: str
    ) -> Dict[str, Any]:
        """Retrieve relevant skills and graph context for a query."""
        # Get relevant graph context
        graph_context = self.sage_read(query, top_k=5)

        # Find matching skills
        query_lower = query.lower()
        matching_skills = []
        for skill in self.skills.values():
            if (
                query_lower in skill.name.lower()
                or query_lower in skill.description.lower()
                or any(q in skill.description.lower() for q in query_lower.split())
            ):
                matching_skills.append(skill)

        # Try SDPO to pick best skill
        preferred_skill_id = None
        if matching_skills:
            preferred_skill_id = self.get_preferred_skill(
                query, [s.id for s in matching_skills]
            )

        return {
            "query": query,
            "graph_memory": graph_context,
            "relevant_skills": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "success_rate": s.success_rate,
                }
                for s in matching_skills[:5]
            ],
            "preferred_skill": preferred_skill_id,
        }

    def answer_with_memory(self, query: str) -> str:
        """Answer a query using skills + graph memory + LLM synthesis."""
        context = self.retrieve_with_memory(query)
        graph_text = "\n".join(
            f"- {r['content']}" for r in context["graph_memory"]
        )
        skills_text = "\n".join(
            f"- {s['name']}: {s['description']}"
            for s in context["relevant_skills"]
        )
        pref_text = ""
        if context["preferred_skill"] and context["preferred_skill"] in self.skills:
            pref_skill = self.skills[context["preferred_skill"]]
            if pref_skill.prompt_template:
                pref_text = f"\nRecommended prompt template:\n{pref_skill.prompt_template}"

        try:
            from cogniteam.tools.utils.llm import llm_complete

            prompt = (
                f"Based on the memory context and available skills, answer the query.\n\n"
                f"Query: {query}\n\n"
                f"Graph Memory Context:\n{graph_text[:3000]}\n\n"
                f"Relevant Skills:\n{skills_text[:2000]}{pref_text}\n\n"
                f"Answer:"
            )
            result = llm_complete(prompt, task="fast", max_tokens=1024, timeout_seconds=60)
            if result:
                return result
        except Exception:
            pass

        combined = f"Memory context:\n{graph_text[:500]}\n\nSkills:\n{skills_text[:500]}"
        return combined

    # ── Persistence ────────────────────────────────────────────────────

    def save(self):
        os.makedirs(self._persist_dir, exist_ok=True)

        data = {
            "skills": {k: asdict(v) for k, v in self.skills.items()},
            "preferences": [asdict(p) for p in self.preferences],
            "graph_nodes": {k: asdict(v) for k, v in self.nodes.items()},
            "graph_edges": [asdict(e) for e in self.edges],
        }
        with open(os.path.join(self._persist_dir, "skills.json"), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self):
        try:
            with open(os.path.join(self._persist_dir, "skills.json")) as f:
                data = json.load(f)
                for sid, sdata in data.get("skills", {}).items():
                    self.skills[sid] = Skill(**sdata)
                self.preferences = [
                    SkillPreference(**p) for p in data.get("preferences", [])
                ]
                for nid, ndata in data.get("graph_nodes", {}).items():
                    self.nodes[nid] = GraphMemoryNode(**ndata)
                    self.graph.add_node(
                        nid,
                        type=ndata.get("node_type", "observation"),
                        content=ndata.get("content", "")[:100],
                    )
                for edata in data.get("graph_edges", []):
                    edge = GraphMemoryEdge(**edata)
                    self.edges.append(edge)
                    self.graph.add_edge(
                        edge.source, edge.target,
                        relation=edge.relation, weight=edge.weight,
                    )
        except (FileNotFoundError, json.JSONDecodeError):
            pass


_instances: Dict[str, SkillPolicyManager] = {}


def get_skills(namespace: str = "default") -> SkillPolicyManager:
    if namespace not in _instances:
        _instances[namespace] = SkillPolicyManager(namespace)
        _instances[namespace].load()
    return _instances[namespace]
