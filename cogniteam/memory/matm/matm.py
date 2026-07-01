"""
MATM: Multi-Agent Transactive Memory System.

Reference: "Multi-Agent Transactive Memory for Collaborative LLM Systems" (CMU)

Core idea: Each agent maintains a 'directory' of expertise for other agents,
enabling efficient knowledge sharing without redundancy.

Key algorithms:
1. Transactive Encoding: When agent A learns something, it stores a pointer
   to agent B's expertise (not the full content)
2. Transactive Storage: Knowledge is stored with metadata about which agent
   is best suited to retrieve it
3. Transactive Retrieval: Query first checks the directory to find which
   agent has relevant knowledge, then delegates
4. Expertise Updates: Directory entries are updated based on performance
5. Benefits: reduces redundancy, prevents hallucination, leverages expertise
"""

import json
import math
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

from cogniteam.config.settings import settings


@dataclass
class ExpertiseEntry:
    """Records that agent_a has expertise on topic for agent_b."""
    agent_a: str        # The agent who has the expertise
    agent_b: str        # The agent who stores this directory entry
    topic: str
    confidence: float   # [0,1] how confident agent_b is about agent_a's expertise
    performance: float  # [0,1] historical success rate when delegating to agent_a
    last_updated: float
    interaction_count: int = 0
    keywords: List[str] = field(default_factory=list)


@dataclass
class MemoryChunk:
    """A piece of knowledge stored by an agent."""
    id: str
    agent_id: str
    content: str
    topic: str
    tags: List[str]
    timestamp: float
    importance: float      # [0,1]
    access_count: int = 0
    shared_with: List[str] = field(default_factory=list)  # agents it was shared with
    source_chunks: List[str] = field(default_factory=list)  # transactive pointers


@dataclass
class DelegationRecord:
    """Records a delegation event for performance tracking."""
    delegator: str
    delegate: str
    topic: str
    query: str
    success: bool  # was the delegation result useful?
    timestamp: float


class MATM:
    """Multi-Agent Transactive Memory System."""

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.expertise_directory: Dict[str, List[ExpertiseEntry]] = defaultdict(list)
        self.memories: Dict[str, MemoryChunk] = {}
        self.delegations: List[DelegationRecord] = []
        self.agent_expertise: Dict[str, Set[str]] = defaultdict(set)
        self._persist_dir = os.path.join(
            settings.project_root, ".cogniteam", "matm", namespace
        )

    # ── Expertise Management ───────────────────────────────────────────

    def register_agent(self, agent_id: str, topics: Optional[List[str]] = None):
        """Register an agent with its areas of expertise."""
        if topics:
            self.agent_expertise[agent_id].update(topics)

    def declare_expertise(
        self,
        agent_a: str,
        agent_b: str,
        topic: str,
        keywords: Optional[List[str]] = None,
        initial_confidence: float = 0.5,
    ):
        """Agent_b records that agent_a has expertise on topic."""
        entries = self.expertise_directory[agent_b]
        for entry in entries:
            if entry.agent_a == agent_a and entry.topic == topic:
                entry.last_updated = time.time()
                entry.keywords = keywords or []
                return

        self.expertise_directory[agent_b].append(
            ExpertiseEntry(
                agent_a=agent_a,
                agent_b=agent_b,
                topic=topic,
                confidence=initial_confidence,
                performance=0.5,
                last_updated=time.time(),
                keywords=keywords or [],
            )
        )

    def update_expertise_performance(
        self,
        delegator: str,
        delegate: str,
        topic: str,
        success: bool,
    ):
        """Update performance metrics after a delegation."""
        self.delegations.append(
            DelegationRecord(
                delegator=delegator,
                delegate=delegate,
                topic=topic,
                query=topic,
                success=success,
                timestamp=time.time(),
            )
        )

        for entry in self.expertise_directory[delegator]:
            if entry.agent_a == delegate and entry.topic == topic:
                entry.interaction_count += 1
                n = entry.interaction_count
                old_perf = entry.performance
                # Exponential moving average
                entry.performance = old_perf + (1.0 / n) * (
                    (1.0 if success else 0.0) - old_perf
                )
                entry.confidence = min(
                    1.0, entry.confidence + (0.05 if success else -0.05)
                )
                entry.last_updated = time.time()
                break

    # ── Directory Query ────────────────────────────────────────────────

    def find_expert(
        self, agent_id: str, query: str, top_k: int = 3
    ) -> List[ExpertiseEntry]:
        """Find which agents to delegate to for a query."""
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        scored: List[tuple[float, ExpertiseEntry]] = []
        for entry in self.expertise_directory[agent_id]:
            # Score based on topic overlap + confidence + performance
            topic_score = (
                1.0
                if query_lower in entry.topic.lower()
                else 0.0
            )
            keyword_score = 0.0
            if entry.keywords:
                keyword_score = len(
                    query_tokens & set(k.lower() for k in entry.keywords)
                ) / max(len(query_tokens), 1)
            expertise_score = max(topic_score, keyword_score)

            combined = (
                expertise_score * 0.3
                + entry.confidence * 0.3
                + entry.performance * 0.4
            )
            scored.append((combined, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def get_all_experts_for_query(self, query: str) -> Dict[str, List[ExpertiseEntry]]:
        """Get experts across all agents' directories for a query."""
        result = {}
        for agent_id, entries in self.expertise_directory.items():
            experts = self.find_expert(agent_id, query, top_k=2)
            if experts:
                result[agent_id] = experts
        return result

    # ── Memory Storage and Retrieval ───────────────────────────────────

    def store_memory(
        self,
        agent_id: str,
        content: str,
        topic: str,
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
        shared_with: Optional[List[str]] = None,
    ) -> str:
        """Store a memory chunk from an agent."""
        now = time.time()
        chunk_id = f"matm_{agent_id}_{int(now)}_{hash(content) % 10000}"

        chunk = MemoryChunk(
            id=chunk_id,
            agent_id=agent_id,
            content=content,
            topic=topic,
            tags=tags or [],
            timestamp=now,
            importance=min(1.0, max(0.0, importance)),
            shared_with=shared_with or [],
        )
        self.memories[chunk_id] = chunk

        # Transactive encoding: if shared with other agents, record it
        if shared_with:
            for other_agent in shared_with:
                if other_agent != agent_id:
                    self.declare_expertise(
                        agent_a=agent_id,
                        agent_b=other_agent,
                        topic=topic,
                        keywords=tags,
                    )

        return chunk_id

    def retrieve_memories(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
    ) -> List[MemoryChunk]:
        """Retrieve memories relevant to a query for a specific agent."""
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        # Own memories
        own = [
            m
            for m in self.memories.values()
            if m.agent_id == agent_id
        ]

        # Shared memories (via transactive pointers)
        shared = [
            m
            for m in self.memories.values()
            if m.agent_id != agent_id and agent_id in m.shared_with
        ]

        candidates = own + shared

        scored: List[tuple[float, MemoryChunk]] = []
        for mem in candidates:
            topic_score = 1.0 if query_lower in mem.topic.lower() else 0.0
            tag_score = 0.0
            if mem.tags:
                tag_score = len(
                    query_tokens & set(t.lower() for t in mem.tags)
                ) / max(len(query_tokens), 1)
            content_overlap = len(
                query_tokens & set(mem.content.lower().split())
            ) / max(len(query_tokens), 1)

            score = (
                topic_score * 0.3
                + tag_score * 0.2
                + content_overlap * 0.2
                + mem.importance * 0.2
                + min(mem.access_count / 10, 1) * 0.1
            )
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [m for _, m in scored[:top_k]]
        for m in results:
            m.access_count += 1
        return results

    def transactive_retrieve(
        self, agent_id: str, query: str
    ) -> Dict[str, Any]:
        """Transactive retrieval: check directory, delegate if expert found."""
        experts = self.find_expert(agent_id, query, top_k=2)

        if experts:
            top_expert = experts[0]
            if top_expert.confidence > 0.6 and top_expert.performance > 0.5:
                return {
                    "method": "delegation",
                    "delegate": top_expert.agent_a,
                    "expertise": top_expert.topic,
                    "memories": self.retrieve_memories(
                        top_expert.agent_a, query
                    ),
                }

        # Fallback: retrieve from own memory + shared
        memories = self.retrieve_memories(agent_id, query)
        return {
            "method": "direct",
            "memories": memories,
        }

    # ── Knowledge Synthesis ────────────────────────────────────────────

    def synthesize_knowledge(
        self, agent_id: str, query: str
    ) -> str:
        """Synthesize knowledge from multiple agents' memories into a coherent response."""
        result = self.transactive_retrieve(agent_id, query)
        memories = result.get("memories", [])

        if not memories:
            return "No relevant memories found."

        combined = "\n".join(
            f"[{m.agent_id}] {m.content}" for m in memories[:5]
        )

        try:
            from cogniteam.tools.utils.llm import llm_complete

            delegate_info = ""
            if result["method"] == "delegation":
                delegate_info = f"(retrieved via {result['delegate']})"

            prompt = (
                f"Synthesize the following multi-agent memories into a coherent answer.\n"
                f"{delegate_info}\n\nQuery: {query}\n\n"
                f"Memories:\n{combined[:4000]}\n\nSynthesized answer:"
            )
            result_text = llm_complete(prompt, task="fast", max_tokens=1024, timeout_seconds=60)
            if result_text:
                return result_text
        except Exception:
            pass
        return combined[:500]

    # ── Persistence ────────────────────────────────────────────────────

    def save(self):
        os.makedirs(self._persist_dir, exist_ok=True)

        data = {
            "expertise": {
                k: [asdict(e) for e in v]
                for k, v in self.expertise_directory.items()
            },
            "memories": {
                k: asdict(v) for k, v in self.memories.items()
            },
            "delegations": [asdict(d) for d in self.delegations],
            "agent_expertise": {
                k: list(v) for k, v in self.agent_expertise.items()
            },
        }
        with open(os.path.join(self._persist_dir, "matm.json"), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self):
        try:
            with open(os.path.join(self._persist_dir, "matm.json")) as f:
                data = json.load(f)
                for agent_id, entries in data.get("expertise", {}).items():
                    self.expertise_directory[agent_id] = [
                        ExpertiseEntry(**e) for e in entries
                    ]
                for cid, mdata in data.get("memories", {}).items():
                    self.memories[cid] = MemoryChunk(**mdata)
                self.delegations = [
                    DelegationRecord(**d)
                    for d in data.get("delegations", [])
                ]
                self.agent_expertise = {
                    k: set(v) for k, v in data.get("agent_expertise", {}).items()
                }
        except (FileNotFoundError, json.JSONDecodeError):
            pass


_instances: Dict[str, MATM] = {}


def get_matm(namespace: str = "default") -> MATM:
    if namespace not in _instances:
        _instances[namespace] = MATM(namespace)
        _instances[namespace].load()
    return _instances[namespace]
