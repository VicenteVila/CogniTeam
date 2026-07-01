#!/usr/bin/env python3
"""
End-to-end memory test: exercises all 5 memory modules without LLM.
Creates isolated temp directories so persistence is test-only.
"""
import json
import os
import sys
import tempfile
import time

# Force isolated persist dirs before importing modules
_test_tmp = tempfile.mkdtemp(prefix="cogniteam_memory_test_")
print(f"Test dir: {_test_tmp}")

from cogniteam.config.settings import settings as _cfg
_orig_root = _cfg.project_root
_cfg.project_root = _test_tmp

passed = 0
failed = 0


def check(description: str, condition: bool):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {description}")
    else:
        failed += 1
        print(f"  ❌ {description}")


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── 1. H-MEM ─────────────────────────────────────────────────────────
section("H-MEM (Temporal Memory Tree + Knowledge Graph)")
from cogniteam.memory.hmem import get_hmem
hmem = get_hmem("test_e2e")
hmem._persist_dir = os.path.join(_test_tmp, "hmem_e2e")
hmem.load()

node1 = hmem.add_memory("Alice likes Python for data science", entities=["Alice", "Python"], importance=0.8)
node2 = hmem.add_memory("Bob prefers JavaScript for web dev", entities=["Bob", "JavaScript"], importance=0.6)
node3 = hmem.add_memory("Charlie uses Django for backend", entities=["Charlie", "Django"], importance=0.7)
check("3 level-0 memory nodes created", len([n for n in hmem.nodes.values() if n.level == 0]) == 3)

hmem.add_knowledge("Python", "is_a", "programming language", 0.95)
hmem.add_knowledge("Django", "is_a", "web framework", 0.9)
hmem.add_knowledge("Django", "built_with", "Python", 0.85)
check("Knowledge triplets added", len(hmem.knowledge_triplets) >= 9)

ctx = hmem.get_temporal_context(max_nodes=10)
check("Temporal context returns memories", len(ctx) >= 3)

kg = hmem.get_entity_graph("Django", max_hops=1)
check("Entity graph has connections", len(kg) >= 1)

hybrid = hmem.hybrid_retrieve("Python", top_k_temporal=5, top_k_knowledge=5)
check("Hybrid retrieve has results", len(hybrid["results"]) > 0)
check("Hybrid retrieve matched entities", "Python" in hybrid.get("matched_entities", []))

hmem.consolidate()
level1 = [n for n in hmem.nodes.values() if n.level == 1]
check("Consolidation created level-1 nodes", len(level1) >= 1)

hmem.save()
check("H-MEM saved to disk", os.path.exists(os.path.join(hmem._persist_dir, "nodes.json")))

# ── 2. GraphRAG ──────────────────────────────────────────────────────
section("GraphRAG (Knowledge Graph + Community Detection)")
from cogniteam.memory.graphrag import get_graphrag
graphrag = get_graphrag("test_e2e")
graphrag._persist_dir = os.path.join(_test_tmp, "graphrag_e2e")
graphrag.load()

texts = [
    "Alice works at Microsoft and develops AI tools with Python.",
    "Bob works at Google and builds web apps with JavaScript and Angular.",
    "Charlie from Microsoft collaborates with Alice on AI projects.",
]
for i, t in enumerate(texts):
    graphrag.extract_entities_from_text(t, f"chunk_{i}")
check("Entities extracted into graph", graphrag.graph.number_of_nodes() >= 3)

comms = graphrag.detect_communities()
check("Communities detected", len(comms) >= 1)

summaries = graphrag.summarize_communities()
check("Community summaries generated", len(summaries) >= 1)

global_results = graphrag.global_search("Microsoft AI", top_k=3)
check("Global search returns results", len(global_results) >= 1)

local_results = graphrag.local_search("Python", max_hops=2)
check("Local search returns results", len(local_results) >= 1)

hybrid = graphrag.hybrid_search("Microsoft", top_k_global=2, top_k_local=3)
check("Hybrid search has global_communities", len(hybrid["global_communities"]) > 0)
check("Hybrid search has local_entities", len(hybrid["local_entities"]) > 0)

graphrag.save()
check("GraphRAG saved to disk", os.path.exists(os.path.join(graphrag._persist_dir, "graph.json")))

# ── 3. MATM ──────────────────────────────────────────────────────────
section("MATM (Multi-Agent Transactive Memory)")
from cogniteam.memory.matm import get_matm
matm = get_matm("test_e2e")
matm._persist_dir = os.path.join(_test_tmp, "matm_e2e")
matm.load()

matm.register_agent("AliceAgent", topics=["Python", "AI", "Data"])
matm.register_agent("BobAgent", topics=["JavaScript", "Frontend"])
matm.register_agent("CharlieAgent", topics=["Django", "Backend"])
check("3 agents registered", len(matm.agent_expertise) == 3)

matm.declare_expertise("AliceAgent", "BobAgent", "Python", keywords=["data", "ml"])
matm.declare_expertise("CharlieAgent", "BobAgent", "Django", keywords=["backend", "python"])
check("Expertise declared in directory", len(matm.expertise_directory) > 0)

m1 = matm.store_memory("AliceAgent", "Python is great for ML", "Python", tags=["ml", "data"], importance=0.9, shared_with=["BobAgent"])
m2 = matm.store_memory("CharlieAgent", "Django is for web", "Django", tags=["web", "backend"], importance=0.8, shared_with=["AliceAgent"])
check("Memories stored", len(matm.memories) == 2)

results = matm.retrieve_memories("AliceAgent", "Python", top_k=5)
check("Memory retrieval works", len(results) >= 1)

matm.update_expertise_performance("BobAgent", "AliceAgent", "Python", success=True)
matm.update_expertise_performance("BobAgent", "AliceAgent", "Python", success=True)
matm.update_expertise_performance("BobAgent", "AliceAgent", "Python", success=False)
entry = matm.expertise_directory["BobAgent"][0]
check(f"EMA performance tracked (n={entry.interaction_count})", entry.interaction_count == 3)
check("EMA performance in [0,1]", 0 <= entry.performance <= 1)

retrieve = matm.transactive_retrieve("BobAgent", "Python")
check("Transactive retrieve works", "method" in retrieve)

synthesis = matm.synthesize_knowledge("BobAgent", "Django")
check("Knowledge synthesis returns text", len(synthesis) > 0)

matm.save()
check("MATM saved to disk", os.path.exists(os.path.join(matm._persist_dir, "matm.json")))

# ── 4. Fast-Slow ─────────────────────────────────────────────────────
section("Fast-Slow (GEPA + RL)")
from cogniteam.memory.fastslow import get_fastslow
fs = get_fastslow("test_e2e")
fs._persist_dir = os.path.join(_test_tmp, "fastslow_e2e")
fs.load()

fs.population_size = 4
fs.slow_episodes = 2
fs.fast_episodes = 3
fs.initialize_population("You are a helpful coding assistant that writes clean Python code.")

import random
random.seed(42)
for p in fs.population:
    p.trials = 3
    p.success_rate = random.random()
    p.avg_reward = random.random()

best = fs.run_slow_phase()
check(f"GEPA generation {fs.current_generation} produced best policy", best is not None)
check(f"Population maintained ({len(fs.population)} policies)", len(fs.population) == 4)

fs.run_fast_phase("write_code", reward=0.9, state="code_review")
fs.run_fast_phase("write_code", reward=0.7, state="code_review")
fs.run_fast_phase("debug", reward=0.3, state="code_review")
check("RL experiences stored", len(fs.experiences) == 3)
best_action = fs.get_best_action("code_review")
check("Best action learned", best_action == "write_code")

result = fs.learn(lambda a: 0.8, num_cycles=1, verbose=False)
check("Learn cycle completed", result["cycles"] == 1)

fs.save()
check("Fast-Slow saved to disk", os.path.exists(os.path.join(fs._persist_dir, "fastslow.json")))

# ── 5. Skills + SAGE ─────────────────────────────────────────────────
section("Skills (NanoResearch + SAGE)")
from cogniteam.memory.skills import get_skills
skills = get_skills("test_e2e")
skills._persist_dir = os.path.join(_test_tmp, "skills_e2e")
skills.load()

s1 = skills.register_skill("CodeReview", "Review Python code for PEP8 compliance", category="development")
s2 = skills.register_skill("DebugHelper", "Find and fix bugs in Python code", category="development")
s3 = skills.register_skill("UIGenerator", "Generate HTML/CSS/JS interfaces", category="design")
check("3 skills registered", len(skills.skills) == 3)

skills.record_skill_usage(s1, success=True, score=0.9)
skills.record_skill_usage(s1, success=True, score=0.8)
skills.record_skill_usage(s2, success=False, score=0.3)
check("Skill usage recorded", skills.skills[s1].usage_count == 2)
check("Skill success_rate updated", skills.skills[s1].success_rate == 1.0)
check("Skill with failures has rate < 1", skills.skills[s2].success_rate == 0.0)

skills.add_preference(s1, s2, preferred=s1, context="Python code review")
check("SDPO preference stored", len(skills.preferences) == 1)
check("Preferred skill has higher avg_score", skills.skills[s1].avg_score > skills.skills[s2].avg_score)

preferred = skills.get_preferred_skill("review Python code", [s1, s2])
check("SDPO selects preferred skill", preferred == s1)

n1 = skills.sage_write("PEP8 is a Python code style guide for formatting", node_type="fact")
n2 = skills.sage_write("Black is another Python code auto formatter tool", node_type="fact")
n3 = skills.sage_write("CSS is used for styling web pages with colors", node_type="fact")
check("SAGE nodes created", len(skills.nodes) >= 3)

check("SAGE connected related nodes", len(skills.edges) > 0)

read_results = skills.sage_read("Python", top_k=5)
check("SAGE read returns results", len(read_results) >= 1)
check("SAGE read found Python-related content",
      any("Python" in r["content"] for r in read_results))

retrieved = skills.retrieve_with_memory("Python")
check("retrieve_with_memory returns query", "query" in retrieved)
check("retrieve_with_memory has graph_memory", len(retrieved.get("graph_memory", [])) > 0)
check("retrieve_with_memory has relevant_skills", len(retrieved.get("relevant_skills", [])) > 0)

def eval_fn(skill_id):
    return 0.9 if skill_id == s1 else 0.4

evo = skills.co_evolve(eval_fn, num_rounds=2, verbose=False)
check("Co-evolution completed", evo["rounds"] == 2)

skills.save()
check("Skills saved to disk", os.path.exists(os.path.join(skills._persist_dir, "skills.json")))

# ── 6. After save: verify persistence ────────────────────────────────
section("Persistence Verification")
from cogniteam.memory.hmem import HMEM
hmem2 = HMEM("test_e2e")
hmem2._persist_dir = hmem._persist_dir
hmem2.load()
check("H-MEM loaded: nodes", len(hmem2.nodes) >= 3)
check("H-MEM loaded: triplets", len(hmem2.knowledge_triplets) >= 3)

from cogniteam.memory.graphrag import GraphRAGMemory
graphrag2 = GraphRAGMemory("test_e2e")
graphrag2._persist_dir = graphrag._persist_dir
graphrag2.load()
check("GraphRAG loaded: graph nodes", graphrag2.graph.number_of_nodes() >= 3)
check("GraphRAG loaded: communities", len(graphrag2.communities) >= 1)

from cogniteam.memory.matm import MATM
matm2 = MATM("test_e2e")
matm2._persist_dir = matm._persist_dir
matm2.load()
check("MATM loaded: memories", len(matm2.memories) >= 2)
check("MATM loaded: expertise", len(matm2.expertise_directory) > 0)

from cogniteam.memory.fastslow import FastSlowLearner
fs2 = FastSlowLearner("test_e2e")
fs2._persist_dir = fs._persist_dir
fs2.load()
check("Fast-Slow loaded: population", len(fs2.population) >= 4)
check("Fast-Slow loaded: experiences", len(fs2.experiences) >= 3)

from cogniteam.memory.skills import SkillPolicyManager
skills2 = SkillPolicyManager("test_e2e")
skills2._persist_dir = skills._persist_dir
skills2.load()
check("Skills loaded: skills", len(skills2.skills) >= 3)
check("Skills loaded: graph nodes", len(skills2.nodes) >= 3)
check("Skills loaded: preferences", len(skills2.preferences) >= 1)

# ── Summary ──────────────────────────────────────────────────────────
_cfg.project_root = _orig_root
print(f"\n{'='*60}")
print(f"  MEMORY E2E TEST RESULTS")
print(f"  Passed: {passed} | Failed: {failed} | Total: {passed + failed}")
print(f"{'='*60}")
sys.exit(0 if failed == 0 else 1)
