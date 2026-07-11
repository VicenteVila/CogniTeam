# Domain Consolidation: 13 → 5

## Rationale

The original CogniTeam pipeline shipped with 13 domains, each with a dedicated
world model prompt, few-shot example, and grounding keywords.  In practice, 8 of
those domains were never exercised end‑to‑end and added maintenance overhead
(~530 lines of dead prompts, 48 archetype entries in the fallback list).

We pruned to the **5 domains that have been validated** with real pipeline runs:

| # | Domain | Few‑Shot | World Model Prompt | Archetypes |
|---|--------|----------|-------------------|------------|
| 1 | Web Development | `FEWSHOT_UI` | `WEB_DEV_WORLD_MODEL_PROMPT` | landing‑page, ecommerce, saas‑dashboard, blog‑content, portfolio‑creative, corporate‑business |
| 2 | Software Development | `FEWSHOT_SCRIPT` | `SOFTWARE_DEV_WORLD_MODEL_PROMPT` | mobile‑apps, backend‑apis, databases, devops‑infra, machine‑learning |
| 3 | Education | `FEWSHOT_EDUCATION` | `EDUCATION_WORLD_MODEL_PROMPT` | bootcamps, moocs, certifications, corporate‑training, k12‑curriculum |
| 4 | Graphic Design | `FEWSHOT_GRAPHIC_DESIGN` | `GRAPHIC_DESIGN_WORLD_MODEL_PROMPT` | branding‑identity, editorial‑layout, ui‑ux‑prototyping |
| 8 | Game Development | `FEWSHOT_GAME_DEV` | `GAME_DEV_WORLD_MODEL_PROMPT` | mobile‑casual, pc‑console, multiplayer‑online, vr‑ar, game‑tools |

## Experimental Domains (generic fallback only)

The YAML archetype file (`cogniteam_archetypes.yaml`) still defines these 8
domains, and the pipeline can still process them — but they fall back to a
generic few‑shot (`FEWSHOT_SCRIPT` or `FEWSHOT_CONTENT`) and the generic world
model prompt, so the plan quality may be lower:

| # | Domain | Archetypes |
|---|--------|------------|
| 5 | Data Science | exploratory‑analysis, predictive‑modeling, data‑pipeline |
| 6 | Content Writing | seo‑article, technical‑doc, copywriting |
| 7 | DevOps | ci‑cd, infra‑as‑code, monitoring |
| 9 | Cybersecurity | vulnerability‑scan, security‑audit, incident‑response |
| 10 | Legal & Compliance | contract‑generation, privacy‑policy, terms‑of‑service |
| 11 | Architecture & Spatial | floor‑plan, 3d‑visualization, bim‑coordination |
| 12 | Marketing & Growth | campaign‑landing, email‑sequence, social‑content |
| 13 | Audiovisual & Management | video‑script, podcast‑plan, content‑calendar |

To promote an experimental domain to **supported**, you need:

1. A dedicated few‑shot example (`FEWSHOT_<DOMAIN>`) in `planner_agent.py`
2. A world model prompt (`<DOMAIN>_WORLD_MODEL_PROMPT`) in `planner_agent.py`
3. Corresponding entries in `DOMAIN_FEWSHOT_EXAMPLES` and `DOMAIN_WORLD_MODEL_PROMPTS`
4. Grounding keywords added to `GROUNDING_KEYWORDS_FALLBACK`
5. At least one end‑to‑end pipeline test run

## Validation

```bash
# All 71 Python files must compile
python3 -m py_compile cogniteam/**/*.py main.py

# Existing tests (105 pass, 2 expected to fail on version checks)
python3 -m pytest tests/ -v --tb=short

# Domain classification
python3 -c "
from cogniteam.scoping.loader import classify_task
import asyncio
result = asyncio.run(classify_task('build a space shooter game'))
print(f'Domain: {result[0]}, Confidence: {result[2]:.0%}')
"

# Blueprint JSON should contain only the 5 supported domains
python3 -c "
import json
with open('antigravity_complete_system_7_domains.json') as f:
    bp = json.load(f)
domains = set(bp.get('domains', bp.keys())) if isinstance(bp, dict) else set()
print(f'Blueprint domains: {len(domains)}')
"
```

## Files changed during consolidation

| File | Change |
|------|--------|
| `cogniteam/agents/planner_agent.py` | Removed 8 world model prompts (~530 lines); pruned `GROUNDING_KEYWORDS_FALLBACK` from 48→24; pruned `DOMAIN_FEWSHOT_EXAMPLES` 13→5; pruned `DOMAIN_WORLD_MODEL_PROMPTS` 13→5; added `FEWSHOT_EDUCATION`, `FEWSHOT_GRAPHIC_DESIGN`, `FEWSHOT_GAME_DEV` |
| `antigravity_complete_system_7_domains.json` | Filtered to only 5 domains |
| `tests/test_scoping.py` | Updated assertion thresholds (25 blueprints, 5 domains) |
