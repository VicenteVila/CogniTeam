"""
Fast-Slow Learning: Interleaved Fast (RL) and Slow (GEPA/Evolutionary) Learning.

Reference: "Learning to Learn Faster from Human Feedback with Language Models"
(Berkeley) and "Agents: An Open-source Framework for Autonomous Language Agents"

Key algorithms:
1. Slow Phase (GEPA - Genetic Evolutionary Policy Adaptation):
   - Population of agent policies (prompts/strategies)
   - Fitness evaluation based on task performance
   - Selection, crossover, mutation to evolve policies
   - Explores diverse strategies

2. Fast Phase (RL - Reinforcement Learning):
   - Given a policy from the slow phase, practice via RL
   - Reward signal from task success/failure
   - Policy gradient to fine-tune behavior
   - Exploits the best known strategy

3. Interleaved Schedule:
   - Slow explores for N episodes
   - Fast exploits for M episodes
   - Repeat with periodic evaluation
"""

import json
import math
import os
import random
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from cogniteam.config.settings import settings


@dataclass
class Policy:
    """A policy = system prompt + strategy description."""
    id: str
    system_prompt: str
    strategy_description: str
    parameters: Dict[str, Any]
    fitness: float = 0.0
    generation: int = 0
    trials: int = 0
    success_rate: float = 0.0
    avg_reward: float = 0.0


@dataclass
class Experience:
    """A single RL experience."""
    state: str
    action: str
    reward: float
    done: bool
    policy_id: str
    timestamp: float


class FastSlowLearner:
    """Interleaved Fast (RL) and Slow (GEPA) learning system."""

    def __init__(
        self,
        namespace: str = "default",
        population_size: int = 10,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.2,
        slow_episodes: int = 5,
        fast_episodes: int = 15,
    ):
        self.namespace = namespace
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.slow_episodes = slow_episodes
        self.fast_episodes = fast_episodes

        # State
        self.population: List[Policy] = []
        self.experiences: List[Experience] = []
        self.current_generation: int = 0
        self.phase: str = "slow"  # "slow" or "fast"
        self.phase_counter: int = 0
        self.best_policy: Optional[Policy] = None
        self._eval_fn: Optional[Callable] = None

        self._persist_dir = os.path.join(
            settings.project_root, ".cogniteam", "fastslow", namespace
        )

    # ── Initialization ─────────────────────────────────────────────────

    def initialize_population(self, base_prompt: str):
        """Create initial population by mutating the base prompt."""
        self.population = []
        for i in range(self.population_size):
            mutated = self._mutate_prompt(base_prompt, rate=0.3)
            policy = Policy(
                id=f"policy_{i}_{int(time.time())}",
                system_prompt=mutated,
                strategy_description=f"Initial variant {i}",
                parameters={"variant": i, "temperature": 0.7 + random.uniform(-0.2, 0.2)},
                generation=0,
            )
            self.population.append(policy)
        self.best_policy = self.population[0]

    # ── Slow Phase: GEPA ───────────────────────────────────────────────

    def run_slow_phase(self) -> Policy:
        """Run one generation of GEPA: evaluate, select, crossover, mutate."""
        if len(self.population) < 2:
            return self.population[0] if self.population else Policy(
                id="default", system_prompt="", strategy_description="", parameters={}
            )

        # Evaluate fitness
        for policy in self.population:
            if policy.trials > 0:
                policy.fitness = policy.success_rate * 0.6 + policy.avg_reward * 0.4
            else:
                policy.fitness = 0.0

        self.population.sort(key=lambda p: p.fitness, reverse=True)
        elite_count = max(2, int(self.population_size * self.elite_ratio))
        elites = self.population[:elite_count]

        # Update best
        if elites:
            self.best_policy = elites[0]

        # Create next generation
        next_gen: List[Policy] = list(elites)
        new_generation = self.current_generation + 1

        while len(next_gen) < self.population_size:
            if random.random() < self.crossover_rate and len(elites) >= 2:
                p1, p2 = random.sample(elites, 2)
                child = self._crossover(p1, p2, new_generation)
            else:
                parent = random.choice(elites)
                child = self._mutate_policy(parent, new_generation)

            next_gen.append(child)

        self.population = next_gen[:self.population_size]
        self.current_generation = new_generation

        return self.best_policy

    def _crossover(self, p1: Policy, p2: Policy, generation: int) -> Policy:
        """Combine two policies via crossover on prompt segments."""
        prompts1 = p1.system_prompt.split(". ")
        prompts2 = p2.system_prompt.split(". ")

        max_split = min(len(prompts1), len(prompts2))
        if max_split <= 1:
            child_prompt = (p1.system_prompt + " " + p2.system_prompt)[:500]
        else:
            split = random.randint(1, max_split - 1)
            child_prompt = ". ".join(prompts1[:split] + prompts2[split:])

        child_params = dict(p1.parameters)
        for key in child_params:
            if isinstance(child_params[key], (int, float)):
                p2_val = p2.parameters.get(key, child_params[key])
                child_params[key] = (
                    child_params[key] + p2_val
                ) / 2 + random.uniform(-0.1, 0.1)

        return Policy(
            id=f"policy_cross_{int(time.time())}_{random.randint(1000,9999)}",
            system_prompt=child_prompt,
            strategy_description=f"Crossover of {p1.id[:20]} and {p2.id[:20]}",
            parameters=child_params,
            generation=generation,
        )

    def _mutate_policy(self, parent: Policy, generation: int) -> Policy:
        """Mutate a policy by modifying its prompt and parameters."""
        new_prompt = self._mutate_prompt(parent.system_prompt, self.mutation_rate)
        new_params = dict(parent.parameters)
        for key in new_params:
            if isinstance(new_params[key], float):
                new_params[key] += random.uniform(-0.1, 0.1)
                new_params[key] = max(0.0, min(1.0, new_params[key]))

        return Policy(
            id=f"policy_mut_{int(time.time())}_{random.randint(1000,9999)}",
            system_prompt=new_prompt,
            strategy_description=f"Mutation of {parent.id[:20]}",
            parameters=new_params,
            generation=generation,
        )

    def _mutate_prompt(self, prompt: str, rate: float) -> str:
        """Apply random mutations to a prompt string."""
        words = prompt.split()
        if not words:
            return prompt

        mutations = [
            "carefully", "precisely", "step by step", "thoroughly",
            "concisely", "accurately", "with attention to detail",
        ]

        mutated = list(words)
        for i in range(len(mutated)):
            if random.random() < rate:
                if random.random() < 0.5:
                    # Replace with synonym-like word
                    mutated[i] = random.choice(mutations) if random.random() < 0.3 else mutated[i]
                elif i < len(mutated) - 1:
                    # Swap adjacent words
                    mutated[i], mutated[i + 1] = mutated[i + 1], mutated[i]

        result = " ".join(mutated)
        if result == prompt and random.random() < 0.5:
            result = prompt + " " + random.choice(mutations)
        return result

    # ── Fast Phase: RL ─────────────────────────────────────────────────

    def run_fast_phase(self, action: str, reward: float, state: str = ""):
        """Process one RL step: store experience and update policy stats."""
        if not self.best_policy:
            return

        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            done=False,
            policy_id=self.best_policy.id,
            timestamp=time.time(),
        )
        self.experiences.append(exp)

        # Update policy stats
        self.best_policy.trials += 1
        n = self.best_policy.trials
        old_avg = self.best_policy.avg_reward
        self.best_policy.avg_reward = old_avg + (reward - old_avg) / n
        self.best_policy.success_rate = (
            self.best_policy.success_rate * (n - 1) + (1.0 if reward > 0.5 else 0.0)
        ) / n

    def get_best_action(self, state: str) -> str:
        """Get the best action for a state based on past experiences."""
        state_experiences = [
            e for e in self.experiences if e.state == state
        ]
        if not state_experiences:
            return "explore"

        # Choose action with highest avg reward
        action_rewards: Dict[str, List[float]] = {}
        for exp in state_experiences:
            action_rewards.setdefault(exp.action, []).append(exp.reward)

        best_action = max(
            action_rewards,
            key=lambda a: sum(action_rewards[a]) / len(action_rewards[a]),
        )
        return best_action

    # ── Step ───────────────────────────────────────────────────────────

    def step(self, eval_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """Run one step of the fast-slow cycle."""
        self._eval_fn = eval_fn or self._eval_fn
        self.phase_counter += 1

        if self.phase == "slow":
            if self.phase_counter >= self.slow_episodes:
                result = self.run_slow_phase()
                self.phase = "fast"
                self.phase_counter = 0
                return {
                    "phase": "slow_complete",
                    "best_fitness": result.fitness if result else 0,
                    "population_size": len(self.population),
                    "generation": self.current_generation,
                }
            else:
                return {"phase": "slow", "progress": f"{self.phase_counter}/{self.slow_episodes}"}

        else:  # fast phase
            if self.phase_counter >= self.fast_episodes:
                self.phase = "slow"
                self.phase_counter = 0
                return {
                    "phase": "fast_complete",
                    "experiences": len(self.experiences),
                    "best_avg_reward": self.best_policy.avg_reward if self.best_policy else 0,
                }
            else:
                return {"phase": "fast", "progress": f"{self.phase_counter}/{self.fast_episodes}"}

    def learn(
        self,
        task_fn: Callable[[str], float],
        num_cycles: int = 3,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run the full fast-slow learning cycle for a given task."""
        history = []

        for cycle in range(num_cycles):
            # Slow: explore diverse strategies
            slow_result = None
            for _ in range(self.slow_episodes):
                result = self.step()
                slow_result = result
                if verbose:
                    print(f"  Slow step: {result}")

            # Fast: exploit the best strategy
            for _ in range(self.fast_episodes):
                action = "default"
                if self.best_policy:
                    action = f"policy_{self.best_policy.id[:8]}"

                reward = task_fn(action)
                self.run_fast_phase(action, reward)

                result = self.step()
                if verbose:
                    print(f"  Fast step: reward={reward:.3f}")

            history.append(
                {
                    "cycle": cycle + 1,
                    "generation": self.current_generation,
                    "best_fitness": self.best_policy.fitness if self.best_policy else 0,
                    "avg_reward": self.best_policy.avg_reward if self.best_policy else 0,
                }
            )

        return {
            "cycles": num_cycles,
            "final_generation": self.current_generation,
            "best_policy": self.best_policy,
            "history": history,
        }

    # ── Persistence ────────────────────────────────────────────────────

    def save(self):
        os.makedirs(self._persist_dir, exist_ok=True)

        data = {
            "population": [asdict(p) for p in self.population],
            "experiences": [asdict(e) for e in self.experiences[-1000:]],  # keep last 1000
            "current_generation": self.current_generation,
            "phase": self.phase,
            "phase_counter": self.phase_counter,
        }
        if self.best_policy:
            data["best_policy"] = asdict(self.best_policy)

        with open(os.path.join(self._persist_dir, "fastslow.json"), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self):
        try:
            with open(os.path.join(self._persist_dir, "fastslow.json")) as f:
                data = json.load(f)
                self.population = [Policy(**p) for p in data.get("population", [])]
                self.experiences = [Experience(**e) for e in data.get("experiences", [])]
                self.current_generation = data.get("current_generation", 0)
                self.phase = data.get("phase", "slow")
                self.phase_counter = data.get("phase_counter", 0)
                if data.get("best_policy"):
                    self.best_policy = Policy(**data["best_policy"])
        except (FileNotFoundError, json.JSONDecodeError):
            pass


_instances: Dict[str, FastSlowLearner] = {}


def get_fastslow(namespace: str = "default") -> FastSlowLearner:
    if namespace not in _instances:
        _instances[namespace] = FastSlowLearner(namespace)
        _instances[namespace].load()
    return _instances[namespace]
