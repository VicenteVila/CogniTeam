"""Tests for Fast-Slow memory module."""
import json
import os
import random
import tempfile
import time

import pytest
from cogniteam.memory.fastslow.fastslow import FastSlowLearner, Policy, Experience, get_fastslow


@pytest.fixture
def fs():
    learner = FastSlowLearner(
        namespace="test",
        population_size=6,
        mutation_rate=0.3,
        crossover_rate=0.7,
        elite_ratio=0.3,
        slow_episodes=2,
        fast_episodes=3,
    )
    return learner


def test_initialize_population(fs):
    fs.initialize_population("You are a helpful assistant that writes code.")
    assert len(fs.population) == 6
    assert fs.best_policy is not None
    for p in fs.population:
        assert p.system_prompt != ""
        assert "temperature" in p.parameters


def test_mutate_prompt_changes_text(fs):
    original = "You are a helpful assistant."
    mutated = fs._mutate_prompt(original, rate=0.9)
    # With high mutation rate, should change
    assert isinstance(mutated, str)
    assert len(mutated) > 0


def test_mutate_prompt_empty(fs):
    assert fs._mutate_prompt("", 0.5) == ""


def test_crossover(fs):
    p1 = Policy(id="p1", system_prompt="Step one. Step two. Step three.",
                 strategy_description="p1", parameters={"temp": 0.7}, fitness=0.9, generation=0)
    p2 = Policy(id="p2", system_prompt="Alpha. Beta. Gamma.",
                 strategy_description="p2", parameters={"temp": 0.5}, fitness=0.8, generation=0)
    child = fs._crossover(p1, p2, generation=1)
    assert child.generation == 1
    assert child.system_prompt != ""
    assert "p1" not in child.id


def test_mutate_policy(fs):
    parent = Policy(id="parent", system_prompt="Be concise and clear.",
                    strategy_description="parent", parameters={"temperature": 0.7},
                    fitness=0.8, generation=0)
    child = fs._mutate_policy(parent, generation=1)
    assert child.generation == 1
    assert len(child.strategy_description) > 0
    assert "parent" not in child.id[:10]


def test_run_slow_phase_basic(fs):
    fs.initialize_population("Base prompt for testing.")
    for p in fs.population:
        p.trials = 5
        p.success_rate = random.random()
        p.avg_reward = random.random()

    result = fs.run_slow_phase()
    assert result is not None
    assert fs.current_generation == 1
    assert len(fs.population) == fs.population_size


def test_run_slow_phase_empty_population(fs):
    result = fs.run_slow_phase()
    assert result is not None


def test_run_fast_phase(fs):
    fs.initialize_population("Test prompt")
    fs.run_fast_phase("action_1", reward=0.8, state="state_1")
    assert len(fs.experiences) == 1
    assert fs.best_policy.trials == 1
    assert fs.best_policy.avg_reward == 0.8


def test_run_fast_phase_multiple_updates(fs):
    fs.initialize_population("Test")
    fs.run_fast_phase("action_1", reward=1.0, state="s1")
    fs.run_fast_phase("action_2", reward=0.0, state="s2")
    fs.run_fast_phase("action_1", reward=0.5, state="s1")
    assert fs.best_policy.trials == 3
    assert 0.4 < fs.best_policy.avg_reward < 0.6


def test_get_best_action(fs):
    fs.initialize_population("Test")
    fs.run_fast_phase("good_action", reward=1.0, state="same_state")
    fs.run_fast_phase("bad_action", reward=0.0, state="same_state")
    best = fs.get_best_action("same_state")
    assert best == "good_action"


def test_get_best_action_unknown_state(fs):
    fs.initialize_population("Test")
    assert fs.get_best_action("unknown") == "explore"


def test_step_fast_slow_cycle(fs):
    fs.initialize_population("Test")
    result = fs.step()
    assert "phase" in result
    assert fs.phase_counter >= 0


def test_step_completes_slow_phase(fs):
    fs.initialize_population("Test")
    fs.phase_counter = fs.slow_episodes - 1
    result = fs.step()
    assert result["phase"] == "slow_complete"
    assert fs.phase == "fast"
    assert fs.phase_counter == 0


def test_step_completes_fast_phase(fs):
    fs.initialize_population("Test")
    fs.phase = "fast"
    fs.phase_counter = fs.fast_episodes - 1
    result = fs.step()
    assert result["phase"] == "fast_complete"
    assert fs.phase == "slow"
    assert fs.phase_counter == 0


def test_learn_basic(fs):
    fs.initialize_population("Be helpful.")

    def task_fn(action):
        return random.random()

    result = fs.learn(task_fn, num_cycles=1, verbose=False)
    assert result["cycles"] == 1
    assert len(result["history"]) == 1
    assert result["best_policy"] is not None


def test_persistence(fs):
    fs.initialize_population("Persist test prompt")
    fs.run_fast_phase("act", reward=0.9, state="s1")
    dirname = fs._persist_dir
    fs.save()

    fs2 = FastSlowLearner(namespace="test")
    fs2._persist_dir = dirname
    fs2.load()
    assert len(fs2.population) == 6
    assert len(fs2.experiences) >= 1


def test_persistence_empty():
    with tempfile.TemporaryDirectory() as td:
        fs = FastSlowLearner(namespace="empty_test")
        fs._persist_dir = os.path.join(td, "fs_empty")
        fs.save()
        fs2 = FastSlowLearner(namespace="empty_test")
        fs2._persist_dir = fs._persist_dir
        fs2.load()
        assert fs2.population == []
