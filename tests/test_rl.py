"""The training environment admits the algorithm list and a small policy."""

from __future__ import annotations

import math
import random

from nio.rl import LinearPolicy, TrainingEnv, rollout, rollout_policy, train_policy


def test_reset_and_step_contract():
    env = TrainingEnv(seed=1, max_steps=5)
    observation, info = env.reset(seed=4)
    assert len(observation) == env.obs_dim
    assert info["steps"] == 0
    point = [0.0, 0.0]
    observation, reward, terminated, truncated, info = env.step(point)
    assert len(observation) == env.obs_dim
    assert math.isfinite(reward)
    assert reward >= 0.0
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["best_value"] <= info["value"] or info["best_value"] == info["value"]
    for value, (lo, hi) in zip(info["position"], env.bounds):
        assert lo <= value <= hi


def test_registered_optimizer_can_take_an_episode():
    env = TrainingEnv(max_steps=12, seed=0)
    _total, info = rollout(env, "gwo", population_size=12, seed=2)
    assert math.isfinite(info["best_value"])
    assert info["best_value"] < 0.5
    assert info["steps"] >= 1


def test_policy_trains_and_roundtrips_json():
    env = TrainingEnv(max_steps=6, seed=0)
    policy, returns = train_policy(env, episodes=3, seed=1)
    assert len(returns) == 3
    assert all(math.isfinite(value) for value in returns)
    restored = LinearPolicy.from_json(policy.to_json())
    observation, _info = env.reset(seed=5)
    rng = random.Random(0)
    original = policy.act(observation, rng, greedy=True)
    copied = restored.act(observation, random.Random(0), greedy=True)
    assert original == copied
    _total, info = rollout_policy(env, restored, seed=5)
    assert math.isfinite(info["best_value"])
