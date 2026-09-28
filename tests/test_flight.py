"""Tests for the onboard leader-swarm environment."""

from __future__ import annotations

import math
import random

from nio.flight import (
    LeaderSwarmEnv,
    LinearPolicy,
    fixed_heading_actions,
    plan_actions,
    plan_actions_grid,
    train_policy,
)


def _rollout(env, actions_for, seed=3, limit=12):
    observation, info = env.reset(seed=seed)
    total = 0.0
    done = False
    steps = 0
    while not done and steps < limit:
        observation, reward, terminated, truncated, info = env.step(actions_for(env, observation))
        total += reward
        done = terminated or truncated
        steps += 1
    return observation, total, info


def test_reset_and_step_contract():
    env = LeaderSwarmEnv(n_drones=6, n_leaders=2, seed=1)
    observation, info = env.reset(seed=4)
    assert len(observation) == 2
    assert len(observation[0]) == env.obs_dim
    assert "positions" in info
    observation, reward, terminated, truncated, info = env.step([0, 8])
    assert len(observation) == 2
    assert math.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    for position in info["positions"]:
        assert 0.0 <= position[0] <= env.width
        assert 0.0 <= position[1] <= env.height


def test_planner_and_grid_actions_are_legal():
    env = LeaderSwarmEnv(seed=2)
    env.reset(seed=2)
    for actions in (plan_actions(env, algorithm="gwo", iterations=6, population_size=8), plan_actions_grid(env, "aco"), plan_actions_grid(env, "rfd"), fixed_heading_actions(env)):
        assert len(actions) == env.n_leaders
        assert all(0 <= action < env.n_actions for action in actions)


def test_policy_trains_and_roundtrips_json():
    env = LeaderSwarmEnv(n_drones=4, n_leaders=1, max_steps=8, seed=0)
    policy, returns = train_policy(env, episodes=3, seed=1)
    assert len(returns) == 3
    assert all(math.isfinite(value) for value in returns)
    restored = LinearPolicy.from_json(policy.to_json())
    observation, _info = env.reset(seed=5)
    original = policy.act(observation[0], random.Random(0), greedy=True)
    copied = restored.act(observation[0], random.Random(0), greedy=True)
    assert original == copied


def test_episode_reward_is_finite():
    env = LeaderSwarmEnv(max_steps=15, seed=1)
    _observation, total, info = _rollout(env, lambda e, o: fixed_heading_actions(e), seed=9, limit=15)
    assert math.isfinite(total)
    assert info["steps"] >= 1
