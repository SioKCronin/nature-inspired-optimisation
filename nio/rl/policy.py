"""A small proposal policy for :class:`nio.rl.TrainingEnv`.

Each coordinate is a dot product with the observation, squashed into the
bounds. Training keeps episodes that beat a running mean and nudges the
weights toward the points those episodes proposed. Weights serialise to JSON.
"""

from __future__ import annotations

import json
import math
import random
from typing import List, Sequence, Tuple

from .env import TrainingEnv


class LinearPolicy:
    """Map an observation to a point inside the environment bounds."""

    def __init__(
        self,
        obs_dim: int,
        bounds: Sequence[Tuple[float, float]],
        lr: float = 0.08,
        seed: int | None = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.lr = float(lr)
        generator = random.Random(seed)
        width = obs_dim + 1
        self.weights = [
            [generator.uniform(-0.02, 0.02) for _ in range(width)]
            for _ in self.bounds
        ]

    @property
    def dimension(self) -> int:
        return len(self.bounds)

    def _features(self, observation: Sequence[float]) -> List[float]:
        return list(observation) + [1.0]

    def act(
        self,
        observation: Sequence[float],
        rng: random.Random,
        greedy: bool = False,
    ) -> List[float]:
        features = self._features(observation)
        point = []
        for index, weight in enumerate(self.weights):
            score = sum(value * feature for value, feature in zip(weight, features))
            if not greedy:
                score += rng.gauss(0.0, 0.2)
            unit = math.tanh(score)
            lo, hi = self.bounds[index]
            center = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            point.append(center + half * unit)
        return point

    def reinforce(
        self, samples: Sequence[Tuple[Sequence[float], Sequence[float]]], scale: float
    ) -> None:
        if not samples or scale == 0.0:
            return
        step = self.lr * scale / float(len(samples))
        for observation, action in samples:
            features = self._features(observation)
            for index, weight in enumerate(self.weights):
                lo, hi = self.bounds[index]
                center = 0.5 * (lo + hi)
                half = 0.5 * (hi - lo) or 1.0
                target = (float(action[index]) - center) / half
                if target > 0.999:
                    target = 0.999
                elif target < -0.999:
                    target = -0.999
                score = sum(value * feature for value, feature in zip(weight, features))
                pred = math.tanh(score)
                grad = (target - pred) * (1.0 - pred * pred)
                for feature_index, feature in enumerate(features):
                    weight[feature_index] += step * grad * feature

    def to_json(self) -> str:
        payload = {
            "obs_dim": self.obs_dim,
            "bounds": self.bounds,
            "lr": self.lr,
            "weights": self.weights,
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, text: str) -> "LinearPolicy":
        payload = json.loads(text)
        policy = cls(payload["obs_dim"], payload["bounds"], lr=payload.get("lr", 0.08))
        policy.weights = [[float(value) for value in row] for row in payload["weights"]]
        return policy


def rollout_policy(
    env: TrainingEnv,
    policy: LinearPolicy,
    *,
    seed: int | None = None,
    greedy: bool = True,
) -> Tuple[float, dict]:
    """Play one episode with a policy. Returns total reward and the last info."""
    rng = random.Random(seed)
    observation, _info = env.reset(seed=seed)
    total = 0.0
    terminated = False
    truncated = False
    info = env.info()
    while not terminated and not truncated:
        action = policy.act(observation, rng, greedy=greedy)
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
    return total, info


def train_policy(
    env: TrainingEnv,
    *,
    episodes: int = 8,
    seed: int | None = None,
    lr: float = 0.08,
) -> Tuple[LinearPolicy, List[float]]:
    """Fit a linear proposal. Episodes below the running mean leave the weights alone."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    policy = LinearPolicy(env.obs_dim, env.bounds, lr=lr, seed=seed)
    rng = random.Random(0 if seed is None else seed + 1)
    returns: List[float] = []
    baseline = 0.0
    seen = False
    for episode in range(episodes):
        episode_seed = None if seed is None else seed + episode
        observation, _info = env.reset(seed=episode_seed)
        samples: List[Tuple[Sequence[float], Sequence[float]]] = []
        total = 0.0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = policy.act(observation, rng, greedy=False)
            samples.append((list(observation), list(action)))
            observation, reward, terminated, truncated, _info = env.step(action)
            total += reward
        returns.append(total)
        if not seen:
            baseline = total
            seen = True
            continue
        advantage = total - baseline
        baseline = 0.8 * baseline + 0.2 * total
        if advantage > 0.0:
            policy.reinforce(samples, advantage)
    return policy, returns
