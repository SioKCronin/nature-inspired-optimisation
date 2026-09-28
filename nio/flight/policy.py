"""A heading policy small enough to copy onto a drone.

Two linear scores, east and north, are dot products with the leader's
observation. Their direction is snapped to a compass heading. Training is
REINFORCE on episodes that beat the running mean. Weights serialise to JSON.
"""

from __future__ import annotations

import json
import math
import random
from typing import List, Sequence, Tuple

from .env import HEADING_COUNT, LeaderSwarmEnv


class LinearPolicy:
    """Shared heading policy. Leaders use the same weights on their own observation."""

    def __init__(self, obs_dim: int, n_actions: int, lr: float = 0.15, seed: int | None = None) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.lr = lr
        self.temperature = 0.45
        generator = random.Random(seed)
        self.east = [generator.uniform(-0.02, 0.02) for _ in range(obs_dim + 1)]
        self.north = [generator.uniform(-0.02, 0.02) for _ in range(obs_dim + 1)]
        # Point at the goal. Feature 0 is eastward goal offset, feature 1 is northward.
        if obs_dim >= 2:
            self.east[0] = 1.0
            self.north[1] = 1.0
        # Feature 4 is the nearest obstacle's northward offset. Steer off it.
        if obs_dim >= 5:
            self.north[4] = -0.35

    @property
    def n_parameters(self) -> int:
        return len(self.east) + len(self.north)

    def _features(self, observation: Sequence[float]) -> List[float]:
        return list(observation) + [1.0]

    def _scores(self, observation: Sequence[float]) -> Tuple[float, float, List[float]]:
        features = self._features(observation)
        east = sum(weight * value for weight, value in zip(self.east, features))
        north = sum(weight * value for weight, value in zip(self.north, features))
        preferred = math.atan2(north, east)
        logits = []
        for heading in range(HEADING_COUNT):
            angle = heading * (math.pi / 4.0)
            logits.append(math.cos(angle - preferred) / self.temperature)
        # Prefer the faster of the two speeds unless the slower one is a much better heading.
        scores = logits + [value + 0.15 for value in logits]
        scores = scores[: self.n_actions]
        return east, north, scores

    def probabilities(self, observation: Sequence[float]) -> List[float]:
        _east, _north, scores = self._scores(observation)
        peak = max(scores)
        shifted = [math.exp(max(-20.0, min(20.0, score - peak))) for score in scores]
        total = sum(shifted) + 1e-12
        return [value / total for value in shifted]

    def act(self, observation: Sequence[float], rng: random.Random, greedy: bool = False) -> int:
        probabilities = self.probabilities(observation)
        if greedy:
            return max(range(self.n_actions), key=lambda index: probabilities[index])
        pick = rng.random()
        acc = 0.0
        for index, probability in enumerate(probabilities):
            acc += probability
            if acc >= pick:
                return index
        return self.n_actions - 1

    def update(self, observation: Sequence[float], action: int, advantage: float) -> None:
        self.apply_batch([(observation, action)], advantage)

    def apply_batch(self, samples: Sequence[Tuple[Sequence[float], int]], scale: float) -> None:
        """Apply one averaged REINFORCE update for a whole episode."""
        if not samples:
            return
        clipped = max(-1.0, min(1.0, scale))
        east_grad = [0.0 for _ in self.east]
        north_grad = [0.0 for _ in self.north]
        for observation, action in samples:
            features = self._features(observation)
            east, north, scores = self._scores(observation)
            peak = max(scores)
            shifted = [math.exp(max(-20.0, min(20.0, score - peak))) for score in scores]
            total = sum(shifted) + 1e-12
            probabilities = [value / total for value in shifted]
            preferred = math.atan2(north, east)
            grad_preferred = 0.0
            for action_index, probability in enumerate(probabilities):
                match = 1.0 if action_index == action else 0.0
                angle = (action_index % HEADING_COUNT) * (math.pi / 4.0)
                d_logit = math.sin(angle - preferred) / self.temperature
                grad_preferred += (match - probability) * d_logit
            grad_preferred *= clipped
            denom = east * east + north * north + 1e-8
            d_east = -north / denom
            d_north = east / denom
            for index, feature in enumerate(features):
                east_grad[index] += grad_preferred * d_east * feature
                north_grad[index] += grad_preferred * d_north * feature
        count = float(len(samples))
        for index, value in enumerate(east_grad):
            self.east[index] += self.lr * value / count
            self.north[index] += self.lr * north_grad[index] / count

    def to_json(self) -> str:
        return json.dumps(
            {
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "lr": self.lr,
                "temperature": self.temperature,
                "east": self.east,
                "north": self.north,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> "LinearPolicy":
        data = json.loads(payload)
        policy = cls(int(data["obs_dim"]), int(data["n_actions"]), lr=float(data.get("lr", 0.15)))
        policy.east = data["east"]
        policy.north = data["north"]
        policy.temperature = float(data.get("temperature", policy.temperature))
        return policy


def train_policy(
    env: LeaderSwarmEnv,
    episodes: int = 40,
    seed: int = 0,
    lr: float = 0.2,
    gamma: float = 1.0,
) -> Tuple[LinearPolicy, List[float]]:
    """Train one shared policy on team return. Returns the policy and episode returns.

    Each episode is scored by its undiscounted return. Only episodes that beat
    the running mean move the weights, so a drone that scrapes an obstacle does
    not get reinforced for the early progress that preceded the collision.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    policy = LinearPolicy(env.obs_dim, env.n_actions, lr=lr, seed=seed)
    rng = random.Random(seed + 1)
    returns: List[float] = []
    running = 0.0
    best_return = float("-inf")
    best_samples: List[Tuple[List[float], int]] = []
    for episode in range(episodes):
        observation, _info = env.reset(seed=seed + episode)
        pairs = []
        rewards: List[float] = []
        done = False
        while not done:
            actions = [policy.act(leader_obs, rng) for leader_obs in observation]
            pairs.append(([list(leader_obs) for leader_obs in observation], list(actions)))
            observation, reward, terminated, truncated, _info = env.step(actions)
            rewards.append(float(reward))
            done = terminated or truncated
        if gamma == 1.0:
            episode_return = sum(rewards)
        else:
            gain = 0.0
            for reward in reversed(rewards):
                gain = reward + gamma * gain
            episode_return = gain
        returns.append(sum(rewards))
        samples = [
            (list(leader_obs), action)
            for observations, actions in pairs
            for leader_obs, action in zip(observations, actions)
        ]
        if episode_return > best_return:
            best_return = episode_return
            best_samples = samples
        if episode == 0:
            running = episode_return
            continue
        advantage = episode_return - running
        running += (episode_return - running) / (episode + 1)
        if advantage <= 0.0:
            continue
        scale = min(1.0, advantage / 80.0)
        for _ in range(4):
            policy.apply_batch(samples, scale)
    if best_samples:
        for _ in range(8):
            policy.apply_batch(best_samples, 1.0)
    return policy, returns
