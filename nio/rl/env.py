"""Training episode for any algorithm in the corpus.

An action is a candidate point inside the bounds. The reward is how much that
point improved the best value seen. ``reset`` and ``step`` return the
Gymnasium 5-tuple, and they do not import Gymnasium.

``rollout`` lets one registered optimizer take the episode. The name is any
key in :data:`nio.registry.OPTIMIZERS`.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Tuple

from nio.base import sphere
from nio.registry import get_optimizer

Objective = Callable[[Sequence[float]], float]


class TrainingEnv:
    """Bounded minimisation episode. One step proposes one point."""

    def __init__(
        self,
        objective: Objective = sphere,
        bounds: Sequence[Tuple[float, float]] = ((-5.12, 5.12), (-5.12, 5.12)),
        max_steps: int = 30,
        tolerance: float = 1e-2,
        seed: int | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if len(bounds) == 0:
            raise ValueError("bounds must be non-empty")
        self.objective = objective
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.dimension = len(self.bounds)
        self.max_steps = int(max_steps)
        self.tolerance = float(tolerance)
        self.obs_dim = self.dimension * 2 + 3
        self.random = random.Random(seed)
        self.position: List[float] = []
        self.best_position: List[float] = []
        self.value = float("inf")
        self.best_value = float("inf")
        self.steps = 0

    def _clamp(self, action: Sequence[float]) -> List[float]:
        if len(action) != self.dimension:
            raise ValueError(f"expected {self.dimension} values, got {len(action)}")
        point = []
        for value, (lo, hi) in zip(action, self.bounds):
            number = float(value)
            if number < lo:
                number = lo
            elif number > hi:
                number = hi
            point.append(number)
        return point

    def _scale(self) -> float:
        total = 0.0
        for lo, hi in self.bounds:
            span = hi - lo
            total += span * span
        return total if total > 0.0 else 1.0

    def observation(self) -> List[float]:
        scale = self._scale()
        fraction = self.steps / float(self.max_steps)
        return (
            list(self.position)
            + list(self.best_position)
            + [self.value / scale, self.best_value / scale, fraction]
        )

    def info(self) -> dict:
        return {
            "position": list(self.position),
            "best_position": list(self.best_position),
            "value": self.value,
            "best_value": self.best_value,
            "steps": self.steps,
        }

    def reset(self, seed: int | None = None) -> Tuple[List[float], dict]:
        if seed is not None:
            self.random = random.Random(seed)
        self.steps = 0
        self.position = [self.random.uniform(lo, hi) for lo, hi in self.bounds]
        self.value = float(self.objective(self.position))
        self.best_position = list(self.position)
        self.best_value = self.value
        return self.observation(), self.info()

    def step(
        self, action: Sequence[float]
    ) -> Tuple[List[float], float, bool, bool, dict]:
        previous = self.best_value
        self.position = self._clamp(action)
        self.value = float(self.objective(self.position))
        if self.value < self.best_value:
            self.best_value = self.value
            self.best_position = list(self.position)
        self.steps += 1
        reward = previous - self.best_value
        terminated = self.best_value <= self.tolerance
        truncated = self.steps >= self.max_steps
        return self.observation(), reward, terminated, truncated, self.info()


def rollout(
    env: TrainingEnv,
    algorithm: str,
    *,
    population_size: int = 16,
    seed: Optional[int] = None,
) -> Tuple[float, dict]:
    """Hand the episode to one optimizer from the registry.

    Each optimizer step offers its current best point as the action.
    """
    optimizer = get_optimizer(algorithm)(
        objective=env.objective,
        bounds=env.bounds,
        population_size=population_size,
        seed=0 if seed is None else int(seed),
    )
    optimizer.iterations = env.max_steps
    optimizer.initialise()
    _observation, _info = env.reset(seed=seed)
    total = 0.0
    observation, reward, terminated, truncated, info = env.step(optimizer.best_position)
    total += reward
    for iteration in range(env.max_steps - 1):
        if terminated or truncated:
            break
        optimizer.iteration = iteration
        if hasattr(env.objective, "update"):
            env.objective.update(iteration)
        optimizer.step()
        observation, reward, terminated, truncated, info = env.step(optimizer.best_position)
        total += reward
    del observation
    return total, info
