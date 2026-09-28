"""Lightweight 2D leader-swarm environment for onboard path coordination.

The environment has no third-party dependencies. ``reset`` and ``step`` follow
the Gymnasium 5-tuple so another agent can drive it without installing
Gymnasium itself.

Leaders receive discrete heading and speed actions. Followers hold formation
with boids and are pulled toward the nearest leader. One shared reward asks
the leaders to make progress together and to stay clear of obstacles.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from nio.motion import boids_step
from nio.pathfinding import OccupancyGrid

HEADING_COUNT = 8
SPEEDS = (1.6, 2.8)
OBS_DIM = 13


def _angle_to_index(angle: float) -> int:
    return int(round(angle / (math.pi / 4.0))) % HEADING_COUNT


class LeaderSwarmEnv:
    """Horizontal world in which lead drones choose the next step of a path."""

    def __init__(
        self,
        n_drones: int = 8,
        n_leaders: int = 2,
        width: float = 100.0,
        height: float = 100.0,
        max_steps: int = 70,
        goal: Sequence[float] = (90.0, 18.0),
        goal_radius: float = 8.0,
        obstacles: Optional[Sequence[Sequence[float]]] = None,
        start: Sequence[float] = (10.0, 66.0),
        seed: int | None = None,
    ) -> None:
        if n_drones < 1:
            raise ValueError("n_drones must be positive")
        if not 1 <= n_leaders <= n_drones:
            raise ValueError("n_leaders must be between 1 and n_drones")
        self.n_drones = n_drones
        self.n_leaders = n_leaders
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.goal = (float(goal[0]), float(goal[1]))
        self.goal_radius = goal_radius
        self.start = (float(start[0]), float(start[1]))
        self.obstacles = list(obstacles) if obstacles is not None else [
            (50.0, 68.0, 14.0),
        ]
        self.drone_radius = 1.2
        self.obs_dim = OBS_DIM
        self.n_actions = HEADING_COUNT * len(SPEEDS)
        self.random = random.Random(seed)
        self.positions: List[List[float]] = []
        self.velocities: List[List[float]] = []
        self.headings: List[float] = []
        self.steps = 0
        self.trace: List[List[List[float]]] = []
        self._leader_distance = 0.0

    def reset(self, seed: int | None = None) -> Tuple[List[List[float]], dict]:
        if seed is not None:
            self.random = random.Random(seed)
        self.steps = 0
        self.positions = []
        sx, sy = self.start
        for _ in range(self.n_drones):
            self.positions.append(
                [
                    sx + self.random.uniform(-1.5, 1.5),
                    sy + self.random.uniform(-3.0, 3.0),
                ]
            )
        self.velocities = [[0.0, 0.0] for _ in range(self.n_drones)]
        self.headings = [0.0 for _ in range(self.n_drones)]
        self.trace = [self._copy_positions()]
        self._leader_distance = self.leader_goal_distance()
        return self.observation(), self.info()

    def step(self, actions: Sequence[int]) -> Tuple[List[List[float]], float, bool, bool, dict]:
        if len(actions) != self.n_leaders:
            raise ValueError(f"expected {self.n_leaders} actions, got {len(actions)}")
        for action in actions:
            if not 0 <= int(action) < self.n_actions:
                raise ValueError(f"action {action} is outside 0..{self.n_actions - 1}")

        previous = self._leader_distance
        collisions = 0
        for index, action in enumerate(actions):
            heading_index = int(action) % HEADING_COUNT
            speed = SPEEDS[int(action) // HEADING_COUNT]
            heading = heading_index * (math.pi / 4.0)
            self.headings[index] = heading
            self.velocities[index] = [speed * math.cos(heading), speed * math.sin(heading)]
            self.positions[index][0] += self.velocities[index][0]
            self.positions[index][1] += self.velocities[index][1]
            self.positions[index], hit = self._resolve(self.positions[index])
            collisions += int(hit)

        follower_ids = list(range(self.n_leaders, self.n_drones))
        if follower_ids:
            anchors = [self.positions[i] for i in range(self.n_leaders)]
            f_pos = [self.positions[i] for i in follower_ids]
            f_vel = [self.velocities[i] for i in follower_ids]
            moved, velocities = boids_step(
                f_pos,
                f_vel,
                perception=28.0,
                separation_radius=4.5,
                max_speed=SPEEDS[-1],
                separation_weight=0.6,
                alignment_weight=0.15,
                cohesion_weight=0.12,
                anchors=anchors,
                anchor_weight=0.45,
            )
            for local, index in enumerate(follower_ids):
                self.positions[index] = moved[local]
                self.velocities[index] = velocities[local]
                self.headings[index] = math.atan2(velocities[local][1], velocities[local][0])
                self.positions[index], hit = self._resolve(self.positions[index])
                collisions += int(hit)

        self.steps += 1
        self.trace.append(self._copy_positions())
        self._leader_distance = self.leader_goal_distance()
        progress = previous - self._leader_distance
        separation_penalty = self._leader_separation_penalty()
        reward = progress - 1.5 * collisions - separation_penalty - 0.02
        terminated = self._leader_distance <= self.goal_radius
        if terminated:
            reward += 20.0
        truncated = self.steps >= self.max_steps and not terminated
        return self.observation(), reward, terminated, truncated, self.info(collisions=collisions)

    def observation(self) -> List[List[float]]:
        return [self._leader_observation(index) for index in range(self.n_leaders)]

    def info(self, collisions: int = 0) -> dict:
        return {
            "positions": self._copy_positions(),
            "leader_distance": self._leader_distance,
            "steps": self.steps,
            "collisions": collisions,
        }

    def leader_centroid(self) -> Tuple[float, float]:
        sx = sum(self.positions[i][0] for i in range(self.n_leaders))
        sy = sum(self.positions[i][1] for i in range(self.n_leaders))
        return sx / self.n_leaders, sy / self.n_leaders

    def leader_goal_distance(self) -> float:
        cx, cy = self.leader_centroid()
        return math.hypot(cx - self.goal[0], cy - self.goal[1])

    def waypoint_cost(self, waypoint: Sequence[float], origin: Sequence[float] | None = None) -> float:
        """Cost of a candidate waypoint: goal distance, clearance, and separation."""
        x, y = float(waypoint[0]), float(waypoint[1])
        start = origin if origin is not None else self.leader_centroid()
        cost = math.hypot(x - self.goal[0], y - self.goal[1])
        travel = math.hypot(x - start[0], y - start[1])
        if travel > 30.0:
            cost += 0.4 * (travel - 30.0)
        samples = (0.25, 0.5, 0.75, 1.0)
        for fraction in samples:
            sx = start[0] + fraction * (x - start[0])
            sy = start[1] + fraction * (y - start[1])
            cost += self._clearance_penalty(sx, sy, margin=6.0)
        if self.n_leaders > 1:
            spread = 0.0
            count = 0
            for i in range(self.n_leaders):
                for j in range(i + 1, self.n_leaders):
                    spread += math.hypot(
                        self.positions[i][0] - self.positions[j][0],
                        self.positions[i][1] - self.positions[j][1],
                    )
                    count += 1
            if count and spread / count > 28.0:
                cost += 0.15 * (spread / count - 28.0)
        return cost

    def action_toward(self, leader: int, waypoint: Sequence[float], fast: bool = True) -> int:
        dx = float(waypoint[0]) - self.positions[leader][0]
        dy = float(waypoint[1]) - self.positions[leader][1]
        heading = _angle_to_index(math.atan2(dy, dx))
        speed_index = 1 if fast else 0
        return speed_index * HEADING_COUNT + heading

    def occupancy(self, resolution: int = 18) -> Tuple[OccupancyGrid, Tuple[int, int], Tuple[int, int]]:
        grid = OccupancyGrid.from_obstacles(self.width, self.height, self.obstacles, resolution)
        cx, cy = self.leader_centroid()
        start = grid.locate(cx, cy, self.width, self.height)
        goal = grid.locate(self.goal[0], self.goal[1], self.width, self.height)
        return grid, start, goal

    def _leader_observation(self, index: int) -> List[float]:
        x, y = self.positions[index]
        gx, gy = self.goal
        diagonal = math.hypot(self.width, self.height)
        if self.obstacles:
            nearest = min(
                self.obstacles,
                key=lambda obstacle: math.hypot(x - obstacle[0], y - obstacle[1]) - obstacle[2],
            )
            obstacle_x = (nearest[0] - x) / self.width
            obstacle_y = (nearest[1] - y) / self.height
            clearance = (math.hypot(x - nearest[0], y - nearest[1]) - nearest[2]) / self.width
        else:
            obstacle_x = 0.0
            obstacle_y = 0.0
            clearance = 1.0
        cx, cy = self.leader_centroid()
        if self.n_leaders > 1:
            other = 1 if index == 0 else 0
            lx = (self.positions[other][0] - x) / self.width
            ly = (self.positions[other][1] - y) / self.height
        else:
            lx = ly = 0.0
        goal_x, goal_y = gx - x, gy - y
        goal_norm = math.hypot(goal_x, goal_y) + 1e-9
        if self.obstacles:
            obs_norm = math.hypot(nearest[0] - x, nearest[1] - y) + 1e-9
            alignment = (
                goal_x * (nearest[0] - x) + goal_y * (nearest[1] - y)
            ) / (goal_norm * obs_norm)
            threat = max(0.0, alignment) * max(0.0, 1.0 - clearance * 4.0)
        else:
            threat = 0.0
        return [
            goal_x / self.width,
            goal_y / self.height,
            math.hypot(goal_x, goal_y) / diagonal,
            obstacle_x,
            obstacle_y,
            clearance,
            (cx - x) / self.width,
            (cy - y) / self.height,
            math.sin(self.headings[index]),
            math.cos(self.headings[index]),
            lx,
            ly,
            threat,
        ]

    def _clearance_penalty(self, x: float, y: float, margin: float) -> float:
        penalty = 0.0
        for cx, cy, radius in self.obstacles:
            clearance = math.hypot(x - cx, y - cy) - radius
            if clearance < margin:
                penalty += (margin - clearance) ** 2
        return penalty

    def _leader_separation_penalty(self) -> float:
        if self.n_leaders < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(self.n_leaders):
            for j in range(i + 1, self.n_leaders):
                total += math.hypot(
                    self.positions[i][0] - self.positions[j][0],
                    self.positions[i][1] - self.positions[j][1],
                )
                count += 1
        mean = total / count
        if mean <= 30.0:
            return 0.0
        return 0.03 * (mean - 30.0)

    def _resolve(self, position: Sequence[float]) -> Tuple[List[float], bool]:
        x, y = float(position[0]), float(position[1])
        hit = False
        for cx, cy, radius in self.obstacles:
            dx = x - cx
            dy = y - cy
            dist = math.hypot(dx, dy)
            limit = radius + self.drone_radius
            if dist < limit:
                hit = True
                if dist < 1e-6:
                    dx, dy, dist = 1.0, 0.0, 1.0
                x = cx + dx / dist * limit
                y = cy + dy / dist * limit
        if x < 0.0 or x > self.width or y < 0.0 or y > self.height:
            hit = True
        x = min(self.width, max(0.0, x))
        y = min(self.height, max(0.0, y))
        return [x, y], hit

    def _copy_positions(self) -> List[List[float]]:
        return [list(position) for position in self.positions]
