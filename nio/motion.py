"""Collective movement models.

These are not minimizers. They update positions and velocities of a group.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

Vector = List[float]


def _limit(vx: float, vy: float, max_speed: float) -> Tuple[float, float]:
    speed = math.hypot(vx, vy)
    if speed > max_speed and speed > 0.0:
        scale = max_speed / speed
        return vx * scale, vy * scale
    return vx, vy


def boids_step(
    positions: Sequence[Sequence[float]],
    velocities: Sequence[Sequence[float]],
    *,
    perception: float = 25.0,
    separation_radius: float = 4.0,
    max_speed: float = 2.5,
    separation_weight: float = 0.8,
    alignment_weight: float = 0.3,
    cohesion_weight: float = 0.2,
    anchors: Optional[Sequence[Sequence[float]]] = None,
    anchor_weight: float = 0.0,
) -> Tuple[List[Vector], List[Vector]]:
    """Reynolds boids. Returns new positions and velocities.

    Reference:
    Reynolds, C. W. (1987). Flocks, herds, and schools: a distributed behavioral
    model. Computer Graphics, 21(4), 25–34.
    """
    new_positions: List[Vector] = []
    new_velocities: List[Vector] = []
    count = len(positions)
    for i in range(count):
        sep_x = sep_y = 0.0
        ali_x = ali_y = 0.0
        coh_x = coh_y = 0.0
        seen = 0
        sep_seen = 0
        ix, iy = positions[i]
        for j in range(count):
            if i == j:
                continue
            dx = positions[j][0] - ix
            dy = positions[j][1] - iy
            dist = math.hypot(dx, dy) + 1e-9
            if dist < perception:
                seen += 1
                ali_x += velocities[j][0]
                ali_y += velocities[j][1]
                coh_x += positions[j][0]
                coh_y += positions[j][1]
                if dist < separation_radius:
                    sep_x -= dx / dist
                    sep_y -= dy / dist
                    sep_seen += 1
        vx = velocities[i][0]
        vy = velocities[i][1]
        if seen:
            ali_x /= seen
            ali_y /= seen
            coh_x = coh_x / seen - ix
            coh_y = coh_y / seen - iy
        if sep_seen:
            sep_x /= sep_seen
            sep_y /= sep_seen
        vx = vx + separation_weight * sep_x + alignment_weight * (ali_x - vx) + cohesion_weight * coh_x
        vy = vy + separation_weight * sep_y + alignment_weight * (ali_y - vy) + cohesion_weight * coh_y
        if anchors and anchor_weight:
            ax, ay = min(
                anchors,
                key=lambda anchor: math.hypot(anchor[0] - ix, anchor[1] - iy),
            )
            vx += anchor_weight * (ax - ix)
            vy += anchor_weight * (ay - iy)
        vx, vy = _limit(vx, vy, max_speed)
        new_velocities.append([vx, vy])
        new_positions.append([ix + vx, iy + vy])
    return new_positions, new_velocities


def vicsek_step(
    positions: Sequence[Sequence[float]],
    headings: Sequence[float],
    *,
    radius: float = 5.0,
    noise: float = 0.1,
    speed: float = 1.0,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Vector], List[float]]:
    """Vicsek self-propelled particles. Returns new positions and headings.

    Reference:
    Vicsek, T., Czirók, A., Ben-Jacob, E., Cohen, I., & Shochet, O. (1995).
    Novel type of phase transition in a system of self-driven particles.
    Physical Review Letters, 75(6), 1226–1229.
    """
    generator = rng or random.Random()
    new_headings: List[float] = []
    new_positions: List[Vector] = []
    for i, (x, y) in enumerate(positions):
        sin_sum = 0.0
        cos_sum = 0.0
        for j, (ox, oy) in enumerate(positions):
            if math.hypot(ox - x, oy - y) <= radius:
                sin_sum += math.sin(headings[j])
                cos_sum += math.cos(headings[j])
        heading = math.atan2(sin_sum, cos_sum) + generator.uniform(-noise, noise)
        new_headings.append(heading)
        new_positions.append([x + speed * math.cos(heading), y + speed * math.sin(heading)])
    return new_positions, new_headings
