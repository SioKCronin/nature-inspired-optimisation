"""Interview demo: fixed heading, grey-wolf plan, and a pocket policy.

Run from the repository root::

    python examples/leader_swarm_demo.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nio.flight import (  # noqa: E402
    LeaderSwarmEnv,
    fixed_heading_actions,
    plan_actions,
    train_policy,
)


def run_episode(env, actor, seed):
    observation, _info = env.reset(seed=seed)
    total = 0.0
    done = False
    while not done:
        observation, reward, terminated, truncated, info = env.step(actor(env, observation))
        total += reward
        done = terminated or truncated
    trace = [frame[:] for frame in env.trace]
    return total, info, trace


def maybe_plot(traces, env, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the trail plot.")
        return
    colors = {"Fixed heading": "#4c78a8", "Grey wolf plan": "#f58518", "Trained policy": "#54a24b"}
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    for cx, cy, radius in env.obstacles:
        circle = plt.Circle((cx, cy), radius, color="#d0d0d0", zorder=0)
        axis.add_patch(circle)
    goal = plt.Circle(env.goal, env.goal_radius, fill=False, linestyle="--", color="#333333")
    axis.add_patch(goal)
    for label, trace in traces.items():
        for leader in range(min(2, len(trace[0]))):
            xs = [frame[leader][0] for frame in trace]
            ys = [frame[leader][1] for frame in trace]
            axis.plot(
                xs,
                ys,
                color=colors[label],
                linewidth=2,
                label=label if leader == 0 else None,
            )
    axis.set_xlim(0, env.width)
    axis.set_ylim(0, env.height)
    axis.set_aspect("equal")
    axis.set_title("Lead drones, same world")
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    print(f"Wrote {path}")


def main() -> None:
    seed = 7
    env = LeaderSwarmEnv(n_drones=8, n_leaders=2, seed=seed)
    fixed_return, fixed_info, fixed_trace = run_episode(
        env, lambda episode_env, _obs: fixed_heading_actions(episode_env), seed
    )
    planned_return, planned_info, planned_trace = run_episode(
        env,
        lambda episode_env, _obs: plan_actions(episode_env, algorithm="gwo", iterations=12, population_size=10),
        seed,
    )
    policy, training = train_policy(env, episodes=40, seed=seed)
    rng = random.Random(seed)
    learned_return, learned_info, learned_trace = run_episode(
        env,
        lambda episode_env, observation: [
            policy.act(leader_obs, rng, greedy=True) for leader_obs in observation
        ],
        seed,
    )
    print("Lead-swarm returns on one world")
    print(f"  fixed heading     {fixed_return:8.2f}   distance {fixed_info['leader_distance']:.1f}")
    print(f"  grey wolf plan    {planned_return:8.2f}   distance {planned_info['leader_distance']:.1f}")
    print(f"  trained policy    {learned_return:8.2f}   distance {learned_info['leader_distance']:.1f}")
    print(f"  policy weights    {policy.n_parameters} floats")
    print(f"  training episodes {len(training)}  last return {training[-1]:.2f}")
    maybe_plot(
        {
            "Fixed heading": fixed_trace,
            "Grey wolf plan": planned_trace,
            "Trained policy": learned_trace,
        },
        env,
        Path(__file__).with_name("leader_swarm.png"),
    )


if __name__ == "__main__":
    main()
