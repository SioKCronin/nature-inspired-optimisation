"""Play one algorithm from the list, then a trained policy, on the same episode.

    python examples/train_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nio.rl import TrainingEnv, rollout, rollout_policy, train_policy  # noqa: E402


def main() -> None:
    env = TrainingEnv(max_steps=20, seed=0)
    _wolf_return, wolf = rollout(env, "gwo", seed=0)
    policy, returns = train_policy(env, episodes=6, seed=1)
    _policy_return, played = rollout_policy(env, policy, seed=2)
    print(f"grey wolf best {wolf['best_value']:.4f}")
    print(f"trained policy best {played['best_value']:.4f}")
    print(f"training returns {[round(value, 4) for value in returns]}")


if __name__ == "__main__":
    main()
