from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_common import main_for_experiment  # noqa: E402


def main():
    main_for_experiment(
        default_reward_mode="safe_a",
        experiment_name="Safe PPO A",
        default_artifact_dir=os.path.join("artifacts", "safe_a"),
        default_overrides={
            "gamma": 0.6,
            "eta": 0.8,
            "cmax": 1.9,
        },
    )


if __name__ == "__main__":
    main()
