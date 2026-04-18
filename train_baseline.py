from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_common import main_for_experiment  # noqa: E402


def main():
    main_for_experiment(
        default_reward_mode="baseline",
        experiment_name="Baseline PPO",
        default_artifact_dir=os.path.join("artifacts", "baseline"),
    )


if __name__ == "__main__":
    main()
