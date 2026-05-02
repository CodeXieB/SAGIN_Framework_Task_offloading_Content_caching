from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_common import main_for_experiment  # noqa: E402


def main():
    main_for_experiment(
        default_reward_mode="safe_c",
        experiment_name="Curriculum Safe PPO",
        default_artifact_dir=os.path.join("artifacts", "curriculum_safe"),
        default_overrides={
            "curriculum": True,
            "curriculum_max_level": 2,
            "cmax": 1.9,
            "gamma": 0.5,
            "eta": 1.0,
            "beta3": 0.8,
            "alpha3": 0.05,
        },
    )


if __name__ == "__main__":
    main()
