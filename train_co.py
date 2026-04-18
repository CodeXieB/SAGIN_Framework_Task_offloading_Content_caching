"""
Legacy compatibility entrypoint.

Use train_baseline.py / train_safe_a.py / train_safe_b.py for explicit experiments.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_baseline import main  # noqa: E402


if __name__ == "__main__":
    main()
