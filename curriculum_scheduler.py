"""
Learning-progress curriculum scheduler for SAGIN PPO training.

The scheduler changes only environment difficulty parameters that keep the
observation dimension fixed. Grid size and UAV count must stay unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class CurriculumLevel:
    name: str
    env: Dict[str, int]
    reward_threshold: float


@dataclass
class CurriculumDecision:
    changed: bool
    old_level: int
    new_level: int
    reason: str
    window_reward: float
    value_loss_ema: float
    learning_progress: float
    safe_violation_rate: float


DEFAULT_LEVELS: List[CurriculumLevel] = [
    CurriculumLevel(
        name="easy",
        env={
            "max_active_iot": 4,
            "cache_size": 60,
            "compute_power_uav": 35,
            "energy": 100000,
            "ofdm_slots": 9,
        },
        reward_threshold=-25.0,
    ),
    CurriculumLevel(
        name="medium",
        env={
            "max_active_iot": 7,
            "cache_size": 50,
            "compute_power_uav": 30,
            "energy": 90000,
            "ofdm_slots": 7,
        },
        reward_threshold=-22.0,
    ),
    CurriculumLevel(
        name="target",
        env={
            "max_active_iot": 10,
            "cache_size": 40,
            "compute_power_uav": 25,
            "energy": 80000,
            "ofdm_slots": 6,
        },
        reward_threshold=-20.0,
    ),
    CurriculumLevel(
        name="hard",
        env={
            "max_active_iot": 12,
            "cache_size": 35,
            "compute_power_uav": 22,
            "energy": 70000,
            "ofdm_slots": 5,
        },
        reward_threshold=-18.0,
    ),
]


class CurriculumScheduler:
    """Adapt environment difficulty from reward, value loss, and safety signals."""

    def __init__(
        self,
        levels: Optional[List[CurriculumLevel]] = None,
        start_level: int = 0,
        max_level: Optional[int] = None,
        window: int = 20,
        rho: float = 0.8,
        lp_epsilon: float = 0.15,
        safe_epsilon: float = 0.85,
        drop_patience: int = 2,
        drop_tolerance: float = 2.0,
    ):
        self.levels = levels or DEFAULT_LEVELS
        self.max_level = len(self.levels) - 1 if max_level is None else max(0, min(max_level, len(self.levels) - 1))
        self.level = max(0, min(start_level, self.max_level))
        self.window = max(1, window)
        self.rho = min(max(rho, 0.0), 0.999)
        self.lp_epsilon = max(lp_epsilon, 0.0)
        self.safe_epsilon = max(safe_epsilon, 0.0)
        self.drop_patience = max(drop_patience, 1)
        self.drop_tolerance = max(drop_tolerance, 0.0)
        self.value_loss_ema: Optional[float] = None
        self.learning_progress = float("inf")
        self.window_rewards: List[float] = []

    @property
    def current_level(self) -> CurriculumLevel:
        return self.levels[self.level]

    def apply_to_args(self, args: Any) -> None:
        for name, value in self.current_level.env.items():
            setattr(args, name, value)

    def restore(self, metrics: Dict[str, List[float]]) -> None:
        levels = metrics.get("episode_curriculum_level", [])
        if levels:
            finite = [int(v) for v in levels if np.isfinite(v)]
            if finite:
                self.level = max(0, min(finite[-1], self.max_level))

        emas = metrics.get("episode_value_loss_ema", [])
        if emas:
            finite = [float(v) for v in emas if np.isfinite(v)]
            if finite:
                self.value_loss_ema = finite[-1]

        lps = metrics.get("episode_learning_progress", [])
        if lps:
            finite = [float(v) for v in lps if np.isfinite(v)]
            if finite:
                self.learning_progress = finite[-1]

    def update(self, metrics: Dict[str, List[float]], steps_per_episode: int) -> CurriculumDecision | None:
        rewards = metrics["episode_rewards"]
        v_losses = metrics["episode_v_loss"]
        violations = metrics["episode_violation_steps"]
        if len(rewards) < self.window or len(rewards) % self.window != 0:
            return None

        window_rewards = np.asarray(rewards[-self.window:], dtype=np.float64)
        window_v_losses = np.asarray(v_losses[-self.window:], dtype=np.float64)
        window_violations = np.asarray(violations[-self.window:], dtype=np.float64)

        window_reward = float(np.nanmean(window_rewards))
        window_v_loss = float(np.nanmean(window_v_losses))
        safe_violation_rate = float(np.nanmean(window_violations) / max(steps_per_episode, 1))

        prev_ema = self.value_loss_ema
        if prev_ema is None or not np.isfinite(prev_ema):
            self.value_loss_ema = window_v_loss
            self.learning_progress = float("inf")
        else:
            self.value_loss_ema = self.rho * prev_ema + (1.0 - self.rho) * window_v_loss
            self.learning_progress = abs(self.value_loss_ema - prev_ema) / max(abs(prev_ema), 1.0)

        self.window_rewards.append(window_reward)
        old_level = self.level
        reason = "keep"

        reward_dropping = self._is_reward_dropping()
        threshold = self.current_level.reward_threshold
        can_upgrade = (
            window_reward >= threshold
            and self.learning_progress <= self.lp_epsilon
            and safe_violation_rate <= self.safe_epsilon
        )
        should_downgrade = safe_violation_rate > self.safe_epsilon or reward_dropping

        if can_upgrade and self.level < self.max_level:
            self.level += 1
            reason = "upgrade"
        elif should_downgrade and self.level > 0:
            self.level -= 1
            reason = "downgrade_safety" if safe_violation_rate > self.safe_epsilon else "downgrade_reward"

        return CurriculumDecision(
            changed=self.level != old_level,
            old_level=old_level,
            new_level=self.level,
            reason=reason,
            window_reward=window_reward,
            value_loss_ema=float(self.value_loss_ema),
            learning_progress=float(self.learning_progress),
            safe_violation_rate=safe_violation_rate,
        )

    def _is_reward_dropping(self) -> bool:
        if len(self.window_rewards) < self.drop_patience + 1:
            return False
        recent = self.window_rewards[-(self.drop_patience + 1):]
        deltas = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        return all(delta < -self.drop_tolerance for delta in deltas)
