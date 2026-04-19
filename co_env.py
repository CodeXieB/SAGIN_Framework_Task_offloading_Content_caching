"""
co_env.py — Joint Caching + Offloading RL environment wrapper.

Single-agent centralized control over all UAVs.
Only exposes caching and offloading decisions; IoT aggregation / OFDM
are handled by fixed heuristics inside the wrapper so the RL agent
can focus on the joint caching-offloading problem.
"""
from __future__ import annotations

import builtins
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sagin_env import SAGINEnv, SystemDownException  # noqa: E402

_real_print = builtins.print


@contextmanager
def _mute():
    builtins.print = lambda *a, **kw: None
    try:
        yield
    finally:
        builtins.print = _real_print


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_OFFLOAD_ACTIONS = 4   # local / neighbor / satellite / drop
NUM_CACHE_ACTIONS = 4     # no-op / most-useful / smallest / most-popular
OBS_PER_UAV = 11          # features per UAV (see _build_obs)


class JointCacheOffloadEnv:
    """Gym-like wrapper: obs, act, reward for joint caching + offloading."""

    def __init__(
        self,
        grid: Tuple[int, int] = (3, 3),
        duration: int = 300,
        cache_size: int = 40,
        compute_power_uav: int = 25,
        compute_power_sat: int = 200,
        energy: int = 80000,
        max_queue: int = 15,
        num_sats: int = 2,
        num_iot_per_region: int = 20,
        max_active_iot: int = 10,
        ofdm_slots: int = 6,
        steps_per_episode: int = 50,
        reward_mode: str = "baseline",
        alpha1: float = 1.0,
        alpha2: float = 0.3,
        alpha3: float = 0.1,
        beta1: float = 1.0,
        beta2: float = 1.0,
        beta3: float = 1.0,
        lambda0: float = 0.2,
        k: float = 1.0,
        eta: float = 1.5,
        gamma: float = 0.25,
        cmax: float = 0.35,
    ):
        self.grid = grid
        self.steps_per_episode = steps_per_episode
        self.reward_mode = reward_mode.lower()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.lambda0 = lambda0
        self.k = k
        self.eta = eta
        self.gamma = gamma
        self.cmax = cmax
        with _mute():
            self.env = SAGINEnv(
                X=grid[0], Y=grid[1], duration=duration,
                cache_size=cache_size, compute_power_uav=compute_power_uav,
                compute_power_sat=compute_power_sat, energy=energy,
                max_queue=max_queue, num_sats=num_sats,
                num_iot_per_region=num_iot_per_region,
                max_active_iot=max_active_iot, ofdm_slots=ofdm_slots,
            )
        self.num_uavs = grid[0] * grid[1]
        self.coords = sorted(self.env.uavs.keys())  # List[Tuple[int,int]]

        # Dimensions exposed to the agent
        self.obs_dim = self.num_uavs * OBS_PER_UAV
        self.offload_action_dim = NUM_OFFLOAD_ACTIONS
        self.cache_action_dim = NUM_CACHE_ACTIONS

        # Per-step bookkeeping
        self._step_idx = 0
        self._prev_cache_hits = {}   # {coord: int}
        self._prev_completed = {}    # {coord: int}
        self._prev_dropped = 0
        self._cached_tasks = {}      # {coord: list}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._step_idx = 0
        with _mute():
            for u in self.env.uavs.values():
                u.cache_storage = {}
                u.cache_used_mb = 0.0
                u.aggregated_content = {}
                u.queue = []
                u.next_available_time = 0
                u.energy_used_this_slot = 0.0
                u.energy = u.max_energy
                u.total_tasks = 0
                u.cache_hits = 0
                u.tasks_completed_within_bound = 0
            for s in self.env.sats:
                s.task_queue = []
                s.local_storage = {}
                s.storage_used_mb = 0.0
                s.tasks_received = 0
                s.tasks_completed = 0
                s.tasks_within_bound = 0
            self.env.global_satellite_content_pool = {}
            self.env.subchannel_assignments = {}
            self.env.connected_uavs = set()
            self.env.g_timestep = -1
            self.env.dropped_tasks = 0
            self.env.task_stats = {"uav": {}, "satellite": {}}
        self._prev_cache_hits = {c: 0 for c in self.coords}
        self._prev_completed = {c: 0 for c in self.coords}
        self._prev_dropped = 0
        self._cached_tasks = {}
        return self._build_obs()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        feats = []
        for c in self.coords:
            u = self.env.uavs[c]
            r = self.env.iot_regions[c]
            energy_ratio = u.energy / max(u.max_energy, 1.0)
            queue_ratio = len(u.queue) / max(u.max_queue, 1)
            cache_hit_rate = u.cache_hits / max(u.total_tasks, 1)
            cache_usage_ratio = u.cache_used_mb / max(u.cache_capacity_mb, 1.0)

            tasks = self._cached_tasks.get(c, [])
            active_task_count = len(tasks) / 20.0
            avg_cpu = np.mean([t["required_cpu"] for t in tasks]) / 10.0 if tasks else 0.0
            avg_delay = np.mean([t["delay_bound"] for t in tasks]) / 15.0 if tasks else 0.0

            # Candidate content = aggregated + cache
            candidates = {**u.aggregated_content, **u.cache_storage}
            cand_count = len(candidates) / 50.0
            cand_avg_size = (np.mean([m.get("size", 1.0) for m in candidates.values()]) / 5.0
                            if candidates else 0.0)
            cand_avg_useful = (np.mean([u.content_popularity.get(cid, 0)
                                        for cid in candidates]) / 10.0
                               if candidates else 0.0)

            zipf_param = getattr(r, "current_zipf_param", 1.5) / 3.0

            feats.extend([
                energy_ratio,
                queue_ratio,
                cache_hit_rate,
                cache_usage_ratio,
                active_task_count,
                avg_cpu,
                avg_delay,
                cand_count,
                cand_avg_size,
                cand_avg_useful,
                zipf_param,
            ])
        return np.asarray(feats, dtype=np.float32)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self,
        offload_action: int,
        cache_action: int,
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one environment step.

        Args:
            offload_action: int in [0, NUM_OFFLOAD_ACTIONS)
            cache_action:   int in [0, NUM_CACHE_ACTIONS)
        Returns:
            obs, reward, done, info
        """
        self._step_idx += 1
        done = self._step_idx >= self.steps_per_episode

        self._last_failed_offloads = 0

        with _mute():
            # --- Phase 1: IoT aggregation (heuristic — greedy all) ---
            self._do_iot_aggregation()

            # --- Phase 2: OFDM allocation (heuristic — round-robin) ---
            self._do_ofdm_heuristic()

            # --- Phase 3: Generate tasks ---
            self._cached_tasks = {
                c: u.generate_tasks(self.env.X, self.env.Y, self._step_idx)
                for c, u in self.env.uavs.items()
            }

            # --- Phase 4: Apply offloading action ---
            self._apply_offload(offload_action)

            # --- Phase 5: Apply caching action ---
            self._apply_cache(cache_action)

            # --- Phase 6: Environment dynamics ---
            for s in self.env.sats:
                s.update_coverage(self._step_idx)
            for fn in [self.env.upload_to_satellites,
                       self.env.sync_satellites,
                       self.env.execute_all_tasks,
                       self.env.evict_expired_content]:
                try:
                    fn()
                except Exception:
                    pass

            # Reset per-slot energy counters
            for u in self.env.uavs.values():
                u.energy_used_this_slot = 0.0

        reward, info = self._compute_reward()
        obs = self._build_obs()
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Internal: IoT aggregation (fixed heuristic)
    # ------------------------------------------------------------------
    def _do_iot_aggregation(self):
        """Select ALL active devices and aggregate their content."""
        for c in self.coords:
            u = self.env.uavs[c]
            r = self.env.iot_regions[c]
            active = r.sample_active_devices()
            if active:
                content_list = r.generate_content(active, self._step_idx, grid_coord=c)
                u.aggregated_content = {tuple(ct["id"]): ct for ct in content_list}
            else:
                u.aggregated_content = {}

    # ------------------------------------------------------------------
    # Internal: OFDM heuristic (round-robin top-N by queue length)
    # ------------------------------------------------------------------
    def _do_ofdm_heuristic(self):
        self.env.subchannel_assignments = {c: {} for c in self.coords}
        self.env.connected_uavs = set()
        # Tie-break when all queues are empty: shuffle so satellite is not always
        # assigned to the same fixed subset of UAVs (stable sort artifact).
        shuffled = list(self.coords)
        np.random.shuffle(shuffled)
        ranked = sorted(
            shuffled,
            key=lambda c: len(self.env.uavs[c].queue),
            reverse=True,
        )
        sat_load: Dict[int, int] = defaultdict(int)
        n = 0
        for c in ranked:
            if n >= self.env.ofdm_slots:
                break
            best_sat = min(self.env.sats, key=lambda s: sat_load[s.sat_id])
            self.env.subchannel_assignments[c][best_sat.sat_id] = True
            sat_load[best_sat.sat_id] += 1
            self.env.connected_uavs.add(c)
            n += 1

    # ------------------------------------------------------------------
    # Internal: Offloading
    # ------------------------------------------------------------------
    _OFFLOAD_CATS = ["local", "neighbor", "satellite", "drop"]

    def _apply_offload(self, action_idx: int):
        """Apply the same offloading strategy to every UAV's tasks."""
        target = self._OFFLOAD_CATS[action_idx]
        failed = 0
        for c in self.coords:
            tasks = self._cached_tasks.get(c, [])
            for task in tasks:
                if not self._exec_offload(task, c, target):
                    failed += 1
        self._last_failed_offloads = failed

    def _exec_offload(self, task: dict, coord: Tuple[int, int], target: str) -> bool:
        if target == "local":
            return self.env.uavs[coord].receive_task(task, from_coord=coord)
        elif target == "neighbor":
            x, y = coord
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.env.uavs:
                    nb = self.env.uavs[(nx, ny)]
                    if len(nb.queue) < nb.max_queue:
                        if nb.receive_task(task, from_coord=coord):
                            return True
            return False
        elif target == "satellite":
            for s in self.env.sats:
                if self.env.subchannel_assignments.get(coord, {}).get(s.sat_id, False):
                    s.receive_task(task, from_coord=self.env.uavs[coord].uav_pos)
                    return True
            return False
        else:  # drop
            return False

    # ------------------------------------------------------------------
    # Internal: Caching
    # ------------------------------------------------------------------
    def _apply_cache(self, action_idx: int):
        """Apply caching strategy to every UAV."""
        for c in self.coords:
            u = self.env.uavs[c]
            candidates = {**u.aggregated_content, **u.cache_storage}
            if action_idx == 0:
                # No-op: keep current cache, just add aggregated if fits
                self._cache_greedy_fill(u)
            elif action_idx == 1:
                # Most useful (highest popularity score)
                self._cache_by_key(u, candidates,
                                   key=lambda cid: u.content_popularity.get(cid, 0),
                                   reverse=True)
            elif action_idx == 2:
                # Smallest first (pack more items)
                self._cache_by_key(u, candidates,
                                   key=lambda cid: candidates[cid].get("size", 999),
                                   reverse=False)
            elif action_idx == 3:
                # Popularity-density: favor high popularity per MB (distinct from idx 1)
                def score(cid):
                    pop = float(u.content_popularity.get(cid, 0))
                    sz = max(float(candidates[cid].get("size", 1.0)), 0.1)
                    return pop / sz

                self._cache_by_key(u, candidates, key=score, reverse=True)

    @staticmethod
    def _cache_greedy_fill(u):
        """Keep existing cache, greedily add aggregated content."""
        for cid, meta in u.aggregated_content.items():
            sz = meta.get("size", 1.0)
            if u.cache_used_mb + sz <= u.cache_capacity_mb:
                u.cache_storage[cid] = meta
                u.cache_used_mb += sz

    @staticmethod
    def _cache_by_key(u, candidates, key, reverse: bool):
        """Rebuild cache from candidates sorted by *key*."""
        sorted_cids = sorted(candidates.keys(), key=key, reverse=reverse)
        new_cache = {}
        used = 0.0
        for cid in sorted_cids:
            sz = candidates[cid].get("size", 1.0)
            if used + sz <= u.cache_capacity_mb:
                new_cache[cid] = candidates[cid]
                used += sz
        u.cache_storage = new_cache
        u.cache_used_mb = used

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _collect_reward_terms(self) -> dict:
        completed = 0
        new_hits = 0
        for c in self.coords:
            u = self.env.uavs[c]
            dc = u.tasks_completed_within_bound - self._prev_completed.get(c, 0)
            completed += max(dc, 0)
            self._prev_completed[c] = u.tasks_completed_within_bound

            dh = u.cache_hits - self._prev_cache_hits.get(c, 0)
            new_hits += max(dh, 0)
            self._prev_cache_hits[c] = u.cache_hits

        # env.dropped_tasks is not updated on this RL path (only full sagin_env step).
        dropped_env = max(self.env.dropped_tasks - self._prev_dropped, 0)
        self._prev_dropped = self.env.dropped_tasks
        failed = int(getattr(self, "_last_failed_offloads", 0))
        dropped = dropped_env + failed

        energy_ratio = np.mean([
            1.0 - u.energy / max(u.max_energy, 1.0)
            for u in self.env.uavs.values()
        ])
        queue_ratio = np.mean([
            len(u.queue) / max(u.max_queue, 1)
            for u in self.env.uavs.values()
        ])

        scale = float(max(self.num_uavs, 1))
        return {
            "completed": completed,
            "new_cache_hits": new_hits,
            "dropped": dropped,
            "failed_offloads": failed,
            "energy_ratio": energy_ratio,
            "queue_ratio": queue_ratio,
            "C_t": completed / scale,
            "H_t": new_hits / scale,
            "D_t": dropped / scale,
            "E_t": float(energy_ratio),
            "Q_t": float(queue_ratio),
            "F_t": failed / scale,
        }

    def _compute_perf_reward(self, terms: dict) -> float:
        return (
            self.alpha1 * terms["C_t"]
            + self.alpha2 * terms["H_t"]
            - self.alpha3 * terms["D_t"]
        )

    def _compute_safe_cost(self, terms: dict) -> float:
        return (
            self.beta1 * terms["E_t"]
            + self.beta2 * terms["Q_t"]
            + self.beta3 * terms["F_t"]
        )

    def _compute_baseline_reward(self, terms: dict) -> float:
        reward = (
            1.0 * terms["completed"]
            + 0.3 * terms["new_cache_hits"]
            - 0.1 * max(terms["dropped"], 0)
            - 0.05 * terms["energy_ratio"]
            - 0.05 * terms["queue_ratio"]
        )
        return reward / max(self.num_uavs, 1)

    def _compute_safe_reward_a(self, r_perf: float, c_safe: float) -> Tuple[float, float, bool]:
        lambda_t = self.lambda0 + self.k * max(0.0, c_safe - self.cmax)
        violated = c_safe > self.cmax
        if violated:
            reward = -self.eta * c_safe
        else:
            reward = r_perf - lambda_t * c_safe
        return reward, lambda_t, violated

    def _compute_safe_reward_b(self, r_perf: float, c_safe: float) -> Tuple[float, float, bool]:
        lambda_t = self.lambda0 + self.k * max(0.0, c_safe - self.cmax)
        violated = c_safe > self.cmax
        if violated:
            reward = self.gamma * r_perf - self.eta * (c_safe - self.cmax)
        else:
            reward = r_perf - lambda_t * c_safe
        return reward, lambda_t, violated

    def _compute_reward(self) -> Tuple[float, dict]:
        terms = self._collect_reward_terms()
        r_perf = self._compute_perf_reward(terms)
        c_safe = self._compute_safe_cost(terms)

        if self.reward_mode == "safe_a":
            reward, lambda_t, violated = self._compute_safe_reward_a(r_perf, c_safe)
        elif self.reward_mode in {"safe_b", "safe_c"}:
            reward, lambda_t, violated = self._compute_safe_reward_b(r_perf, c_safe)
        else:
            reward = self._compute_baseline_reward(terms)
            lambda_t = self.lambda0 + self.k * max(0.0, c_safe - self.cmax)
            violated = c_safe > self.cmax

        info = dict(terms)
        info.update({
            "r_perf": r_perf,
            "c_safe": c_safe,
            "lambda_t": lambda_t,
            "constraint_violated": int(violated),
            "reward_mode": self.reward_mode,
        })
        return reward, info
