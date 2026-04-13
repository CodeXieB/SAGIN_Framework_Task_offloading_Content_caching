"""
joint_ppo_agent.py — Single-agent PPO with shared encoder + dual action heads.

Uses an MLP encoder (not GRU) so that act() and update() compute the same
logits for a given observation — GRU+PPO without storing hidden per step
breaks the importance ratio and prevents convergence.

Architecture:
    obs -> MLP encoder -> shared features
        -> offload_head  (4-class Categorical)
        -> cache_head    (4-class Categorical)
        -> value_head    (scalar)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Transition:
    obs: np.ndarray
    offload_action: int
    cache_action: int
    old_logprob: float
    value: float
    reward: float
    done: float


class JointPPOAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        offload_actions: int = 4,
        cache_actions: int = 4,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # MLP encoder: same function for rollout and PPO update (critical for convergence)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.proj = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh())

        self.offload_head = nn.Linear(64, offload_actions)
        self.cache_head = nn.Linear(64, cache_actions)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self._init_heads()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.rollout: List[Transition] = []
        self.to(self.device)

    def _init_heads(self) -> None:
        for m in [self.offload_head, self.cache_head]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def reset_hidden(self, batch: int = 1) -> None:
        """No-op kept for API compatibility with train_co.py."""
        pass

    def _forward(self, obs: torch.Tensor):
        """obs: (B, obs_dim)"""
        feat = self.encoder(obs)
        pol = self.proj(feat)
        off_logits = self.offload_head(pol)
        cache_logits = self.cache_head(pol)
        value = self.value_head(feat).squeeze(-1)
        return off_logits, cache_logits, value

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        off_logits, cache_logits, value = self._forward(x)
        off_logits = off_logits.squeeze(0)
        cache_logits = cache_logits.squeeze(0)

        off_dist = torch.distributions.Categorical(logits=off_logits)
        cache_dist = torch.distributions.Categorical(logits=cache_logits)

        if deterministic:
            off_a = int(off_logits.argmax().item())
            cache_a = int(cache_logits.argmax().item())
        else:
            off_a = int(off_dist.sample().item())
            cache_a = int(cache_dist.sample().item())

        logprob = (
            off_dist.log_prob(torch.tensor(off_a, device=self.device))
            + cache_dist.log_prob(torch.tensor(cache_a, device=self.device))
        )
        return off_a, cache_a, float(logprob.item()), float(value.item())

    def store(self, obs, off_a, cache_a, logprob, value, reward, done):
        self.rollout.append(
            Transition(
                obs=np.asarray(obs, dtype=np.float32),
                offload_action=off_a,
                cache_action=cache_a,
                old_logprob=logprob,
                value=value,
                reward=reward,
                done=float(done),
            )
        )

    def update(self, last_value: float = 0.0, epochs: int = 2, mini_batch: int = 64):
        if len(self.rollout) == 0:
            return {}

        obs = torch.as_tensor(
            np.stack([t.obs for t in self.rollout]),
            dtype=torch.float32,
            device=self.device,
        )
        off_acts = torch.tensor(
            [t.offload_action for t in self.rollout],
            dtype=torch.long,
            device=self.device,
        )
        cache_acts = torch.tensor(
            [t.cache_action for t in self.rollout],
            dtype=torch.long,
            device=self.device,
        )
        old_lp = torch.tensor(
            [t.old_logprob for t in self.rollout],
            dtype=torch.float32,
            device=self.device,
        )
        rewards = np.array([t.reward for t in self.rollout], dtype=np.float32)
        values = np.array([t.value for t in self.rollout], dtype=np.float32)
        dones = np.array([t.done for t in self.rollout], dtype=np.float32)

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            next_done = 0.0 if t == T - 1 else dones[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - next_done) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages[t] = gae
        returns = advantages + values

        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        # Avoid blowing up advantages when one short rollout has near-constant GAE
        # (common with T≈50); that caused occasional late-training policy "re-randomize"
        # and sharp reward dips (~ep 400 in logs).
        adv_mean = adv_t.mean()
        adv_std = adv_t.std(unbiased=False).clamp(min=0.3)
        adv_t = (adv_t - adv_mean) / adv_std

        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_ent = 0.0
        n_updates = 0

        indices = np.arange(T)
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, T, mini_batch):
                end = min(start + mini_batch, T)
                idx = indices[start:end]

                mb_obs = obs[idx]
                mb_off = off_acts[idx]
                mb_cache = cache_acts[idx]
                mb_old_lp = old_lp[idx]
                mb_adv = adv_t[idx]
                mb_ret = ret_t[idx]

                off_logits, cache_logits, vals = self._forward(mb_obs)

                off_dist = torch.distributions.Categorical(logits=off_logits)
                cache_dist = torch.distributions.Categorical(logits=cache_logits)

                new_lp = off_dist.log_prob(mb_off) + cache_dist.log_prob(mb_cache)
                ratio = torch.exp(new_lp - mb_old_lp)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                v_loss = F.mse_loss(vals, mb_ret)
                entropy = (off_dist.entropy() + cache_dist.entropy()).mean()

                loss = pg_loss + self.value_coef * v_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_ent += entropy.item()
                n_updates += 1

        self.rollout.clear()
        n = max(n_updates, 1)
        return {
            "pg_loss": total_pg_loss / n,
            "v_loss": total_v_loss / n,
            "entropy": total_ent / n,
        }
