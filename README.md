# B7 SAGIN — Joint caching + offloading (PPO)

Minimal bundle: SAGIN simulation core + a single-agent PPO line that jointly learns **offload** (local / neighbor / satellite / drop) and **cache policy** (four discrete placement heuristics per step).

## Contents

| File | Role |
|------|------|
| `sagin_env.py`, `uav.py`, `satellite.py`, `iot_region.py` | Environment |
| `communication_model.py`, `content_popularity_predictor.py` | Helpers |
| `co_env.py` | RL wrapper (obs / reward / fixed IoT+OFDM heuristics) |
| `joint_ppo_agent.py` | Shared encoder + offload/cache heads + value (PPO) |
| `train_co.py` | Training entry |
| `verify_env.py` | Quick learnability check (optional) |
| `watch_rewards_plot.py` | Re-plot `co_ppo_rewards.npy` (optional) |

## Requirements

- Python **3.8+** recommended (3.9+ tested)
- `numpy`, `matplotlib`, `torch`

## Train

```bash
cd SAGIN_Framework_Task_offloading_Content_caching
python train_co.py --episodes 500 --steps 50 --plot-every 5
```

- Checkpoints / curves / `npy` are gitignored; use `--resume` to continue after `co_ppo_checkpoint.pt` exists.
- Live curve: `--plot-every N` refreshes `co_ppo_reward_curve.png` and `co_ppo_rewards.npy` every N episodes.

## Verify environment (optional)

```bash
python verify_env.py
```

## Packaging

Artifacts (`*.pt`, `*.npy`, `*.png`, logs) are ignored by git; delete any leftovers before zipping. To shrink the archive, exclude the `.git` folder when creating the zip.
