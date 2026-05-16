"""
REINFORCE training loop for the RL trading policy.

Algorithm  : REINFORCE with exponential moving-average baseline subtraction
             (reduces variance without introducing bias)
Epochs     : multiple passes over all historical sliding-window episodes
Loss       : -log π(a|s) × (R - baseline)
Optimizer  : Adam (only PolicyHead parameters)
Gradient   : clipped at 1.0 norm
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from robot.environment import TradingEnvironment
from robot.model import TradingPolicy

CHECKPOINT_DIR  = Path(__file__).parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "trading_policy.pt"
ACTIONS_STR     = {0: "down", 1: "hold", 2: "up"}


def train(
    ticker:         str   = "BZ=F",
    horizon_weeks:  int   = 8,
    history_years:  int   = 3,
    n_epochs:       int   = 4,
    lr:             float = 3e-4,
    ema_alpha:      float = 0.05,   # EMA smoothing for baseline
    model_name:     str   = "Qwen/Qwen2.5-0.5B",
    device:         str   = "cpu",
) -> TradingPolicy:
    """Train TradingPolicy on historical data. Returns the trained policy."""

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # ── Environment ────────────────────────────────────────────────────────
    env = TradingEnvironment(ticker=ticker, horizon_weeks=horizon_weeks,
                             history_years=history_years)
    n_prices = env.load()
    episodes = env.episode_indices(step=2)
    print(f"     History: {n_prices} weekly bars → {len(episodes)} training windows")

    # ── Policy ─────────────────────────────────────────────────────────────
    policy = TradingPolicy(model_name=model_name, device=device)
    policy._ensure_loaded()
    policy.train()

    optimizer = optim.Adam(policy.head.parameters(), lr=lr)

    baseline  = 0.0          # EMA reward baseline for variance reduction
    best_acc  = 0.0
    best_state: dict = {}

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, n_epochs + 1):
        np.random.shuffle(episodes)
        rewards_epoch: list[float] = []
        correct:       list[bool]  = []

        for idx in episodes:
            state_text = env.state_text(idx)
            action, log_prob, _ = policy.sample_action(state_text)
            r = env.reward(idx, action)

            # Baseline-subtracted REINFORCE
            advantage = r - baseline
            baseline  = (1 - ema_alpha) * baseline + ema_alpha * r

            loss = -log_prob * advantage
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.head.parameters(), 1.0)
            optimizer.step()

            rewards_epoch.append(r)
            future_pct = env.future_return_pct(idx)
            correct.append(
                (action == 2 and future_pct >  1.5) or
                (action == 0 and future_pct < -1.5) or
                (action == 1 and abs(future_pct) <= 1.5)
            )

        mean_r = float(np.mean(rewards_epoch))
        dir_acc = float(np.mean(correct))

        # Action distribution
        action_counts = {0: 0, 1: 0, 2: 0}
        for i, idx in enumerate(episodes):
            a, _, _ = policy.sample_action(env.state_text(idx))
            action_counts[a] += 1

        print(
            f"     Epoch {epoch}/{n_epochs} | reward={mean_r:+.3f} | "
            f"dir_acc={dir_acc:.1%} | "
            f"down={action_counts[0]} hold={action_counts[1]} up={action_counts[2]}"
        )

        if dir_acc > best_acc:
            best_acc   = dir_acc
            best_state = {k: v.clone() for k, v in policy.head.state_dict().items()}

    # Restore best checkpoint
    if best_state:
        policy.head.load_state_dict(best_state)

    # ── Save checkpoint ────────────────────────────────────────────────────
    torch.save({
        "policy_head": best_state or policy.head_state_dict(),
        "ticker":         ticker,
        "horizon_weeks":  horizon_weeks,
        "n_epochs":       n_epochs,
        "n_windows":      len(episodes),
        "best_dir_acc":   best_acc,
        "model_name":     model_name,
    }, CHECKPOINT_PATH)
    print(f"     Checkpoint saved → {CHECKPOINT_PATH.name} "
          f"(best dir_acc={best_acc:.1%})")

    return policy


def load_checkpoint(model_name: str = "Qwen/Qwen2.5-0.5B", device: str = "cpu") -> tuple[TradingPolicy, dict]:
    """Load a previously saved checkpoint. Returns (policy, meta)."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"No checkpoint at {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    policy = TradingPolicy(model_name=model_name, device=device)
    policy._ensure_loaded()
    policy.load_head_state(ckpt["policy_head"])
    policy.eval()
    return policy, ckpt
