"""
Qwen-based trading policy network.

Architecture:
  Qwen2.5-0.5B  (FROZEN — provides rich text understanding)
      ↓ mean-pool last hidden states
  Linear(hidden_size → 256) → LayerNorm → GELU → Dropout(0.15)
      ↓
  Linear(256 → 64) → GELU
      ↓
  Linear(64 → 3)   → softmax  →  [P(down), P(hold), P(up)]

Only the MLP policy head is trained via REINFORCE.
Qwen weights are never updated, keeping memory/compute requirements low.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default small model — user can override via MODEL_NAME env var or argument
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
N_ACTIONS = 3  # down=0, hold=1, up=2


class PolicyHead(nn.Module):
    """Trainable MLP head that maps Qwen embedding → action logits."""

    def __init__(self, hidden_size: int, mid_dim: int = 256, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(mid_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TradingPolicy(nn.Module):
    """Full policy: frozen Qwen encoder + trainable PolicyHead."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        super().__init__()
        self.model_name = model_name
        self.device     = device
        self._encoder_loaded = False

        # Lazy-load encoder on first call to avoid import cost at module load
        self.tokenizer = None
        self.encoder   = None
        self.head: Optional[PolicyHead] = None

    def _ensure_loaded(self) -> None:
        if self._encoder_loaded:
            return
        from transformers import AutoModel, AutoTokenizer  # type: ignore
        print(f"     Loading {self.model_name} (first run: downloading weights ~1 GB)...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.encoder = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.to(self.device)

        hidden_size = self.encoder.config.hidden_size
        self.head   = PolicyHead(hidden_size).to(self.device)
        self._encoder_loaded = True
        print(f"     Encoder hidden size: {hidden_size} | Policy head params: "
              f"{sum(p.numel() for p in self.head.parameters()):,}")

    # ── Encoding ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        self._ensure_loaded()
        tokens = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        ).to(self.device)
        out = self.encoder(**tokens)
        # Mean-pool over sequence length, ignoring padding
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return emb  # [1, hidden_size]

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, text: str) -> torch.Tensor:
        """Returns logits [1, N_ACTIONS]."""
        self._ensure_loaded()
        emb = self.encode(text)
        return self.head(emb)

    def action_probs(self, text: str) -> torch.Tensor:
        logits = self.forward(text)
        return F.softmax(logits, dim=-1)

    def sample_action(self, text: str) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action from policy. Returns (action_idx, log_prob, probs)."""
        probs = self.action_probs(text)
        dist  = torch.distributions.Categorical(probs)
        act   = dist.sample()
        return int(act.item()), dist.log_prob(act), probs.detach()

    def greedy_action(self, text: str) -> tuple[int, torch.Tensor]:
        """Argmax action for inference. Returns (action_idx, probs)."""
        probs = self.action_probs(text)
        return int(probs.argmax(dim=-1).item()), probs.detach()

    # ── Serialisation ──────────────────────────────────────────────────────

    def head_state_dict(self) -> dict:
        self._ensure_loaded()
        return self.head.state_dict()

    def load_head_state(self, state: dict) -> None:
        self._ensure_loaded()
        self.head.load_state_dict(state)
