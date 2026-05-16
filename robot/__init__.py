"""
RL Trading Robot — Qwen2.5-0.5B backbone + REINFORCE policy head.

Note: conda's LightGBM links its own OpenMP. Set KMP_DUPLICATE_LIB_OK=TRUE
before importing torch to avoid an abort on macOS. This module sets it
automatically at import time.

Architecture:
  Qwen2.5-0.5B (frozen encoder)  →  mean-pooled embedding
  →  trainable MLP policy head   →  action probabilities (down / hold / up)

Training:
  REINFORCE on 3 years of historical price windows.
  Reward = +1 correct direction, -1 wrong, 0 flat/hold.
  Only the 3-layer MLP head is updated; Qwen weights stay frozen.

Inference:
  Full pipeline state serialized as text → Qwen → policy head → decision.
"""
import os
# conda's LightGBM ships its own OpenMP; without this flag torch aborts on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
