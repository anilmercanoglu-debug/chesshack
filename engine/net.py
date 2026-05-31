"""Policy/value network: pre-activation ResNet + Squeeze-Excite, two heads.

Policy head: Conv1x1(C->73) -> flatten -> 4672 logits (cheap; no dead Linear).
Value head:  Conv1x1(C->32) -> BN -> ReLU -> FC(2048->256) -> ReLU -> FC(256->3) WDL logits.
Scalar value v = softmax(wdl)[win] - softmax(wdl)[loss] in [-1,1].

One ChessNet(NetConfig); dev/prod/scale differ only by config. Checkpoints carry the
NetConfig and the loader asserts it, so a Phase-2 warm start can never silently drift.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NetConfig, N_PLANES, POLICY_SIZE, N_MOVE_TYPES

FORMAT_VERSION = 1
NEG_INF = -1e9


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))               # global average pool -> [N,C]
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    """Pre-activation residual block with a Squeeze-Excite gate."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.se = SEBlock(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.bn1(x), inplace=True))
        h = self.conv2(F.relu(self.bn2(h), inplace=True))
        h = self.se(h)
        return x + h


class ChessNet(nn.Module):
    def __init__(self, cfg: NetConfig = NetConfig()):
        super().__init__()
        self.cfg = cfg
        C = cfg.channels
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.n_planes, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(C, cfg.se_reduction) for _ in range(cfg.blocks)])

        # Policy head: Conv1x1(C->73) -> flatten -> 4672
        self.policy_conv = nn.Conv2d(C, N_MOVE_TYPES, 1)

        # Value head: Conv1x1(C->32) -> BN -> ReLU -> FC -> ReLU -> FC(3)
        self.value_conv = nn.Conv2d(C, cfg.value_conv, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(cfg.value_conv)
        self.value_fc1 = nn.Linear(cfg.value_conv * 64, cfg.value_hidden)
        self.value_fc2 = nn.Linear(cfg.value_hidden, 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.body(self.stem(x))
        p = self.policy_conv(x).flatten(1)              # [N, 4672]
        v = F.relu(self.value_bn(self.value_conv(x)), inplace=True).flatten(1)
        v = F.relu(self.value_fc1(v), inplace=True)
        wdl = self.value_fc2(v)                          # [N, 3] (win, draw, loss) logits
        return p, wdl


def value_from_wdl(wdl_logits: torch.Tensor) -> torch.Tensor:
    """Scalar value in [-1,1] = P(win) - P(loss)."""
    p = F.softmax(wdl_logits, dim=-1)
    return p[..., 0] - p[..., 2]


def masked_log_softmax(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """log-softmax over legal moves only (illegal -> -inf). `legal_mask` is bool [N,4672]."""
    masked = policy_logits.masked_fill(~legal_mask, NEG_INF)
    return F.log_softmax(masked, dim=-1)


def masked_policy(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """softmax probabilities over legal moves only."""
    return masked_log_softmax(policy_logits, legal_mask).exp()


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(path, net: ChessNet, extra: Optional[dict] = None) -> None:
    ckpt = {
        "format_version": FORMAT_VERSION,
        "net_config": asdict(net.cfg),
        "state_dict": net.state_dict(),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path, map_location="cpu", expect_cfg: Optional[NetConfig] = None
                    ) -> Tuple[ChessNet, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = NetConfig(**ckpt["net_config"])
    if expect_cfg is not None:
        assert cfg == expect_cfg, f"checkpoint config {cfg} != expected {expect_cfg}"
    net = ChessNet(cfg)
    net.load_state_dict(ckpt["state_dict"])
    return net, ckpt
