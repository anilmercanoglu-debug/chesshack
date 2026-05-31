"""STEP 2 gate for engine/net.py: shapes, masked-softmax sums, WDL, ckpt identity,
param counts match the spec's verified numbers."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import chess
import numpy as np
import torch

from config import DEV_NET, PROD_NET, SCALE_NET, POLICY_SIZE
from engine.net import (
    ChessNet, value_from_wdl, masked_policy, count_params,
    save_checkpoint, load_checkpoint,
)
from engine.encoding import board_to_planes, legal_mask


def test_param_counts():
    # spec's verified targets (±3% tolerance)
    for cfg, target in [(DEV_NET, 10.7e6), (PROD_NET, 24.5e6), (SCALE_NET, 45.5e6)]:
        n = count_params(ChessNet(cfg))
        rel = abs(n - target) / target
        print(f"  C={cfg.channels} B={cfg.blocks}: {n/1e6:.2f}M (target {target/1e6:.1f}M, {rel*100:.1f}%)")
        assert rel < 0.05, f"param count {n/1e6:.2f}M off from {target/1e6:.1f}M by {rel*100:.1f}%"
    print("[params] dev/prod/scale param counts match spec")


def test_forward_shapes_and_sums():
    net = ChessNet(DEV_NET).eval()
    x = torch.randn(4, DEV_NET.n_planes, 8, 8)
    with torch.no_grad():
        p, wdl = net(x)
    assert p.shape == (4, POLICY_SIZE), p.shape
    assert wdl.shape == (4, 3), wdl.shape
    v = value_from_wdl(wdl)
    assert v.shape == (4,)
    assert (v >= -1).all() and (v <= 1).all(), "value out of [-1,1]"
    # WDL softmax sums to 1
    probs = torch.softmax(wdl, dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-5)
    print("[forward] shapes OK, value in [-1,1], WDL sums to 1")


def test_masked_policy_sums_to_one():
    net = ChessNet(DEV_NET).eval()
    boards = [chess.Board(), chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")]
    x = torch.from_numpy(np.stack([board_to_planes(b) for b in boards]))
    masks = torch.from_numpy(np.stack([legal_mask(b) for b in boards]))
    with torch.no_grad():
        p, _ = net(x)
    probs = masked_policy(p, masks)
    sums = probs.sum(-1)
    assert torch.allclose(sums, torch.ones(len(boards)), atol=1e-5), sums
    # zero mass on illegal moves
    assert (probs[~masks] < 1e-8).all(), "nonzero prob on illegal move"
    print("[mask] masked policy sums to 1, illegal moves get 0 mass")


def test_checkpoint_identity():
    net = ChessNet(DEV_NET).eval()
    x = torch.randn(2, DEV_NET.n_planes, 8, 8)
    with torch.no_grad():
        p0, w0 = net(x)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt.pt"
        save_checkpoint(path, net)
        net2, meta = load_checkpoint(path, expect_cfg=DEV_NET)
        net2.eval()
        with torch.no_grad():
            p1, w1 = net2(x)
    assert torch.allclose(p0, p1, atol=1e-6) and torch.allclose(w0, w1, atol=1e-6)
    assert meta["format_version"] == 1
    print("[ckpt] save/load round-trip identical, config asserted")


def test_cpu_latency():
    net = ChessNet(DEV_NET).eval()
    for bs in (1, 16, 64):
        x = torch.randn(bs, DEV_NET.n_planes, 8, 8)
        with torch.no_grad():
            net(x)  # warmup
            t = time.time()
            for _ in range(5):
                net(x)
            dt = (time.time() - t) / 5
        print(f"  CPU forward bs={bs:3d}: {dt*1000:.1f} ms ({bs/dt:.0f} pos/s)")


if __name__ == "__main__":
    test_param_counts()
    test_forward_shapes_and_sums()
    test_masked_policy_sums_to_one()
    test_checkpoint_identity()
    test_cpu_latency()
    print("\nALL NET GATES PASSED ✅")
