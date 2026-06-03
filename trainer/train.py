"""Mode-parameterized trainer. --mode distill (Phase 1) | selfplay (Phase 2).

Loss = policy KL (soft target over legal moves) + value WDL cross-entropy + L2 (AdamW wd).
The legal mask is derived from the target (legal moves have >0 mass, illegal exactly 0),
so the Dataset need not carry a separate mask. bf16 autocast + channels_last on CUDA.

Phase-2 warm start: --init-from data/nets/distilled.pt (config asserted on load).

Usage:
  python -m trainer.train --mode distill --data data/distill_dev --net dev --steps 500
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DEV_NET, PROD_NET, SCALE_NET, NETS_DIR
from engine.net import ChessNet, masked_log_softmax, save_checkpoint, load_checkpoint, count_params
from trainer.dataset import ChessDataset


def policy_value_loss(policy_logits, wdl_logits, policy_target, wdl_target):
    legal = policy_target > 0
    logp = masked_log_softmax(policy_logits, legal)
    loss_p = -(policy_target * logp).sum(dim=1).mean()
    logw = F.log_softmax(wdl_logits, dim=1)
    loss_v = -(wdl_target * logw).sum(dim=1).mean()
    # policy top-1: does the net's best legal move match the target's best move?
    with torch.no_grad():
        masked = policy_logits.masked_fill(~legal, -1e9)
        top1 = (masked.argmax(1) == policy_target.argmax(1)).float().mean()
    return loss_p, loss_v, top1


def train(mode: str, data_dir: str, net_cfg, steps: int, batch: int, lr: float,
          init_from: str | None, out_path: Path, num_workers: int = 4,
          log_every: int = 50):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = dev == "cuda"
    print(f"[train] mode={mode} device={dev} net=C{net_cfg.channels}/B{net_cfg.blocks} "
          f"params={count_params(ChessNet(net_cfg))/1e6:.1f}M")

    if init_from:
        net, _ = load_checkpoint(init_from, map_location=dev, expect_cfg=net_cfg)
        print(f"[train] warm-started from {init_from}")
    else:
        net = ChessNet(net_cfg)
    net = net.to(dev)
    if use_amp:
        net = net.to(memory_format=torch.channels_last)

    ds = ChessDataset(data_dir)
    print(f"[train] dataset: {len(ds)} positions")
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=num_workers,
                        drop_last=True, persistent_workers=num_workers > 0)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))

    net.train()
    step = 0
    t0 = time.time()
    pl_ema = vl_ema = acc_ema = None
    done = False
    while not done:
        for planes, policy_t, wdl_t in loader:
            planes = planes.to(dev, non_blocking=True)
            if use_amp:
                planes = planes.to(memory_format=torch.channels_last)
            policy_t = policy_t.to(dev, non_blocking=True)
            wdl_t = wdl_t.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                p_logits, w_logits = net(planes)
                loss_p, loss_v, top1 = policy_value_loss(p_logits, w_logits, policy_t, wdl_t)
                loss = loss_p + loss_v
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 4.0)
            opt.step()
            sched.step()

            a = 0.05
            pl_ema = loss_p.item() if pl_ema is None else (1 - a) * pl_ema + a * loss_p.item()
            vl_ema = loss_v.item() if vl_ema is None else (1 - a) * vl_ema + a * loss_v.item()
            acc_ema = top1.item() if acc_ema is None else (1 - a) * acc_ema + a * top1.item()
            step += 1
            if step % log_every == 0:
                rate = step * batch / (time.time() - t0)
                print(f"[train] step {step:5d}/{steps}  pl={pl_ema:.3f} vl={vl_ema:.3f} "
                      f"top1={acc_ema:.3f}  lr={sched.get_last_lr()[0]:.2e}  ({rate:.0f} pos/s)")
            if step >= steps:
                done = True
                break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out_path, net.to("cpu"),
                    extra={"mode": mode, "steps": step, "final_pl": pl_ema, "final_vl": vl_ema})
    print(f"[train] saved {out_path}  (final pl={pl_ema:.3f} vl={vl_ema:.3f} top1={acc_ema:.3f})")
    return {"pl": pl_ema, "vl": vl_ema, "top1": acc_ema}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["distill", "selfplay"], default="distill")
    ap.add_argument("--data", type=str, default="data/distill_dev")
    ap.add_argument("--net", choices=["dev", "prod", "scale"], default="dev")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--init-from", type=str, default=None)
    ap.add_argument("--out", type=str, default=str(NETS_DIR / "distilled.pt"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = {"dev": DEV_NET, "prod": PROD_NET, "scale": SCALE_NET}[args.net]
    train(args.mode, args.data, cfg, args.steps, args.batch, args.lr,
          args.init_from, Path(args.out), num_workers=args.workers)
