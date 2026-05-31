"""Phase-2 self-play RL driver (the uncapped ratchet). ONE process orchestrates:

  - InferenceServer (broker thread, holds the CHAMPION net on GPU),
  - K spawned CPU worker processes generating 600-sim self-play games into a game queue,
  - a rolling replay buffer,
  - a trainer (train_net) doing ~1 grad step per `steps_per_fresh` fresh positions,
  - a promotion GATE: periodically gate train_net vs champion; only on >=55% does the
    champion (and the server) adopt the new weights — so the generator improves monotonically,
  - anti-stall: periodic search_gain; if low, bump sims along the ladder.

Targets: policy = MCTS visit distribution (KL); value = 0.85*z + 0.15*q_root (MSE on the
WDL head's scalar projection). No Stockfish in the loss -> no teacher cap.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import time
from typing import List, Optional

import chess
import numpy as np
import torch
import torch.nn.functional as F

from config import (DEV_NET, PROD_NET, POLICY_SIZE, SELFPLAY, MCTS as MCTS_CFG, NETS_DIR)
from engine.net import ChessNet, load_checkpoint, save_checkpoint, masked_log_softmax, value_from_wdl
from engine.encoding import board_to_planes, legal_mask
from engine.inference_server import InferenceServer, ServerEvaluator
from trainer.gate import gate
from bench.search_gain import search_gain, recommend_sims

MAX_PLIES = 320


# --------------------------------------------------------------------------- #
# Worker process (module-level for spawn)
# --------------------------------------------------------------------------- #
def _selfplay_worker(worker_id, request_q, result_q, game_q, stop_evt, sims_value, seed, params):
    import numpy as _np
    import chess as _chess
    from engine.mcts import mcts_search, policy_target, pick_move, root_value
    from engine.inference_server import ServerEvaluator

    ev = ServerEvaluator(worker_id, request_q, result_q)
    rng = _np.random.default_rng(seed)
    lb = params["leaf_batch"]; temp = params["temperature"]; tplies = params["temperature_plies"]
    zw = params["zw"]; qw = params["qw"]
    cpuct = params["c_puct"]; da = params["dir_alpha"]; de = params["dir_eps"]

    while not stop_evt.is_set():
        sims = int(sims_value.value)
        board = _chess.Board()
        samples = []
        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
            if stop_evt.is_set():
                return
            root = mcts_search(board, ev, sims, c_puct=cpuct, leaf_batch=lb,
                               add_noise=True, dirichlet_alpha=da, dirichlet_eps=de, rng=rng)
            if root.terminal:
                break
            pi = policy_target(root, 1.0).astype(_np.float32)
            q = root_value(root)
            samples.append((board.fen(), root.move_idx.copy(), pi, float(q), board.turn))
            t = temp if plies < tplies else 0.0
            board.push(pick_move(root, temperature=t, rng=rng))
            plies += 1

        res = board.result(claim_draw=True)
        ws = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(res, 0.5)
        out = []
        for fen, idx, pi, q, stm in samples:
            z = (2 * ws - 1) if stm == _chess.WHITE else -(2 * ws - 1)
            out.append((fen, idx, pi, zw * z + qw * q))
        if out:
            try:
                game_q.put(out, timeout=1.0)
            except queue.Full:
                pass


# --------------------------------------------------------------------------- #
# Rolling replay buffer
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.data: List = []
        self.pos = 0

    def add(self, samples) -> None:
        for s in samples:
            if len(self.data) < self.cap:
                self.data.append(s)
            else:
                self.data[self.pos] = s
                self.pos = (self.pos + 1) % self.cap

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, n: int, rng) -> List:
        n = min(n, len(self.data))
        idxs = rng.integers(0, len(self.data), size=n)
        return [self.data[i] for i in idxs]


def _make_batch(samples, device):
    planes, targets, masks, values = [], [], [], []
    for fen, idx, pi, v in samples:
        b = chess.Board(fen)
        planes.append(board_to_planes(b))
        t = np.zeros(POLICY_SIZE, np.float32)
        t[idx] = pi
        targets.append(t)
        masks.append(legal_mask(b))
        values.append(v)
    x = torch.from_numpy(np.stack(planes)).to(device)
    tp = torch.from_numpy(np.stack(targets)).to(device)
    mask = torch.from_numpy(np.stack(masks)).to(device)
    tv = torch.tensor(values, dtype=torch.float32, device=device)
    return x, tp, mask, tv


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_selfplay(init_ckpt: str, net_cfg, n_workers: int, total_steps: int,
                 device: str = "cpu", batch: int = 256, lr: float = 5e-4,
                 sims: int = SELFPLAY.sims, min_samples: int = 2000,
                 gate_every: int = 2000, gate_games: int = SELFPLAY.gate_games,
                 sg_every: int = 2000, sg_games: int = 40,
                 capacity: int = SELFPLAY.replay_capacity,
                 steps_per_fresh: float = SELFPLAY.steps_per_fresh,
                 max_batch: int = 256, out_dir=NETS_DIR, log_every: int = 100):
    ctx = mp.get_context("spawn")
    champion, _ = load_checkpoint(init_ckpt, map_location=device, expect_cfg=net_cfg)
    champion = champion.to(device).eval()
    train_net = ChessNet(net_cfg).to(device)
    train_net.load_state_dict(champion.state_dict())

    server = InferenceServer(champion, device, n_workers, ctx,
                             max_batch=max_batch, max_wait_ms=2.0)
    game_q = ctx.Queue(maxsize=n_workers * 4)
    stop = ctx.Event()
    sims_value = ctx.Value("i", int(sims))
    params = dict(leaf_batch=MCTS_CFG.leaf_batch, temperature=SELFPLAY.temperature,
                  temperature_plies=SELFPLAY.temperature_plies,
                  zw=SELFPLAY.value_z_weight, qw=SELFPLAY.value_q_weight,
                  c_puct=MCTS_CFG.c_puct, dir_alpha=MCTS_CFG.dirichlet_alpha,
                  dir_eps=MCTS_CFG.dirichlet_eps)
    procs = [ctx.Process(target=_selfplay_worker,
                         args=(i, server.request_q, server.result_qs[i], game_q, stop,
                               sims_value, 1234 + i, params), daemon=True)
             for i in range(n_workers)]
    for p in procs:
        p.start()
    print(f"[selfplay] {n_workers} workers, sims={sims}, device={device}, "
          f"net=C{net_cfg.channels}/B{net_cfg.blocks}")

    opt = torch.optim.AdamW(train_net.parameters(), lr=lr, weight_decay=1e-4)
    replay = ReplayBuffer(capacity)
    rng = np.random.default_rng(0)
    steps = games = 0
    fresh = 0.0
    pl_ema = vl_ema = None
    out_dir = __import__("pathlib").Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    try:
        while steps < total_steps:
            # drain whatever games are ready
            for _ in range(n_workers * 2):
                try:
                    g = game_q.get(timeout=0.05)
                except queue.Empty:
                    break
                replay.add(g)
                games += 1
                fresh += len(g)

            if len(replay) < min_samples:
                continue

            while fresh >= steps_per_fresh and steps < total_steps:
                fresh -= steps_per_fresh
                batch_s = replay.sample(batch, rng)
                x, tp, mask, tv = _make_batch(batch_s, device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                    p_logits, w_logits = train_net(x)
                    loss_p = -(tp * masked_log_softmax(p_logits, mask)).sum(1).mean()
                    loss_v = F.mse_loss(value_from_wdl(w_logits), tv)
                    loss = loss_p + loss_v
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_net.parameters(), 4.0)
                opt.step()
                steps += 1
                a = 0.02
                pl_ema = loss_p.item() if pl_ema is None else (1 - a) * pl_ema + a * loss_p.item()
                vl_ema = loss_v.item() if vl_ema is None else (1 - a) * vl_ema + a * loss_v.item()

                if steps % log_every == 0:
                    print(f"[selfplay] step {steps:6d} games={games} buf={len(replay)} "
                          f"pl={pl_ema:.3f} vl={vl_ema:.3f} sims={sims_value.value} "
                          f"avg_batch={server.avg_batch:.1f} ({games/(time.time()-t0):.2f} g/s)")

                if steps % gate_every == 0:
                    d = gate(train_net, champion, device=device, games=gate_games,
                             sims=sims_value.value)
                    if d["promote"]:
                        champion.load_state_dict(train_net.state_dict())
                        server.update_net(train_net.state_dict())
                        save_checkpoint(out_dir / "champion.pt", champion.to(device),
                                        extra={"step": steps})
                        print(f"[gate]   PROMOTED at step {steps}: candidate {d['candidate_score']:.3f} "
                              f">= {d['threshold']:.2f} (W{d['wins']}/D{d['draws']}/L{d['losses']})")
                    else:
                        print(f"[gate]   held at step {steps}: {d['candidate_score']:.3f} < {d['threshold']:.2f}")

                if steps % sg_every == 0:
                    sg = search_gain(champion, sims_value.value, n_games=sg_games, device=device)
                    wr = sg["mcts_winrate"]
                    new_sims = recommend_sims(sims_value.value, wr)
                    if new_sims != sims_value.value:
                        sims_value.value = new_sims
                        print(f"[anti-stall] search_gain {wr:.3f} low -> sims bumped to {new_sims}")
                    else:
                        print(f"[anti-stall] search_gain {wr:.3f} (sims {sims_value.value})")
    finally:
        stop.set()
        server.stop()
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=3)
        save_checkpoint(out_dir / "champion.pt", champion.to(device), extra={"step": steps})
        print(f"[selfplay] stopped at step {steps}, games={games}, "
              f"elapsed={time.time()-t0:.0f}s, avg_batch={server.avg_batch:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", type=str, default=str(NETS_DIR / "distilled.pt"))
    ap.add_argument("--net", choices=["dev", "prod"], default="dev")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--sims", type=int, default=SELFPLAY.sims)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEV_NET if args.net == "dev" else PROD_NET
    run_selfplay(args.init_from, cfg, args.workers, args.steps, device=dev,
                 batch=args.batch, sims=args.sims)
