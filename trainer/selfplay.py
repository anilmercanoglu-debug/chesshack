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
import math
import multiprocessing as mp
import queue
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import chess
import numpy as np
import torch
import torch.nn.functional as F

from config import (DEV_NET, PROD_NET, SCALE_NET, START_NET, NetConfig, POLICY_SIZE, SELFPLAY,
                    MCTS as MCTS_CFG, NETS_DIR)
from engine.net import (ChessNet, load_checkpoint, save_checkpoint, masked_log_softmax,
                        value_from_wdl, grow_net, count_params)
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
    from engine.mate_search import find_forced_mate

    ev = ServerEvaluator(worker_id, request_q, result_q)
    rng = _np.random.default_rng(seed)
    lb = params["leaf_batch"]; temp = params["temperature"]; tplies = params["temperature_plies"]
    zw = params["zw"]; qw = params["qw"]
    cpuct = params["c_puct"]; da = params["dir_alpha"]; de = params["dir_eps"]
    openings = params.get("openings"); rof = float(params.get("random_open_frac", 0.0))
    mate_depth = int(params.get("mate_depth", 0)); mate_nodes = int(params.get("mate_nodes", 50000))

    while not stop_evt.is_set():
        sims = int(sims_value.value)
        board = _chess.Board()
        # seed a fraction of games from a real opening position (anti echo-chamber): the
        # forced-opening moves are NOT recorded; samples come from the self-play continuation.
        if openings and rof and rng.random() < rof:
            board = _chess.Board(openings[int(rng.integers(len(openings)))])
        samples = []
        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
            if stop_evt.is_set():
                return
            if mate_depth:                       # play a forced mate if one exists (no sample recorded;
                mm, _ = find_forced_mate(board, mate_depth, mate_nodes)   # the correct OUTCOME still
                if mm is not None:                # propagates z to the earlier MCTS-recorded positions)
                    board.push(mm); plies += 1; continue
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
def _elo_gain(score: float) -> float:
    """Implied Elo gain over the previous champion from a gate win-rate (clamped)."""
    s = min(max(score, 0.5001), 0.9999)
    return 400.0 * math.log10(s / (1.0 - s))


def run_selfplay(init_ckpt: str, net_cfg, n_workers: int, total_steps: int,
                 device: str = "cpu", batch: int = 256, lr: float = 5e-4,
                 sims: int = SELFPLAY.sims, min_samples: int = 2000,
                 gate_every_games: int = SELFPLAY.gate_every_games,
                 gate_games: int = SELFPLAY.gate_games,
                 gate_winrate: float = SELFPLAY.gate_winrate,
                 sg_every_games: int = SELFPLAY.sg_every_games,
                 sg_games: int = SELFPLAY.sg_games,
                 state_every_games: int = SELFPLAY.state_every_games,
                 capacity: int = SELFPLAY.replay_capacity,
                 steps_per_fresh: float = SELFPLAY.steps_per_fresh,
                 max_batch: int = 256, out_dir=NETS_DIR, log_every: int = 100,
                 leaf_batch: int = SELFPLAY.worker_leaf_batch,
                 base_elo: float = SELFPLAY.base_elo, resume: bool = False,
                 bench_every_promos: int = 0, bench_every_games: int = 0, grow: bool = False,
                 random_open_frac: float = 0.0, openings_path: str = None,
                 mate_depth: int = 0, mate_nodes: int = 50000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "selfplay_state.pt"

    openings = None
    if openings_path:
        openings = [ln.strip() for ln in open(openings_path) if ln.strip()]
        print(f"[selfplay] {len(openings)} opening positions from {openings_path} "
              f"(random_open_frac={random_open_frac})")

    # --- nets + optimizer ---
    if init_ckpt == "scratch":
        champion = ChessNet(net_cfg)
        print("[selfplay] starting from a RANDOM net (pure self-play, no distillation)")
    else:
        champion, _ = load_checkpoint(init_ckpt, map_location=device, expect_cfg=net_cfg)
    champion = champion.to(device).eval()
    train_net = ChessNet(net_cfg).to(device)
    train_net.load_state_dict(champion.state_dict())
    opt = torch.optim.AdamW(train_net.parameters(), lr=lr, weight_decay=1e-4)

    steps = games = promotions = 0
    elo_est = float(base_elo)
    cur_sims = int(sims)

    # --- resume from a saved training state (survives Colab restarts) ---
    # Rebuild the nets at the SAVED config so a self-grown (deeper) net resumes correctly.
    if resume and state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=False)
        net_cfg = NetConfig(**st["net_config"])
        champion = ChessNet(net_cfg).to(device).eval(); champion.load_state_dict(st["champion"])
        train_net = ChessNet(net_cfg).to(device); train_net.load_state_dict(st["train_net"])
        opt = torch.optim.AdamW(train_net.parameters(), lr=lr, weight_decay=1e-4)
        opt.load_state_dict(st["optimizer"])
        steps, games, promotions = st["step"], st["games"], st["promotions"]
        elo_est, cur_sims = st["elo_est"], st["sims"]
        print(f"[selfplay] RESUMED: step={steps} games={games} promotions={promotions} "
              f"est.Elo~{elo_est:.0f} sims={cur_sims} blocks={net_cfg.blocks}")

    ctx = mp.get_context("spawn")
    server = InferenceServer(champion, device, n_workers, ctx,
                             max_batch=max_batch, max_wait_ms=2.0)
    game_q = ctx.Queue(maxsize=n_workers * 4)
    stop = ctx.Event()
    sims_value = ctx.Value("i", cur_sims)
    params = dict(leaf_batch=leaf_batch, temperature=SELFPLAY.temperature,
                  temperature_plies=SELFPLAY.temperature_plies,
                  zw=SELFPLAY.value_z_weight, qw=SELFPLAY.value_q_weight,
                  c_puct=MCTS_CFG.c_puct, dir_alpha=MCTS_CFG.dirichlet_alpha,
                  dir_eps=MCTS_CFG.dirichlet_eps,
                  random_open_frac=random_open_frac, openings=openings,
                  mate_depth=mate_depth, mate_nodes=mate_nodes)
    procs = [ctx.Process(target=_selfplay_worker,
                         args=(i, server.request_q, server.result_qs[i], game_q, stop,
                               sims_value, 1234 + i, params), daemon=True)
             for i in range(n_workers)]
    for p in procs:
        p.start()
    print(f"[selfplay] {n_workers} workers, sims={cur_sims}, leaf_batch={leaf_batch}, "
          f"device={device}, net=C{net_cfg.channels}/B{net_cfg.blocks} | gate every "
          f"{gate_every_games} games ({gate_games} games), est.Elo~{elo_est:.0f}")

    replay = ReplayBuffer(capacity)
    rng = np.random.default_rng(0)
    fresh = 0.0
    pl_ema = vl_ema = None
    last_gate = last_sg = last_state = last_bench = last_grow = games
    consecutive_holds = 0
    t0 = time.time()
    games0 = games          # games at session start -> g/s reflects THIS session, not the resumed total

    def save_state():
        torch.save({
            "net_config": asdict(net_cfg), "champion": champion.state_dict(),
            "train_net": train_net.state_dict(), "optimizer": opt.state_dict(),
            "step": steps, "games": games, "promotions": promotions,
            "elo_est": elo_est, "sims": sims_value.value,
        }, state_path)

    try:
        while steps < total_steps:
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
                          f"pl={pl_ema:.3f} vl={vl_ema:.3f} sims={sims_value.value} est.Elo~{elo_est:.0f} "
                          f"avg_batch={server.avg_batch:.1f} ({(games-games0)/(time.time()-t0):.2f} g/s)")

            # ---- game-based gate (promotion) ----
            if games - last_gate >= gate_every_games:
                last_gate = games
                d = gate(train_net, champion, device=device, games=gate_games,
                         winrate=gate_winrate, sims=sims_value.value, openings=openings,
                         mate_depth=mate_depth)
                if d["promote"]:
                    before = elo_est
                    elo_est += _elo_gain(d["candidate_score"])
                    promotions += 1
                    consecutive_holds = 0
                    champion.load_state_dict(train_net.state_dict())
                    server.update_net(train_net.state_dict())
                    save_checkpoint(out_dir / "champion.pt", champion.to(device),
                                    extra={"step": steps, "games": games, "elo_est": elo_est})
                    save_state()
                    print(f"[gate]   PROMOTED #{promotions} @ {games} games: "
                          f"{d['candidate_score']:.3f} (W{d['wins']}/D{d['draws']}/L{d['losses']})  "
                          f"est.Elo ~{before:.0f} -> ~{elo_est:.0f}")
                    if bench_every_promos and promotions % bench_every_promos == 0:
                        _auto_bench(champion, device, games, promotions, elo_est, openings=openings)
                else:
                    consecutive_holds += 1
                    print(f"[gate]   held @ {games} games: {d['candidate_score']:.3f} "
                          f"< {d['threshold']:.2f} ({consecutive_holds} in a row)")
                    # ---- self-grow: plateau (repeated holds) -> add identity ResBlocks ----
                    if (grow and consecutive_holds >= SELFPLAY.grow_after_holds
                            and net_cfg.blocks < SELFPLAY.grow_max_blocks
                            and games - last_grow >= SELFPLAY.grow_cooldown_games):
                        add = min(SELFPLAY.grow_block_step, SELFPLAY.grow_max_blocks - net_cfg.blocks)
                        champion = grow_net(champion, add).to(device).eval()
                        train_net = grow_net(train_net, add).to(device)
                        opt = torch.optim.AdamW(train_net.parameters(), lr=lr, weight_decay=1e-4)
                        net_cfg = train_net.cfg
                        server.replace_net(champion)
                        consecutive_holds = 0
                        last_grow = games
                        print(f"[grow]   plateau -> +{add} blocks = {net_cfg.blocks} "
                              f"(~{count_params(train_net)/1e6:.1f}M) @ {games} games")

            # ---- game-based anti-stall (search_gain) ----
            if games - last_sg >= sg_every_games:
                last_sg = games
                wr = search_gain(champion, sims_value.value, n_games=sg_games, device=device)["mcts_winrate"]
                new_sims = recommend_sims(sims_value.value, wr)
                if new_sims != sims_value.value:
                    sims_value.value = new_sims
                    print(f"[anti-stall] search_gain {wr:.3f} low -> sims bumped to {new_sims}")
                else:
                    print(f"[anti-stall] search_gain {wr:.3f} (sims {sims_value.value})")

            # ---- game-based real-Elo checkup (independent of promotions) ----
            if bench_every_games and games - last_bench >= bench_every_games:
                last_bench = games
                _auto_bench(champion, device, games, promotions, elo_est, openings=openings)

            # ---- periodic full-state checkpoint (for --resume) ----
            if games - last_state >= state_every_games:
                last_state = games
                save_state()
    finally:
        stop.set()
        server.stop()
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=3)
        save_state()
        save_checkpoint(out_dir / "champion.pt", champion.to(device),
                        extra={"step": steps, "games": games, "elo_est": elo_est})
        print(f"[selfplay] stopped: step={steps} games={games} promotions={promotions} "
              f"est.Elo~{elo_est:.0f} elapsed={time.time()-t0:.0f}s avg_batch={server.avg_batch:.1f}")


def _auto_bench(net, device, games, promotions, elo_est, openings=None):
    """Optional real Elo bench on the champion (logs to elo_history.json). Wide ladder from
    1320 + early-stop: works for a weak from-scratch net (loses low -> stops fast, fits low)
    through a strong one (climbs the ladder)."""
    try:
        from bench.elo import measure_elo, append_history
        res = measure_elo(net, device=device, rungs=(1320, 1500, 1700, 1900, 2100),
                          games_per_rung=10, sims=400, verbose=False, openings=openings)
        append_history({"games": games, "promotions": promotions,
                        "elo": res["elo"], "ci": res["ci"], "elo_est": elo_est})
        print(f"[bench]   real Elo {res['elo']:.0f} ± {res['ci']:.0f} "
              f"(vs est.~{elo_est:.0f}) @ {games} games")
    except Exception as e:
        print(f"[bench]   skipped: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", type=str, default=str(NETS_DIR / "distilled.pt"),
                    help="checkpoint to start from, or 'scratch' for a random net (pure self-play)")
    ap.add_argument("--net", choices=["dev", "prod", "scale", "start"], default="dev")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--sims", type=int, default=SELFPLAY.sims)
    ap.add_argument("--gate-every-games", type=int, default=SELFPLAY.gate_every_games)
    ap.add_argument("--gate-games", type=int, default=SELFPLAY.gate_games)
    ap.add_argument("--gate-winrate", type=float, default=SELFPLAY.gate_winrate)
    ap.add_argument("--sg-every-games", type=int, default=SELFPLAY.sg_every_games)
    ap.add_argument("--leaf-batch", type=int, default=SELFPLAY.worker_leaf_batch)
    ap.add_argument("--base-elo", type=float, default=SELFPLAY.base_elo)
    ap.add_argument("--bench-every-promos", type=int, default=0)
    ap.add_argument("--bench-every-games", type=int, default=0,
                    help="run a real SF-anchored Elo checkup every N generated games")
    ap.add_argument("--grow", action="store_true",
                    help="self-grow: add identity ResBlocks when the net plateaus (gate holds)")
    ap.add_argument("--mate-depth", type=int, default=0,
                    help="forced-mate search in self-play+gate (0=off; keep small, e.g. 5-8, for speed)")
    ap.add_argument("--mate-nodes", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--capacity", type=int, default=SELFPLAY.replay_capacity)
    ap.add_argument("--openings", type=str, default=None,
                    help="file of opening FENs (one per line) to seed games from")
    ap.add_argument("--random-open-frac", type=float, default=SELFPLAY.random_open_frac,
                    help="fraction of games started from a random opening in --openings")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {"dev": DEV_NET, "prod": PROD_NET, "scale": SCALE_NET, "start": START_NET}[args.net]
    run_selfplay(args.init_from, cfg, args.workers, args.steps, device=dev,
                 batch=args.batch, lr=args.lr, sims=args.sims, capacity=args.capacity,
                 gate_every_games=args.gate_every_games,
                 gate_games=args.gate_games, gate_winrate=args.gate_winrate,
                 sg_every_games=args.sg_every_games,
                 leaf_batch=args.leaf_batch, base_elo=args.base_elo,
                 bench_every_promos=args.bench_every_promos,
                 bench_every_games=args.bench_every_games, grow=args.grow, resume=args.resume,
                 random_open_frac=args.random_open_frac, openings_path=args.openings,
                 mate_depth=args.mate_depth, mate_nodes=args.mate_nodes)
