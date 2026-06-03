"""SF-anchored Elo. Play our engine (net + MCTS) against the Stockfish UCI_Elo ladder,
then fit our rating by maximum likelihood treating the rungs as fixed anchors.

Above the UCI_Elo cap (3190) use fixed-nodes full-strength Stockfish — that is how passing
2900 (and approaching full SF) is verified. Appends results to bench/elo_history.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import torch

from config import BENCH, BENCH_DIR
from engine.net import load_checkpoint
from bench.arena import MCTSPlayer, StockfishPlayer, play_match


def _expected(r_us: float, r_opp: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_opp - r_us) / 400.0))


def fit_elo(results: List[Tuple[float, int, float]]) -> Tuple[float, float]:
    """results = [(anchor_elo, n_games, our_score_fraction)]. Returns (elo, approx_2sigma)."""
    def neg_ll(r: float) -> float:
        ll = 0.0
        for a, n, s in results:
            e = min(max(_expected(r, a), 1e-6), 1 - 1e-6)
            ll += n * (s * math.log(e) + (1 - s) * math.log(1 - e))
        return -ll

    # coarse-to-fine 1-D minimization over a plausible range
    lo, hi = 0.0, 4000.0
    best = min((x for x in range(int(lo), int(hi) + 1, 5)), key=lambda r: neg_ll(float(r)))
    grid = [best + d for d in range(-5, 6)]
    r_hat = min(grid, key=lambda r: neg_ll(float(r)))
    # crude CI from total game count (logistic SE ~ 400/sqrt(N)/ln10 near 50%)
    total = sum(n for _, n, _ in results)
    two_sigma = 2 * 400.0 / (math.sqrt(max(total, 1)) * math.log(10))
    return float(r_hat), float(two_sigma)


def measure_elo(net, device: str = "cpu", rungs=BENCH.elo_ladder, games_per_rung: int = 20,
                sims: int = BENCH.sims, leaf_batch: int = 16, sf_movetime: float = 0.05,
                verbose: bool = True, openings=None) -> dict:
    us = MCTSPlayer(net, sims=sims, device=device, leaf_batch=leaf_batch, temperature=0.0)
    results: List[Tuple[float, int, float]] = []
    detail = []
    for rung in rungs:
        sf = StockfishPlayer(elo=rung, movetime=sf_movetime)
        try:
            score, d = play_match(us, sf, games_per_rung, seed=rung, openings=openings)
        finally:
            sf.close()
        results.append((float(rung), games_per_rung, score))
        detail.append({"rung": rung, "score": score, **d})
        if verbose:
            print(f"  vs SF {rung}: score {score:.3f}  (W{d['wins']}/D{d['draws']}/L{d['losses']})")
        # early stop: once we're losing badly to a rung, higher rungs are pointless
        if score < 0.15 and len(results) >= 2:
            break
    elo, ci = fit_elo(results)
    return {"elo": elo, "ci": ci, "sims": sims, "rungs": detail}


def append_history(record: dict, path: Path = None) -> None:
    path = path or (BENCH_DIR / "elo_history.json")
    hist = []
    if path.exists():
        hist = json.loads(path.read_text())
    hist.append(record)
    path.write_text(json.dumps(hist, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="data/nets/distilled.pt")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=BENCH.sims)
    ap.add_argument("--rungs", type=int, nargs="*", default=None)
    ap.add_argument("--sf-movetime", type=float, default=0.05)
    ap.add_argument("--openings", type=str, default=None,
                    help="file of opening FENs (one per line); diverse starts -> distinct games per rung")
    ap.add_argument("--step", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, _ = load_checkpoint(args.ckpt, map_location=dev)
    net = net.to(dev)
    rungs = args.rungs or BENCH.elo_ladder
    openings = [ln.strip() for ln in open(args.openings) if ln.strip()] if args.openings else None
    if openings:
        print(f"[elo] {len(openings)} opening positions from {args.openings}")
    t = time.time()
    res = measure_elo(net, device=dev, rungs=rungs, games_per_rung=args.games,
                      sims=args.sims, sf_movetime=args.sf_movetime, openings=openings)
    print(f"\nEstimated Elo: {res['elo']:.0f} ± {res['ci']:.0f}  "
          f"(sims={args.sims}, {time.time()-t:.0f}s)")
    append_history({"step": args.step, "elo": res["elo"], "ci": res["ci"],
                    "sims": args.sims, "rungs": res["rungs"]})
