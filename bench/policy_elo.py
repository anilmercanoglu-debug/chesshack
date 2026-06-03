"""SF-anchored Elo of the RAW POLICY (no search): the net's argmax move vs the Stockfish
UCI_Elo ladder. This measures the prior/policy head in isolation — the quantity self-play is
supposed to raise. Compare against elo.py (MCTS@sims) to see how much search adds (search_gain)
and whether policy gains transfer. Reuses elo.py's fit/anchor logic; same history file.

    python -m bench.policy_elo --ckpt data/nets/champion.pt --games 40
"""
from __future__ import annotations

import argparse
import time
from typing import List, Tuple

import torch

from config import BENCH
from engine.net import load_checkpoint
from bench.arena import RawNetPlayer, StockfishPlayer, play_match
from bench.elo import fit_elo, append_history


def measure_policy_elo(net, device: str = "cpu", rungs=BENCH.elo_ladder,
                       games_per_rung: int = 20, sf_movetime: float = 0.05,
                       verbose: bool = True, openings=None) -> dict:
    us = RawNetPlayer(net, device=device, temperature=0.0)
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
        if score < 0.15 and len(results) >= 2:
            break
    elo, ci = fit_elo(results)
    return {"elo": elo, "ci": ci, "sims": 0, "rungs": detail}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="data/nets/champion.pt")
    ap.add_argument("--games", type=int, default=20)
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
        print(f"[policy_elo] {len(openings)} opening positions from {args.openings}")
    t = time.time()
    res = measure_policy_elo(net, device=dev, rungs=rungs, games_per_rung=args.games,
                             sf_movetime=args.sf_movetime, openings=openings)
    print(f"\nRAW POLICY Elo: {res['elo']:.0f} ± {res['ci']:.0f}  "
          f"(no search, {time.time()-t:.0f}s)")
    append_history({"step": args.step, "elo": res["elo"], "ci": res["ci"],
                    "sims": 0, "rungs": res["rungs"], "kind": "raw_policy"})
