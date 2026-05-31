"""Anti-stall monitor (the make-or-break instrument).

Measures the win-rate of MCTS(net, sims) vs the raw net's argmax-policy (no search), over a
fixed match. If search doesn't make the net meaningfully stronger, the self-play ratchet has
no teeth and Phase-2 will stall (the documented ~1200 failure). We require >= search_gain_min
(0.60); if it drops below search_gain_bump_below (0.55) the self-play loop bumps sims along
SELFPLAY.sims_ladder. This makes the MCTS>net gap observable and self-restoring.
"""
from __future__ import annotations

import argparse

import torch

from config import DEV_NET, PROD_NET, SELFPLAY
from engine.net import load_checkpoint, ChessNet
from bench.arena import MCTSPlayer, RawNetPlayer, play_match


def search_gain(net, sims: int, n_games: int = 40, device: str = "cpu",
                leaf_batch: int = 16, seed: int = 0) -> dict:
    mcts = MCTSPlayer(net, sims=sims, device=device, leaf_batch=leaf_batch, temperature=0.0)
    raw = RawNetPlayer(net, device=device, temperature=0.0)
    score, detail = play_match(mcts, raw, n_games, seed=seed)
    detail.update({"sims": sims, "mcts_winrate": score})
    return detail


def recommend_sims(current_sims: int, winrate: float) -> int:
    """Bump along the ladder if search isn't winning enough."""
    if winrate >= SELFPLAY.search_gain_bump_below:
        return current_sims
    ladder = list(SELFPLAY.sims_ladder)
    for s in ladder:
        if s > current_sims:
            return s
    return current_sims  # already at the top of the ladder -> caller scales net width


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="data/nets/distilled.pt")
    ap.add_argument("--sims", type=int, default=SELFPLAY.sims)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--leaf-batch", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, _ = load_checkpoint(args.ckpt, map_location=dev)
    net = net.to(dev)
    d = search_gain(net, args.sims, n_games=args.games, device=dev, leaf_batch=args.leaf_batch)
    wr = d["mcts_winrate"]
    print(f"search_gain @ {args.sims} sims: MCTS vs raw-net winrate = {wr:.3f} "
          f"(W{d['wins']}/D{d['draws']}/L{d['losses']})")
    if wr >= SELFPLAY.search_gain_min:
        print(f"  >= {SELFPLAY.search_gain_min:.2f} target — ratchet has teeth")
    elif wr >= SELFPLAY.search_gain_bump_below:
        print(f"  below {SELFPLAY.search_gain_min:.2f} target but >= {SELFPLAY.search_gain_bump_below:.2f} — watch, no bump yet")
    else:
        print(f"  < {SELFPLAY.search_gain_bump_below:.2f} — bump sims to {recommend_sims(args.sims, wr)}")
