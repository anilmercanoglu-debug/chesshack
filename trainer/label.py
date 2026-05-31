"""Phase-1 Stockfish labeler — 16 parallel persistent engines.

For each FEN: analyse at MultiPV=K, nodes=100k, full strength. Build the sparse policy
target (the K best moves with softmax-over-cp weights) and the WDL value target (from the
best line's POV score). Append compact shards; write a manifest. Resumable (skips shards
already on disk by index).

Usage:
  python -m trainer.label --n 5000 --out data/distill --seed 1
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import chess
import chess.engine
import numpy as np

from config import STOCKFISH, DISTILL, DISTILL_DIR
from engine.encoding import move_to_index
from trainer.dataset import write_shard, update_manifest, K
from trainer.positions import generate_positions

_ENGINE: Optional[chess.engine.SimpleEngine] = None


def _init_worker():
    global _ENGINE
    _ENGINE = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    _ENGINE.configure({
        "Threads": DISTILL.sf_threads,
        "Hash": DISTILL.sf_hash_mb,
        "UCI_ShowWDL": True,
    })


def _label_one(fen: str) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Return (fen, polidx[K], polw[K], wdl[3]) or None for terminal/failed positions."""
    global _ENGINE
    board = chess.Board(fen)
    if board.is_game_over():
        return None
    try:
        infos = _ENGINE.analyse(board, chess.engine.Limit(nodes=DISTILL.sf_nodes),
                                multipv=DISTILL.multipv)
    except chess.engine.EngineError:
        return None
    if not infos:
        return None

    idx = np.full(K, -1, dtype=np.int16)
    cps = np.full(K, -1e9, dtype=np.float64)
    turn = board.turn
    for j, info in enumerate(infos[:K]):
        pv = info.get("pv")
        if not pv:
            continue
        mv = pv[0]
        idx[j] = move_to_index(mv, board)
        cps[j] = info["score"].pov(turn).score(mate_score=30000)

    valid = idx >= 0
    if not valid.any():
        return None
    # softmax over cp for the valid moves -> weights
    w = np.zeros(K, dtype=np.float64)
    cv = cps[valid]
    z = (cv - cv.max()) / DISTILL.tau_cp
    e = np.exp(z)
    w[valid] = e / e.sum()

    # WDL from the best line (POV side-to-move)
    best = infos[0]["score"].pov(turn)
    wdl_obj = best.wdl(ply=board.ply())
    wdl = np.array([wdl_obj.wins, wdl_obj.draws, wdl_obj.losses], dtype=np.float64)
    wdl = wdl / max(wdl.sum(), 1.0)
    return fen, idx, w.astype(np.float16), wdl.astype(np.float16)


def label_dataset(n: int, out_dir: Path, seed: int = 0, pgn_path: Optional[str] = None,
                  n_workers: int = DISTILL.n_workers) -> None:
    import multiprocessing as mp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[label] generating {n} candidate positions ...")
    fens = generate_positions(n, seed=seed, pgn_path=pgn_path)
    print(f"[label] {len(fens)} unique positions; labeling on {n_workers} workers "
          f"(nodes={DISTILL.sf_nodes}, multipv={DISTILL.multipv}) ...")

    shard_size = DISTILL.shard_size
    buf_fen, buf_idx, buf_w, buf_wdl = [], [], [], []
    shards, total = [], 0
    shard_idx = 0
    t0 = time.time()

    def flush():
        nonlocal shard_idx
        if not buf_fen:
            return
        name = write_shard(out_dir, shard_idx,
                           buf_fen, np.stack(buf_idx), np.stack(buf_w), np.stack(buf_wdl))
        shards.append(name)
        shard_idx += 1
        buf_fen.clear(); buf_idx.clear(); buf_w.clear(); buf_wdl.clear()

    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers, initializer=_init_worker) as pool:
        for res in pool.imap_unordered(_label_one, fens, chunksize=8):
            if res is None:
                continue
            fen, idx, w, wdl = res
            buf_fen.append(fen); buf_idx.append(idx); buf_w.append(w); buf_wdl.append(wdl)
            total += 1
            if len(buf_fen) >= shard_size:
                flush()
            if total % 1000 == 0:
                rate = total / (time.time() - t0)
                print(f"[label]   {total} labeled  ({rate:.0f} pos/s)")
    flush()

    settings = {
        "sf_nodes": DISTILL.sf_nodes, "multipv": DISTILL.multipv,
        "tau_cp": DISTILL.tau_cp, "topk_mass": DISTILL.topk_mass,
        "stockfish": str(STOCKFISH),
    }
    update_manifest(out_dir, shards, settings, total)
    dt = time.time() - t0
    print(f"[label] done: {total} positions in {len(shards)} shards, "
          f"{dt:.1f}s, {total/dt:.0f} pos/s -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", type=str, default=str(DISTILL_DIR))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pgn", type=str, default=None)
    ap.add_argument("--workers", type=int, default=DISTILL.n_workers)
    args = ap.parse_args()
    label_dataset(args.n, Path(args.out), seed=args.seed, pgn_path=args.pgn,
                  n_workers=args.workers)
