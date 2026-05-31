"""Shard format + torch Dataset, shared by Phase-1 distill shards AND Phase-2 replay.

We store, per position: the FEN (exact, tiny) + the sparse top-K policy (the MultiPV moves
and their softmax weights) + the WDL probability triple. The DENSE training target is
reconstructed on the fly from the FEN: 0.92 mass on the K stored moves (by their weights),
0.08 spread uniformly over the OTHER legal moves. Planes are encoded on the fly via the one
canonical engine.encoding (no precision loss, no duplicated encoder, shards stay tiny and
inspectable). Encoding is CPU-cheap relative to GPU training and runs in DataLoader workers.

A shard is a compressed .npz:
  fens   : object[str]   [N]
  polidx : int16         [N, K]   (-1 padding)
  polw   : float16       [N, K]   (weights over the stored moves, sum<=1; 0 for padding)
  wdl    : float16       [N, 3]   (win, draw, loss) from Stockfish, POV side-to-move
manifest.json lists shards + the exact label settings for reproducibility.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from config import POLICY_SIZE, DISTILL
from engine.encoding import board_to_planes, move_to_index

K = DISTILL.multipv  # stored moves per position


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def write_shard(out_dir: Path, shard_idx: int,
                fens: Sequence[str], polidx: np.ndarray, polw: np.ndarray, wdl: np.ndarray) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"shard_{shard_idx:06d}.npz"
    np.savez_compressed(
        out_dir / name,
        fens=np.array(fens, dtype=object),
        polidx=polidx.astype(np.int16),
        polw=polw.astype(np.float16),
        wdl=wdl.astype(np.float16),
    )
    return name


def update_manifest(out_dir: Path, shards: List[str], label_settings: dict, count: int) -> None:
    out_dir = Path(out_dir)
    manifest = {"shards": shards, "count": count, "label_settings": label_settings}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def read_manifest(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "manifest.json").read_text())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ChessDataset(Dataset):
    """Map-style dataset over one or more shard dirs. Holds the (small) FEN+label arrays
    in RAM; encodes planes + reconstructs the dense policy target per __getitem__."""

    def __init__(self, data_dir, topk_mass: float = DISTILL.topk_mass):
        self.topk_mass = float(topk_mass)
        fens, polidx, polw, wdl = [], [], [], []
        man = read_manifest(data_dir)
        for shard in man["shards"]:
            z = np.load(Path(data_dir) / shard, allow_pickle=True)
            fens.append(z["fens"])
            polidx.append(z["polidx"])
            polw.append(z["polw"])
            wdl.append(z["wdl"])
        self.fens = np.concatenate(fens)
        self.polidx = np.concatenate(polidx).astype(np.int64)
        self.polw = np.concatenate(polw).astype(np.float32)
        self.wdl = np.concatenate(wdl).astype(np.float32)
        assert len(self.fens) == len(self.polidx) == len(self.wdl)

    def __len__(self) -> int:
        return len(self.fens)

    def _dense_policy(self, board: chess.Board, idx_row: np.ndarray, w_row: np.ndarray) -> np.ndarray:
        target = np.zeros(POLICY_SIZE, dtype=np.float32)
        legal = list(board.legal_moves)
        legal_idx = np.fromiter((move_to_index(m, board) for m in legal), dtype=np.int64,
                                count=len(legal))
        stored = idx_row[idx_row >= 0]
        stored_w = w_row[idx_row >= 0]
        wsum = float(stored_w.sum())
        if wsum > 0:
            target[stored] = self.topk_mass * (stored_w / wsum)
        # spread the remaining mass uniformly over the OTHER legal moves
        others = np.setdiff1d(legal_idx, stored, assume_unique=False)
        if len(others) > 0:
            rem = 1.0 - self.topk_mass if wsum > 0 else 1.0
            target[others] = rem / len(others)
        # normalize (guards rounding / the wsum==0 edge case)
        s = target.sum()
        if s > 0:
            target /= s
        return target

    def __getitem__(self, i: int):
        board = chess.Board(self.fens[i])
        planes = board_to_planes(board)
        policy = self._dense_policy(board, self.polidx[i], self.polw[i])
        wdl = self.wdl[i]
        s = wdl.sum()
        wdl = wdl / s if s > 0 else np.array([0, 1, 0], dtype=np.float32)
        return (
            torch.from_numpy(planes),
            torch.from_numpy(policy),
            torch.from_numpy(wdl.astype(np.float32)),
        )
