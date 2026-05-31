"""Net + MCTS move chooser. Used IDENTICALLY by bench/arena and self-play so that
what we measure == what we train."""
from __future__ import annotations

from typing import List, Optional, Tuple

import chess
import numpy as np
import torch

from config import MCTS as MCTS_CFG
from engine.encoding import board_to_planes, legal_mask
from engine.net import masked_policy, value_from_wdl
from engine.mcts import mcts_search, pick_move, policy_target, Node


class NetEvaluator:
    """Batched evaluate_fn for MCTS: boards -> (legal-masked policy probs[B,4672], values[B])."""

    def __init__(self, net, device: str = "cpu"):
        self.net = net.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, boards: List[chess.Board]) -> Tuple[np.ndarray, np.ndarray]:
        planes = np.stack([board_to_planes(b) for b in boards])
        masks = np.stack([legal_mask(b) for b in boards])
        x = torch.from_numpy(planes).to(self.device)
        m = torch.from_numpy(masks).to(self.device)
        p_logits, wdl = self.net(x)
        probs = masked_policy(p_logits, m).float().cpu().numpy()
        vals = value_from_wdl(wdl).float().cpu().numpy()
        return probs, vals


class Player:
    def __init__(self, net, device: str = "cpu", sims: int = MCTS_CFG.sims_bench,
                 leaf_batch: int = MCTS_CFG.leaf_batch):
        self.eval_fn = NetEvaluator(net, device)
        self.sims = sims
        self.leaf_batch = leaf_batch

    def search(self, board: chess.Board, sims: Optional[int] = None,
               add_noise: bool = False, rng=None) -> Node:
        return mcts_search(board, self.eval_fn, sims or self.sims,
                           leaf_batch=self.leaf_batch, add_noise=add_noise, rng=rng)

    def choose(self, board: chess.Board, temperature: float = 0.0, sims: Optional[int] = None,
               add_noise: bool = False, rng=None) -> Tuple[Optional[chess.Move], Node]:
        root = self.search(board, sims=sims, add_noise=add_noise, rng=rng)
        if root.terminal:
            return None, root
        return pick_move(root, temperature, rng), root


class RawPolicyPlayer:
    """No search: plays the argmax of the net's legal policy. For raw-net Elo + search_gain."""

    def __init__(self, net, device: str = "cpu"):
        self.eval_fn = NetEvaluator(net, device)

    def choose(self, board: chess.Board, temperature: float = 0.0, rng=None
               ) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        if not moves:
            return None
        probs, _ = self.eval_fn([board])
        from engine.encoding import move_to_index
        idx = np.fromiter((move_to_index(m, board) for m in moves), dtype=np.int64, count=len(moves))
        p = probs[0][idx]
        if temperature <= 1e-6:
            return moves[int(np.argmax(p))]
        s = p.sum()
        p = p / s if s > 0 else np.full(len(moves), 1.0 / len(moves))
        rng = rng or np.random.default_rng()
        return moves[int(rng.choice(len(moves), p=p))]
