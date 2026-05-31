"""PUCT MCTS with leaf-parallel batched evaluation and virtual loss.

Sign convention (negamax): every node's per-child W/Q is from THAT node's side-to-move
POV. A leaf value is from the leaf's POV; backup negates once per ply going up. The net
value is already POV (encoding is side-to-move relative), so signs stay consistent.

`leaf_batch=1` is exactly serial MCTS (no virtual loss) — the reference. `leaf_batch>1`
collects several leaves per wave using virtual loss to diversify paths, evaluates them in
one batched forward, then backs them all up. Validated equal-ish in tests/test_mcts.py.

evaluate_fn(boards) -> (probs[B,4672] softmaxed over legal moves, values[B] in [-1,1]).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import chess
import numpy as np

from config import MCTS as MCTS_CFG
from engine.encoding import move_to_index

EvaluateFn = Callable[[List[chess.Board]], Tuple[np.ndarray, np.ndarray]]


class Node:
    __slots__ = ("board", "terminal", "value", "moves", "move_idx",
                 "P", "N", "W", "children", "expanded")

    def __init__(self, board: chess.Board):
        self.board = board
        self.expanded = False
        self.value = 0.0
        if board.is_checkmate():
            self.terminal, self.value, self.moves = True, -1.0, []
        elif board.is_stalemate() or board.is_insufficient_material() \
                or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            self.terminal, self.value, self.moves = True, 0.0, []
        else:
            self.terminal = False
            self.moves = list(board.legal_moves)
            n = len(self.moves)
            self.move_idx = np.fromiter((move_to_index(m, board) for m in self.moves),
                                        dtype=np.int64, count=n)
            self.P = np.zeros(n, dtype=np.float32)
            self.N = np.zeros(n, dtype=np.float32)
            self.W = np.zeros(n, dtype=np.float32)
            self.children: List[Optional[Node]] = [None] * n


def _expand(node: Node, probs: np.ndarray, value: float) -> None:
    p = probs[node.move_idx]
    s = p.sum()
    node.P = (p / s).astype(np.float32) if s > 1e-12 else np.full(len(node.moves), 1.0 / len(node.moves), np.float32)
    node.value = float(value)
    node.expanded = True


def _select(node: Node, c_puct: float, fpu: float) -> int:
    sqrt_sum = float(np.sqrt(node.N.sum()))
    visited = node.N > 0
    q = np.where(visited, node.W / np.maximum(node.N, 1.0), node.value - fpu)
    u = c_puct * node.P * sqrt_sum / (1.0 + node.N)
    return int(np.argmax(q + u))


def _add_dirichlet(node: Node, alpha: float, eps: float, rng: np.random.Generator) -> None:
    noise = rng.dirichlet([alpha] * len(node.P)).astype(np.float32)
    node.P = (1 - eps) * node.P + eps * noise


def mcts_search(board: chess.Board, evaluate_fn: EvaluateFn, sims: int,
                c_puct: float = MCTS_CFG.c_puct, fpu: float = MCTS_CFG.fpu_reduction,
                leaf_batch: int = 1, virtual_loss: float = MCTS_CFG.virtual_loss,
                add_noise: bool = False, dirichlet_alpha: float = MCTS_CFG.dirichlet_alpha,
                dirichlet_eps: float = MCTS_CFG.dirichlet_eps,
                rng: Optional[np.random.Generator] = None) -> Node:
    root = Node(board.copy())
    if root.terminal:
        return root
    probs, values = evaluate_fn([root.board])
    _expand(root, probs[0], float(values[0]))
    if add_noise:
        _add_dirichlet(root, dirichlet_alpha, dirichlet_eps, rng or np.random.default_rng())

    vl = virtual_loss if leaf_batch > 1 else 0.0
    done = 0
    while done < sims:
        batch_nodes: List[Node] = []
        batch_paths: List[List[Tuple[Node, int]]] = []
        pending = set()
        target = min(leaf_batch, sims - done)
        attempts = 0
        while len(batch_nodes) < target and attempts < target * 4:
            attempts += 1
            node = root
            path: List[Tuple[Node, int]] = []
            while not node.terminal and node.expanded:
                ci = _select(node, c_puct, fpu)
                path.append((node, ci))
                if vl:
                    node.N[ci] += vl
                    node.W[ci] -= vl
                child = node.children[ci]
                if child is None:
                    nb = node.board.copy()
                    nb.push(node.moves[ci])
                    child = Node(nb)
                    node.children[ci] = child
                node = child
            # node is a leaf: terminal, or unexpanded
            if node.terminal:
                _backup(path, node.value)
                _undo_vloss(path, vl)
                done += 1
                if done >= sims:
                    break
                continue
            if id(node) in pending:
                _undo_vloss(path, vl)  # collision: another descent already owns this leaf
                continue
            pending.add(id(node))
            batch_nodes.append(node)
            batch_paths.append(path)

        if batch_nodes:
            probs, values = evaluate_fn([n.board for n in batch_nodes])
            for k, node in enumerate(batch_nodes):
                _expand(node, probs[k], float(values[k]))
                _backup(batch_paths[k], node.value)
                _undo_vloss(batch_paths[k], vl)
                done += 1
    return root


def _backup(path: List[Tuple[Node, int]], leaf_value: float) -> None:
    v = leaf_value
    for node, ci in reversed(path):
        v = -v
        node.W[ci] += v
        node.N[ci] += 1


def _undo_vloss(path: List[Tuple[Node, int]], vl: float) -> None:
    if not vl:
        return
    for node, ci in path:
        node.N[ci] -= vl
        node.W[ci] += vl


# ---------------------------------------------------------------------------
# Move selection from a searched root
# ---------------------------------------------------------------------------
def visit_counts(root: Node) -> np.ndarray:
    return root.N.copy()


def policy_target(root: Node, temperature: float = 1.0) -> np.ndarray:
    """Normalized visit distribution over legal moves (the Phase-2 policy target)."""
    n = root.N
    if temperature <= 1e-6:
        out = np.zeros_like(n)
        out[int(np.argmax(n))] = 1.0
        return out
    x = n ** (1.0 / temperature)
    s = x.sum()
    return (x / s) if s > 0 else np.full_like(n, 1.0 / len(n))


def pick_move(root: Node, temperature: float = 0.0,
              rng: Optional[np.random.Generator] = None) -> chess.Move:
    if temperature <= 1e-6:
        return root.moves[int(np.argmax(root.N))]
    p = policy_target(root, temperature)
    rng = rng or np.random.default_rng()
    return root.moves[int(rng.choice(len(root.moves), p=p))]


def root_value(root: Node) -> float:
    """Visit-weighted value of the root position from its side-to-move POV."""
    if root.N.sum() == 0:
        return root.value
    return float((root.W).sum() / root.N.sum()) * -1.0  # children W are from child POV-negated; root POV = -mean
