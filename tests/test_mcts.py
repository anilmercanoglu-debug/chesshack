"""HARD GATE for engine/mcts.py: terminal scoring, mate-in-1 found, leaf_batch=1 is
exactly serial, batched (leaf_batch>1) agrees with serial on the best move."""
from __future__ import annotations

import chess
import numpy as np
import torch

from config import DEV_NET
from engine.net import ChessNet
from engine.mcts import mcts_search, pick_move, policy_target, Node
from engine.player import NetEvaluator


def _fixed_eval(seed: int = 0):
    torch.manual_seed(seed)
    net = ChessNet(DEV_NET).eval()
    return NetEvaluator(net, "cpu")


MATES = [
    "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",          # Ra8#
    "7k/5ppp/8/8/8/8/5PPP/3R3K w - - 0 1",       # Rd8#
    "3k4/R7/3K4/8/8/8/8/8 w - - 0 1",            # Ra8#  (king opposition)
]


def test_terminal_scoring():
    cm = Node(chess.Board("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"))
    assert not cm.terminal  # quiet position
    mate = Node(chess.Board("6k1/5Qpp/8/8/8/8/8/6K1 b - - 0 1"))  # black to move, is it mate?
    # construct a real checkmate: white Qg7 mates black Kg8? use a known one
    real = Node(chess.Board("6k1/6Q1/6K1/8/8/8/8/8 b - - 0 1"))  # Kg8, Qg7, Kg6 -> black checkmated
    assert real.terminal and real.value == -1.0, f"checkmate must score -1, got {real.value}"
    stale = Node(chess.Board("5k2/5P2/5K2/8/8/8/8/8 b - - 0 1"))  # black stalemated
    assert stale.terminal and stale.value == 0.0, f"stalemate must score 0, got {stale.value}"
    print("[terminal] checkmate=-1, stalemate=0 OK")


def test_mate_in_one():
    ev = _fixed_eval(0)
    for fen in MATES:
        board = chess.Board(fen)
        # sanity: a mate-in-1 truly exists
        assert any(board.is_checkmate() for board in (_after(board, m) for m in board.legal_moves)), fen
        root = mcts_search(board, ev, sims=80, leaf_batch=1)
        mv = pick_move(root, temperature=0.0)
        assert _after(board, mv).is_checkmate(), f"MCTS missed mate-in-1 in {fen}: played {mv}"
    print(f"[mate] found mate-in-1 in all {len(MATES)} positions")


def _after(board: chess.Board, move: chess.Move) -> chess.Board:
    b = board.copy()
    b.push(move)
    return b


def test_batch1_is_serial():
    """leaf_batch=1 must be deterministic and reproducible (the serial reference)."""
    ev = _fixed_eval(1)
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    r1 = mcts_search(board, ev, sims=200, leaf_batch=1)
    r2 = mcts_search(board, ev, sims=200, leaf_batch=1)
    assert np.array_equal(r1.N, r2.N), "serial MCTS not deterministic"
    assert r1.N.sum() == 200, f"expected 200 sims, got {r1.N.sum()}"
    print("[serial] leaf_batch=1 deterministic, visit total == sims")


def test_batched_agrees_with_serial():
    """Virtual-loss batching is an approximation of serial. The correctness property is
    that the VISIT DISTRIBUTIONS are close; with enough sims they converge exactly. (At low
    sims with an untrained net, values are flat so the argmax is just tie-break noise.)"""
    ev = _fixed_eval(2)
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")

    def cosine(a: Node, b: Node) -> float:
        pa, pb = policy_target(a, 1.0), policy_target(b, 1.0)
        return float(pa @ pb / (np.linalg.norm(pa) * np.linalg.norm(pb) + 1e-9))

    s400 = mcts_search(board, ev, sims=400, leaf_batch=1)
    b400 = mcts_search(board, ev, sims=400, leaf_batch=16)
    assert b400.N.sum() == 400, f"batched visit total {b400.N.sum()} != 400"
    c400 = cosine(s400, b400)
    assert c400 > 0.95, f"batched vs serial visit cosine {c400:.3f} too low at 400 sims"

    s1600 = mcts_search(board, ev, sims=1600, leaf_batch=1)
    b1600 = mcts_search(board, ev, sims=1600, leaf_batch=16)
    c1600 = cosine(s1600, b1600)
    assert c1600 > 0.98, f"cosine {c1600:.3f} should be ~1 at 1600 sims"
    assert int(np.argmax(s1600.N)) == int(np.argmax(b1600.N)), "best move must agree at 1600 sims"
    print(f"[batched] leaf_batch=16 ≈ serial: cosine {c400:.3f}@400, {c1600:.3f}@1600, "
          f"best move agrees at 1600")


def test_mate_in_one_batched():
    ev = _fixed_eval(3)
    board = chess.Board(MATES[0])
    root = mcts_search(board, ev, sims=128, leaf_batch=16)
    assert _after(board, pick_move(root, 0.0)).is_checkmate(), "batched MCTS missed mate"
    print("[batched-mate] leaf_batch=16 finds mate-in-1")


if __name__ == "__main__":
    test_terminal_scoring()
    test_mate_in_one()
    test_batch1_is_serial()
    test_batched_agrees_with_serial()
    test_mate_in_one_batched()
    print("\nALL MCTS GATES PASSED ✅")
