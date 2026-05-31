"""HARD GATE for engine/encoding.py — blocks all training if it fails.

Checks over thousands of real positions:
  1. move<->index is a bijection on legal moves (encode then decode == original)
  2. legal indices are unique (no two legal moves share an index)
  3. legal_mask has exactly len(legal_moves) bits set, at the right indices
  4. board planes are POV-invariant: encode(board) == encode(board.mirror())
  5. plane sanity (piece counts, stm flag, value ranges)
"""
from __future__ import annotations

import random

import chess
import numpy as np

from config import N_PLANES, POLICY_SIZE
from engine.encoding import (
    board_to_planes, move_to_index, index_to_move, legal_mask,
)


def _random_positions(n_games: int, seed: int = 0):
    """Yield positions reached by random play from the start."""
    rng = random.Random(seed)
    for _ in range(n_games):
        board = chess.Board()
        plies = rng.randint(0, 80)
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            yield board.copy(stack=False)
        yield board.copy(stack=False)


def test_move_index_bijection_and_mask():
    n_pos = 0
    n_moves = 0
    for board in _random_positions(400, seed=1):
        n_pos += 1
        legal = list(board.legal_moves)
        seen = {}
        for mv in legal:
            idx = move_to_index(mv, board)
            assert 0 <= idx < POLICY_SIZE, f"index OOR {idx} for {mv} in {board.fen()}"
            assert idx not in seen, (
                f"index collision {idx}: {mv} vs {seen[idx]} in {board.fen()}"
            )
            seen[idx] = mv
            back = index_to_move(idx, board)
            assert back == mv, f"round-trip {mv} -> {idx} -> {back} in {board.fen()}"
            n_moves += 1
        # mask matches exactly the legal indices
        mask = legal_mask(board)
        assert mask.sum() == len(legal), f"mask count {mask.sum()} != {len(legal)}"
        for idx in seen:
            assert mask[idx], f"mask missing legal idx {idx}"
    print(f"[bijection] {n_pos} positions, {n_moves} legal moves: 100% round-trip + unique + mask OK")


def test_pov_invariance():
    """Everything except the absolute color flag (plane 12) must be POV-invariant:
    a position and its color-mirror are the same game from the mover's POV."""
    n = 0
    other = [i for i in range(N_PLANES) if i != 12]
    for board in _random_positions(300, seed=2):
        p = board_to_planes(board)
        pm = board_to_planes(board.mirror())
        assert p.shape == (N_PLANES, 8, 8)
        assert np.array_equal(p[other], pm[other]), (
            f"POV not invariant under mirror (excluding color flag): {board.fen()}"
        )
        # plane 12 is the only color-dependent plane: 1.0 if white to move else 0.0
        assert p[12].mean() == (1.0 if board.turn == chess.WHITE else 0.0)
        assert pm[12].mean() == (1.0 if board.mirror().turn == chess.WHITE else 0.0)
        assert p[12].mean() != pm[12].mean(), "color flag must flip under mirror"
        n += 1
    print(f"[pov] {n} positions: all planes POV-invariant except color flag (12) OK")


def test_plane_sanity():
    # startpos: 8 my pawns, stm flag = 1 (white), all castling, no ep, clock 0
    b = chess.Board()
    p = board_to_planes(b)
    assert p[0].sum() == 8, "8 own pawns at startpos"
    assert p[5].sum() == 1 and p[11].sum() == 1, "one king each side"
    assert p[12].mean() == 1.0, "stm-is-white flag"
    assert p[13].mean() == 1.0 and p[14].mean() == 1.0, "own castling rights"
    assert p[17].sum() == 0.0, "no en-passant at startpos"
    assert p[18].mean() == 0.0, "halfmove clock 0"
    assert 0.0 <= p.min() and p.max() <= 1.0, "planes in [0,1]"
    # after 1.e4 it is Black to move -> stm flag should be 0
    b.push_san("e4")
    p2 = board_to_planes(b)
    assert p2[12].mean() == 0.0, "stm flag 0 when black to move"
    assert p2[17].sum() == 1.0, "en-passant square set after a double pawn push"
    print("[sanity] startpos + e4 plane checks OK")


def test_promotions_roundtrip():
    # position with queen + underpromotions available
    b = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    promo_moves = [m for m in b.legal_moves if m.promotion is not None]
    assert len(promo_moves) == 4, f"expected 4 promotions, got {len(promo_moves)}"
    for mv in promo_moves:
        idx = move_to_index(mv, b)
        assert index_to_move(idx, b) == mv, f"promo round-trip failed for {mv}"
    print(f"[promo] {len(promo_moves)} promotions round-trip OK")


if __name__ == "__main__":
    test_move_index_bijection_and_mask()
    test_pov_invariance()
    test_plane_sanity()
    test_promotions_roundtrip()
    print("\nALL ENCODING GATES PASSED ✅")
