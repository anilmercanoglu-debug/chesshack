"""Canonical board <-> tensor and move <-> policy-index encoding.

EVERYTHING (both training phases + bench) goes through this one module. A single
off-by-one here silently poisons both phases, so tests/test_encoding.py is a HARD GATE.

Board encoding: 19x8x8 float32, ALWAYS from the side-to-move's point of view. When it
is Black to move we rank-flip the board (square ^ 56) and swap "my"/"opponent" colors,
so the network only ever sees "me to move, moving up the board".

Move encoding: AlphaZero 8x8x73 = 4672. For a from-square `f` (in POV coords) and a
move "type" plane `t`, index = f * 73 + t.
  planes  0..55 : queen-like moves = 8 directions x 7 distances
  planes 56..63 : 8 knight moves
  planes 64..72 : 9 underpromotions = 3 pieces (N,B,R) x 3 file-directions (-1,0,+1)
Queen promotions are encoded as the matching queen-like move (decode adds promotion=QUEEN
when a pawn lands on the last rank via a queen plane).
"""
from __future__ import annotations

from typing import Optional

import chess
import numpy as np

from config import N_PLANES, POLICY_SIZE, N_MOVE_TYPES

# Queen directions (file_delta, rank_delta): N, NE, E, SE, S, SW, W, NW
QUEEN_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
# Knight deltas (file_delta, rank_delta), fixed order
KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
# Underpromotion pieces and file-directions
UNDER_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
UNDER_FILE_DIRS = [-1, 0, 1]

_QDIR_INDEX = {d: i for i, d in enumerate(QUEEN_DIRS)}
_KNIGHT_INDEX = {d: i for i, d in enumerate(KNIGHT_DELTAS)}


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _pov(square: int, white_to_move: bool) -> int:
    """Map an absolute square to the side-to-move POV frame (rank-flip if Black)."""
    return square if white_to_move else (square ^ 56)


# ---------------------------------------------------------------------------
# Board -> planes
# ---------------------------------------------------------------------------
def board_to_planes(board: chess.Board) -> np.ndarray:
    """Return a [19,8,8] float32 tensor from the side-to-move POV."""
    stm = board.turn
    white = stm == chess.WHITE
    planes = np.zeros((N_PLANES, 8, 8), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        psq = _pov(sq, white)
        r, f = divmod(psq, 8)
        base = 0 if piece.color == stm else 6
        planes[base + piece.piece_type - 1, r, f] = 1.0

    planes[12, :, :] = 1.0 if white else 0.0
    if board.has_kingside_castling_rights(stm):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(stm):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(not stm):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(not stm):
        planes[16, :, :] = 1.0
    if board.ep_square is not None:
        esq = _pov(board.ep_square, white)
        er, ef = divmod(esq, 8)
        planes[17, er, ef] = 1.0
    planes[18, :, :] = min(board.halfmove_clock, 100) / 100.0
    return planes


# ---------------------------------------------------------------------------
# Move -> index
# ---------------------------------------------------------------------------
def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Map a (legal) move to its policy index in [0, 4672), in side-to-move POV."""
    white = board.turn == chess.WHITE
    f_pov = _pov(move.from_square, white)
    t_pov = _pov(move.to_square, white)
    ff, fr = f_pov % 8, f_pov // 8
    tf, tr = t_pov % 8, t_pov // 8
    df, dr = tf - ff, tr - fr

    promo = move.promotion
    if promo is not None and promo != chess.QUEEN:
        # underpromotion: rank always advances +1 in POV; file delta in {-1,0,+1}
        dir_idx = UNDER_FILE_DIRS.index(df)
        piece_idx = UNDER_PIECES.index(promo)
        plane = 64 + piece_idx * 3 + dir_idx
        return f_pov * N_MOVE_TYPES + plane

    if (abs(df), abs(dr)) in ((1, 2), (2, 1)):
        plane = 56 + _KNIGHT_INDEX[(df, dr)]
        return f_pov * N_MOVE_TYPES + plane

    # queen-like move (includes queen promotion)
    dir_idx = _QDIR_INDEX[(_sign(df), _sign(dr))]
    dist = max(abs(df), abs(dr))
    plane = dir_idx * 7 + (dist - 1)
    return f_pov * N_MOVE_TYPES + plane


# ---------------------------------------------------------------------------
# Index -> move
# ---------------------------------------------------------------------------
def index_to_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """Inverse of move_to_index. Returns None if the index lands off-board."""
    white = board.turn == chess.WHITE
    f_pov, plane = divmod(index, N_MOVE_TYPES)
    ff, fr = f_pov % 8, f_pov // 8

    promo: Optional[int] = None
    if plane < 56:
        dir_idx, dist = divmod(plane, 7)
        dist += 1
        dfu, dru = QUEEN_DIRS[dir_idx]
        df, dr = dfu * dist, dru * dist
    elif plane < 64:
        df, dr = KNIGHT_DELTAS[plane - 56]
    else:
        u = plane - 64
        piece_idx, dir_idx = divmod(u, 3)
        df, dr = UNDER_FILE_DIRS[dir_idx], 1
        promo = UNDER_PIECES[piece_idx]

    tf, tr = ff + df, fr + dr
    if not (0 <= tf < 8 and 0 <= tr < 8):
        return None
    t_pov = tr * 8 + tf

    from_sq = _pov(f_pov, white)
    to_sq = _pov(t_pov, white)

    if promo is None:
        # queen-plane move by a pawn reaching the last rank -> queen promotion
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in (0, 7):
            promo = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion=promo)


# ---------------------------------------------------------------------------
# Legal mask
# ---------------------------------------------------------------------------
def legal_mask(board: chess.Board) -> np.ndarray:
    """Boolean [4672] mask, True at the index of each legal move."""
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    for mv in board.legal_moves:
        mask[move_to_index(mv, board)] = True
    return mask


def legal_indices(board: chess.Board) -> dict:
    """Map {policy_index: chess.Move} for the legal moves of `board`."""
    return {move_to_index(mv, board): mv for mv in board.legal_moves}
