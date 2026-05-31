"""Phase-1 position SOURCING — produce a diverse set of FENs to be labeled.

The #1 risk (per SPEC) is train/play distribution mismatch: the net must see the kinds of
positions MCTS actually reaches, not just tidy opening lines. So we mix sources and stratify
by game phase. v1 implements the cheap, dependency-free sources; PGN ingestion is optional
and used when a PGN path is provided.

Sources:
  - random playouts (off-book, "ugly" positions self-play visits) — cheap, no Stockfish
  - lightly-biased playouts (prefer captures/checks sometimes) for more realistic middlegames
  - sparse-piece endgames (3-7 pieces) sampled by placing random legal material
  - PGN FENs (optional) when a .pgn[.zst] path is supplied

De-dup by FEN (piece placement + stm + castling + ep), phase-stratified, stm-balanced.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Iterator, List, Optional

import chess


def _phase(board: chess.Board) -> str:
    n = chess.popcount(board.occupied)
    if n >= 28:
        return "opening"
    if n >= 14:
        return "middle"
    return "end"


def _fen_key(board: chess.Board) -> str:
    # placement + stm + castling + ep (ignore clocks) for de-dup
    return " ".join(board.fen().split(" ")[:4])


def random_playout(rng: random.Random, max_plies: int = 80, capture_bias: float = 0.0
                   ) -> Iterator[chess.Board]:
    board = chess.Board()
    plies = rng.randint(1, max_plies)
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        if capture_bias > 0 and rng.random() < capture_bias:
            caps = [m for m in moves if board.is_capture(m) or board.gives_check(m)]
            move = rng.choice(caps) if caps else rng.choice(moves)
        else:
            move = rng.choice(moves)
        board.push(move)
        yield board.copy(stack=False)


def sparse_endgame(rng: random.Random) -> Optional[chess.Board]:
    """Place 2 kings + a few random non-king pieces on empty squares; keep if valid."""
    board = chess.Board(None)
    squares = rng.sample(range(64), 64)
    wk, bk = squares[0], squares[1]
    if chess.square_distance(wk, bk) <= 1:
        return None
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    n_extra = rng.randint(1, 5)
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    for sq in squares[2:2 + n_extra]:
        pt = rng.choice(pieces)
        color = rng.choice([chess.WHITE, chess.BLACK])
        if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
            continue
        board.set_piece_at(sq, chess.Piece(pt, color))
    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    if not board.is_valid():
        return None
    return board


def pgn_fens(pgn_path: str, max_games: int, rng: random.Random) -> Iterator[chess.Board]:
    import chess.pgn
    with open(pgn_path) as f:
        for _ in range(max_games):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                board.push(move)
                ply += 1
                if ply % 6 == 0:  # ~1 in 6 plies, capped per game
                    yield board.copy(stack=False)


def generate_positions(n: int, seed: int = 0, pgn_path: Optional[str] = None
                       ) -> List[str]:
    """Return up to `n` de-duplicated, phase-stratified, stm-balanced FENs."""
    rng = random.Random(seed)
    target = {"opening": int(0.35 * n), "middle": int(0.40 * n), "end": n}  # end gets remainder
    counts: Counter = Counter()
    stm_counts = {chess.WHITE: 0, chess.BLACK: 0}
    seen = set()
    out: List[str] = []

    def consider(board: chess.Board) -> None:
        if len(out) >= n:
            return
        key = _fen_key(board)
        if key in seen:
            return
        ph = _phase(board)
        if ph != "end" and counts[ph] >= target[ph]:
            return
        # soft stm balance
        if stm_counts[board.turn] > stm_counts[not board.turn] + n // 20 + 5:
            return
        seen.add(key)
        counts[ph] += 1
        stm_counts[board.turn] += 1
        out.append(board.fen())

    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        r = rng.random()
        if pgn_path and r < 0.40:
            for b in pgn_fens(pgn_path, 1, rng):
                consider(b)
        elif r < 0.70:
            bias = 0.0 if rng.random() < 0.5 else 0.4  # half pure-random, half capture-biased
            for b in random_playout(rng, capture_bias=bias):
                consider(b)
                if len(out) >= n:
                    break
        else:
            b = sparse_endgame(rng)
            if b is not None:
                consider(b)
    return out[:n]


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    fens = generate_positions(n, seed=1)
    phases = Counter(_phase(chess.Board(f)) for f in fens)
    stm = Counter("w" if chess.Board(f).turn else "b" for f in fens)
    print(f"generated {len(fens)} unique FENs | phases={dict(phases)} | stm={dict(stm)}")
    for f in fens[:3]:
        print(" ", f)
