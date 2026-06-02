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


def _open_pgn(pgn_path: str):
    """Open a .pgn or .pgn.zst for text reading. Lichess dumps are .zst and huge, so stream
    them (needs the `zstandard` package) instead of decompressing to disk."""
    if pgn_path.endswith(".zst"):
        try:
            import zstandard
        except ImportError as e:
            raise RuntimeError(
                "Reading .zst needs `pip install zstandard` (or decompress to .pgn first: "
                "`unzstd file.pgn.zst`).") from e
        import io
        dctx = zstandard.ZstdDecompressor()
        return io.TextIOWrapper(dctx.stream_reader(open(pgn_path, "rb")),
                                encoding="utf-8", errors="replace")
    return open(pgn_path, encoding="utf-8", errors="replace")


def pgn_fens(pgn_path: str, rng: random.Random, max_games: Optional[int] = None,
             per_game: int = 12, skip_plies: int = 8) -> Iterator[chess.Board]:
    """Stream FENs from a PGN, advancing through games (file opened ONCE — the old version
    reopened per game with max_games=1, re-reading game #1 forever). Skips the first
    `skip_plies` (shared opening theory → near-duplicates), then takes up to `per_game`
    positions spread evenly across the rest of EACH game, so every game contributes real
    opening, middlegame AND endgame positions (not just the first few plies)."""
    import chess.pgn
    f = _open_pgn(pgn_path)
    games = 0
    try:
        while max_games is None or games < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games += 1
            moves = list(game.mainline_moves())
            usable = len(moves) - skip_plies
            if usable <= 0:
                continue
            k = min(per_game, usable)
            # evenly-spaced ply indices across skip_plies..end, with a small per-game jitter
            jit = rng.randint(0, max(1, usable // (k + 1)))
            want = {skip_plies + min(usable - 1, jit + usable * i // k) for i in range(k)}
            board = game.board()
            for ply, move in enumerate(moves, 1):
                board.push(move)
                if ply in want:
                    yield board.copy(stack=False)
    finally:
        f.close()


def generate_positions(n: int, seed: int = 0, pgn_path: Optional[str] = None,
                       pgn_frac: float = 0.85) -> List[str]:
    """Return up to `n` de-duplicated, phase-stratified, stm-balanced FENs. When a PGN is
    given, ~`pgn_frac` of draws come from real games (the rest synthetic for off-book /
    sparse-endgame coverage); falls back fully to synthetic once the PGN is exhausted."""
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

    pgn_gen = pgn_fens(pgn_path, rng) if pgn_path else None
    pgn_done = pgn_gen is None

    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        if not pgn_done and rng.random() < pgn_frac:
            try:
                consider(next(pgn_gen))
            except StopIteration:
                pgn_done = True  # PGN exhausted -> remaining draws go to synthetic sources
            continue
        # synthetic sources: off-book playouts + sparse endgames (also fill phase buckets
        # / coverage the PGN under-supplies)
        if rng.random() < 0.7:
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
