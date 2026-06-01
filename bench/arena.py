"""Match runner + player adapters. One `play(board)->move` protocol so MCTS, raw-net,
and Stockfish all plug into the same arena (used by search_gain, gate, and elo)."""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import chess
import chess.engine

from config import STOCKFISH, MCTS as MCTS_CFG
from engine.player import Player, RawPolicyPlayer


class MCTSPlayer:
    def __init__(self, net, sims: int = MCTS_CFG.sims_bench, device: str = "cpu",
                 leaf_batch: int = MCTS_CFG.leaf_batch, temperature: float = 0.0):
        self.player = Player(net, device, sims, leaf_batch)
        self.temperature = temperature

    def play(self, board: chess.Board, rng=None) -> Optional[chess.Move]:
        return self.player.choose(board, temperature=self.temperature, rng=rng)[0]

    def close(self):
        pass


class RawNetPlayer:
    def __init__(self, net, device: str = "cpu", temperature: float = 0.0):
        self.p = RawPolicyPlayer(net, device)
        self.temperature = temperature

    def play(self, board: chess.Board, rng=None) -> Optional[chess.Move]:
        return self.p.choose(board, temperature=self.temperature, rng=rng)

    def close(self):
        pass


class StockfishPlayer:
    """Stockfish at a capped strength (UCI_Elo ladder) or full strength (nodes/movetime)."""

    def __init__(self, elo: Optional[int] = None, nodes: Optional[int] = None,
                 movetime: Optional[float] = None, threads: int = 1, hash_mb: int = 64):
        self.eng = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
        cfg = {"Threads": threads, "Hash": hash_mb}
        if elo is not None:
            cfg["UCI_LimitStrength"] = True
            # Stockfish only accepts UCI_Elo in [1320, 3190]; clamp to stay valid.
            cfg["UCI_Elo"] = max(1320, min(3190, int(elo)))
        self.eng.configure(cfg)
        if nodes is not None:
            self.limit = chess.engine.Limit(nodes=nodes)
        elif movetime is not None:
            self.limit = chess.engine.Limit(time=movetime)
        else:
            self.limit = chess.engine.Limit(nodes=100_000)

    def play(self, board: chess.Board, rng=None) -> Optional[chess.Move]:
        return self.eng.play(board, self.limit).move

    def close(self):
        self.eng.quit()


# A handful of short opening lines to decorrelate games (alternating colors covers the rest).
OPENINGS: List[List[str]] = [
    [], ["e2e4"], ["d2d4"], ["e2e4", "c7c5"], ["d2d4", "g8f6"],
    ["e2e4", "e7e5"], ["c2c4"], ["g1f3", "d7d5"], ["e2e4", "e7e6"], ["d2d4", "d7d5"],
]


def play_game(white, black, opening: Optional[List[str]] = None, max_plies: int = 320,
              rng=None) -> float:
    """Return result from White's POV: 1.0 win, 0.5 draw, 0.0 loss."""
    board = chess.Board()
    for uci in (opening or []):
        board.push(chess.Move.from_uci(uci))
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        mover = white if board.turn == chess.WHITE else black
        mv = mover.play(board, rng=rng)
        if mv is None or mv not in board.legal_moves:
            break
        board.push(mv)
        plies += 1
    res = board.result(claim_draw=True)
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(res, 0.5)


def play_match(player_a, player_b, n_games: int, seed: int = 0, max_plies: int = 320
               ) -> Tuple[float, dict]:
    """Alternate colors. Return (player_a score fraction, detail dict)."""
    rng = random.Random(seed)
    nrng = __import__("numpy").random.default_rng(seed)
    score_a = 0.0
    w = d = l = 0
    for i in range(n_games):
        opening = OPENINGS[i % len(OPENINGS)]
        if i % 2 == 0:
            r = play_game(player_a, player_b, opening, max_plies, rng=nrng)
            sa = r
        else:
            r = play_game(player_b, player_a, opening, max_plies, rng=nrng)
            sa = 1.0 - r
        score_a += sa
        w += sa == 1.0
        d += sa == 0.5
        l += sa == 0.0
    return score_a / n_games, {"wins": w, "draws": d, "losses": l, "games": n_games}
