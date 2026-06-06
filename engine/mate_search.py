"""Sound forced-mate finder (NN-independent, pure search).

Proves a forced checkmate — mate regardless of what the defender plays — by searching CHECKING
lines: the attacker tries only moves that give check; the defender must answer EVERY legal reply.
Because the defender is always in check its replies are few (1-3), so the proof tree stays narrow
and continuous-check mates can be proven to deep distances (e.g. mate-in-25) cheaply.

SOUND: a returned move is a guaranteed mate (every defender reply refuted, line ends in checkmate).
INCOMPLETE: only continuous-check mates — misses mates that need a quiet (non-checking) attacker
move. Node-budget capped: returns (None, None) if no checking mate is proven within the budget,
so the caller falls back to normal play (MCTS).

Works with ANY model (it never touches the net) and needs no retraining — it's a play-time module:
if a forced mate exists, play it; otherwise let the engine choose.
"""
from __future__ import annotations

from typing import Optional, Tuple

import chess


class MateSearcher:
    def __init__(self, max_depth: int = 25, node_budget: int = 300_000):
        self.max_depth = max_depth          # in attacker moves (mate-in-N)
        self.node_budget = node_budget
        self.nodes = 0

    def find_mate(self, board: chess.Board) -> Tuple[Optional[chess.Move], Optional[int]]:
        """Iterative-deepening on mate distance. Returns (move, mate_in_N) or (None, None)."""
        for d in range(1, self.max_depth + 1):
            self.nodes = 0
            mv = self._root(board, d)
            if mv is not None:
                return mv, d
            if self.nodes >= self.node_budget:   # ran out of budget at this depth -> stop deepening
                break
        return None, None

    @staticmethod
    def _checks(board: chess.Board):
        return [m for m in board.legal_moves if board.gives_check(m)]

    def _root(self, board: chess.Board, d: int) -> Optional[chess.Move]:
        for m in self._checks(board):
            board.push(m)
            ok = board.is_checkmate() or (d > 1 and self._defender(board, d - 1))
            board.pop()
            if ok:
                return m
            if self.nodes >= self.node_budget:
                return None
        return None

    def _attacker(self, board: chess.Board, d: int) -> bool:
        """Attacker to move: can it force mate in <= d attacker moves?"""
        if self.nodes >= self.node_budget:
            return False
        for m in self._checks(board):
            board.push(m)
            won = board.is_checkmate() or (d > 1 and self._defender(board, d - 1))
            board.pop()
            if won:
                return True
            if self.nodes >= self.node_budget:
                return False
        return False

    def _defender(self, board: chess.Board, d: int) -> bool:
        """Defender to move (in check): do ALL replies let the attacker mate in <= d?"""
        self.nodes += 1
        replies = list(board.legal_moves)
        if not replies:           # stalemate (checkmate is caught by the caller) -> escapes
            return False
        for r in replies:
            board.push(r)
            ok = self._attacker(board, d)
            board.pop()
            if not ok:            # one escape -> not a forced mate
                return False
            if self.nodes >= self.node_budget:
                return False
        return True


def find_forced_mate(board: chess.Board, max_depth: int = 25, node_budget: int = 300_000
                     ) -> Tuple[Optional[chess.Move], Optional[int]]:
    """Convenience wrapper. Returns (mating_move, mate_in_N) or (None, None)."""
    return MateSearcher(max_depth, node_budget).find_mate(board)


class FullWidthMateSearcher:
    """Shallow ALL-MOVES forced-mate finder. Unlike the continuous-check searcher, the attacker
    may play ANY move (including quiet ones), so this catches mates that need a non-checking move
    -- ladder mates, zugzwang mates, quiet key-moves. Complete for the depths it reaches, but the
    branching is full (~30 moves/node) so it's exponential: keep max_depth small (2-3, maybe 4).
    Node-budget capped; returns (None, None) if no mate proven within budget/depth."""

    def __init__(self, max_depth: int = 3, node_budget: int = 200_000):
        self.max_depth = max_depth          # in attacker moves (mate-in-N)
        self.node_budget = node_budget
        self.nodes = 0

    def find_mate(self, board: chess.Board) -> Tuple[Optional[chess.Move], Optional[int]]:
        """Iterative-deepening on mate distance. Returns (move, mate_in_N) or (None, None)."""
        for d in range(1, self.max_depth + 1):
            self.nodes = 0
            mv = self._attacker(board, d)
            if mv is not None:
                return mv, d
            if self.nodes >= self.node_budget:
                break
        return None, None

    def _attacker(self, board: chess.Board, d: int) -> Optional[chess.Move]:
        """Attacker to move: a move that forces mate in <= d attacker moves, or None."""
        for m in board.legal_moves:
            self.nodes += 1
            if self.nodes >= self.node_budget:
                return None
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return m
            ok = d > 1 and self._defender(board, d - 1)
            board.pop()
            if ok:
                return m
        return None

    def _defender(self, board: chess.Board, d: int) -> bool:
        """Defender to move: do ALL replies allow the attacker to mate in <= d?"""
        if self.nodes >= self.node_budget:
            return False
        replies = list(board.legal_moves)
        if not replies:                  # stalemate (checkmate handled by caller) -> escapes
            return False
        for r in replies:
            board.push(r)
            mv = self._attacker(board, d)
            board.pop()
            if mv is None:               # one escape -> not a forced mate
                return False
        return True


def find_any_forced_mate(board: chess.Board, shallow_depth: int = 3, deep_depth: int = 25,
                         shallow_budget: int = 200_000, deep_budget: int = 300_000
                         ) -> Tuple[Optional[chess.Move], Optional[int], Optional[str]]:
    """Combined finder. Tries the SHALLOW full-width searcher first (catches short mates of ANY
    kind, incl. quiet-move ladder/zugzwang mates), then the DEEP continuous-check searcher
    (catches long check-only mates cheaply). Returns (move, mate_in_N, kind) where kind is
    'fullwidth' or 'check', or (None, None, None)."""
    mv, d = FullWidthMateSearcher(shallow_depth, shallow_budget).find_mate(board)
    if mv is not None:
        return mv, d, "fullwidth"
    mv, d = MateSearcher(deep_depth, deep_budget).find_mate(board)
    if mv is not None:
        return mv, d, "check"
    return None, None, None
