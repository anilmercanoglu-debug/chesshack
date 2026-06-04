"""Promotion gate (SPRT-lite). A candidate net only becomes the generator if it beats the
current champion by >= gate_winrate over gate_games. Guarantees every accepted step is a
verified Elo gain, so the generator improves monotonically and can't drift downhill."""
from __future__ import annotations

from config import SELFPLAY, MCTS as MCTS_CFG
from bench.arena import MCTSPlayer, play_match


def gate(candidate_net, champion_net, device: str = "cpu",
         games: int = SELFPLAY.gate_games, sims: int = SELFPLAY.sims,
         winrate: float = SELFPLAY.gate_winrate, leaf_batch: int = MCTS_CFG.leaf_batch,
         seed: int = 0, openings=None, mate_depth: int = 0) -> dict:
    cand = MCTSPlayer(candidate_net, sims=sims, device=device, leaf_batch=leaf_batch,
                      temperature=0.0, mate_depth=mate_depth)
    champ = MCTSPlayer(champion_net, sims=sims, device=device, leaf_batch=leaf_batch,
                       temperature=0.0, mate_depth=mate_depth)
    score, detail = play_match(cand, champ, games, seed=seed, openings=openings)
    detail.update({"candidate_score": score, "threshold": winrate, "promote": score >= winrate})
    return detail
