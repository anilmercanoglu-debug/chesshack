"""Single source of truth for all knobs. See SPEC.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DISTILL_DIR = DATA / "distill"
NETS_DIR = DATA / "nets"
REPLAY_DIR = DATA / "replay"
BENCH_DIR = ROOT / "bench"
STOCKFISH = ROOT / "tools" / "stockfish"

# ---------------------------------------------------------------------------
# Encoding constants (the canonical contract shared by every module)
# ---------------------------------------------------------------------------
N_PLANES = 19           # board feature planes (side-to-move POV)
BOARD = 8               # 8x8
N_MOVE_TYPES = 73       # AlphaZero move planes: 56 queen + 8 knight + 9 underpromo
POLICY_SIZE = BOARD * BOARD * N_MOVE_TYPES   # 4672
MAX_LEGAL_MOVES = 218   # theoretical max in a legal position


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NetConfig:
    channels: int = 192
    blocks: int = 15
    se_reduction: int = 8
    value_hidden: int = 256
    value_conv: int = 32
    n_planes: int = N_PLANES
    policy_size: int = POLICY_SIZE


DEV_NET = NetConfig(channels=192, blocks=15)     # ~10.7M — local CPU dev
PROD_NET = NetConfig(channels=256, blocks=20)    # ~24.5M — 80GB production
SCALE_NET = NetConfig(channels=384, blocks=28)   # ~76M — large scale-up (capacity test)
START_NET = NetConfig(channels=256, blocks=4)    # ~5.4M — self-grow start (deepens toward PROD)


# ---------------------------------------------------------------------------
# Phase 1 — distillation labeling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DistillConfig:
    sf_nodes: int = 100_000
    multipv: int = 4
    sf_threads: int = 1
    sf_hash_mb: int = 64
    tau_cp: float = 90.0          # softmax temperature over centipawn scores (policy target)
    topk_mass: float = 0.92       # mass on the MultiPV moves; rest spread over other legals
    cp_clamp: int = 2000          # for the tanh(cp/350) value fallback
    cp_value_scale: float = 350.0
    shard_size: int = 100_000
    n_workers: int = 16


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MCTSConfig:
    c_puct: float = 1.5
    fpu_reduction: float = 0.25
    virtual_loss: float = 1.0
    leaf_batch: int = 16
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    sims_selfplay: int = 600
    sims_bench: int = 800
    sims_tournament: int = 1600


# ---------------------------------------------------------------------------
# Phase 2 — self-play
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SelfPlayConfig:
    sims: int = 600
    temperature: float = 1.0
    temperature_plies: int = 20
    value_z_weight: float = 0.85      # value = 0.85*z + 0.15*q_root (FIXED)
    value_q_weight: float = 0.15
    replay_capacity: int = 1_000_000
    steps_per_fresh: float = 12.0     # ~1 grad step per 8-16 fresh positions
    random_open_frac: float = 0.25
    adjudicate_value: float = 0.92
    adjudicate_plies: int = 4
    # anti-stall
    search_gain_min: float = 0.60     # MCTS(net) vs raw-net-argmax win-rate target
    search_gain_bump_below: float = 0.55
    sims_ladder: tuple = (50, 100, 200, 400, 800, 1600)   # from-scratch ramp: low sims early, bump as the net catches its own search
    gate_winrate: float = 0.65        # promotion threshold (raised from 0.55: 0.55/30-games passes ~29% on noise; 0.65 ~5%)
    gate_games: int = 50              # games per gate (cheaper than 100; still decent SPRT-lite)
    gate_every_games: int = 4000      # run the gate every N generated games (not steps)
    sg_games: int = 30                # games per search_gain check
    sg_every_games: int = 4000        # run search_gain every N generated games
    state_every_games: int = 2000     # checkpoint full training state (for --resume) every N games
    base_elo: float = 1611.0          # starting Elo estimate (the distilled net's measured Elo)
    worker_leaf_batch: int = 32       # leaves per worker MCTS wave (fewer IPC round-trips -> faster)
    # self-grow (depth-only, function-preserving): when the net plateaus, add ResBlocks
    grow_block_step: int = 4          # blocks added per growth
    grow_max_blocks: int = 28         # stop growing at this depth (C256/B28 ~33M)
    grow_after_holds: int = 5         # consecutive gate-holds that trigger a growth
    grow_cooldown_games: int = 3000   # min games between growths (let new capacity train)


# ---------------------------------------------------------------------------
# Bench / Elo
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchConfig:
    elo_ladder: tuple = (1320, 1500, 1700, 1900, 2100, 2300, 2500, 2700, 2900, 3100, 3190)
    games_per_rung: int = 100
    sims: int = 800


DISTILL = DistillConfig()
MCTS = MCTSConfig()
SELFPLAY = SelfPlayConfig()
BENCH = BenchConfig()
