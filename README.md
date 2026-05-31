# ChessHack

A self-improving chess engine: **PUCT MCTS + a policy/value ResNet (GPU)**, trained in two
phases that share one codebase.

1. **Distillation bootstrap** (offline, supervised): regress Stockfish's labels (best
   move(s) → policy, WDL eval → value) on a fixed, diverse position set. Fast, stable, no
   cold start. Target **~2000 Elo within a day**.
2. **Self-play RL** (uncapped): from the distilled net, generate self-play games and train
   the policy toward MCTS visit counts + value toward game outcome. **No teacher cap → keeps
   climbing past 2900.** An anti-stall monitor (`search_gain`) auto-bumps sims and a
   promotion gate keeps the generator improving monotonically.

> Honest ceiling: ~2000 in a day and continuous progress past 2900 are realistic on one
> 80GB GPU. Beating *full* modern Stockfish (~3600) on one GPU is **not** promised.

See **[SPEC.md](SPEC.md)** for the full design (net dims, encoding, distillation formulas,
anti-stall design, Elo methodology).

## Quickstart (Colab, all-in-one)

Open **`colab_train.ipynb`** in Colab (Runtime → GPU), then run the cells top to bottom:

| Cells | What |
|---|---|
| 1–5 | GPU/CPU check, mount Drive, clone repo, install deps + Stockfish, link `data/`+`bench/` to Drive |
| 6 | **Label** ~800k positions with Stockfish (all CPU cores, ~1h on 48 cores) |
| 7 | **Distill-train** the production net (C256/B20, ~24.5M) on the GPU → ~2000 Elo |
| 8 | **Bench**: Elo vs the Stockfish UCI_Elo ladder + `search_gain` sanity |
| 9 | **Self-play** RL (uncapped ratchet) |
| Play | Browser UI to play the bot (drag-and-drop board, served via Colab proxy) |

Everything persists to Google Drive, so a session restart loses nothing.

## Layout

```
config.py              single source of truth (all knobs)
engine/
  encoding.py          19-plane side-to-move board + AlphaZero 4672 move map
  net.py               pre-act ResNet+SE, conv policy head, WDL value head
  mcts.py              PUCT, leaf-parallel batched search
  player.py            net+MCTS move chooser (+ raw-policy player)
  inference_server.py  coalescing GPU broker for parallel self-play
trainer/
  positions.py         diverse position sourcing
  label.py             16-process Stockfish labeler
  dataset.py           shard format (shared by distill + replay)
  train.py             distill | selfplay trainer
  selfplay.py          Phase-2 ratchet driver (server + workers + gate)
  gate.py              SPRT-lite promotion gate
bench/
  arena.py  elo.py  search_gain.py
serve.py               web UI to play the bot
play.py                console fallback to play the bot
tests/                 encoding, net, mcts hard gates
colab_train.ipynb      Colab driver
```

## Local development (CPU)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# (Stockfish: tools/setup_env.sh fetches a binary to tools/stockfish)
PYTHONPATH=. python tests/test_encoding.py   # hard gates
PYTHONPATH=. python tests/test_net.py
PYTHONPATH=. python tests/test_mcts.py
PYTHONPATH=. python serve.py --ckpt data/nets/distilled.pt --sims 200   # play at localhost:8000
```
The local box is for correctness/dev; strength runs on the GPU.

## Play the bot

- **Web UI:** `python serve.py --ckpt <ckpt> --sims 800` → open `http://localhost:8000`
  (drag-and-drop or click, random/choose color, move list, strength slider). On Colab, the
  Play cell prints a proxied URL.
- **Console:** `python play.py --ckpt <ckpt> --sims 400 --color random`
