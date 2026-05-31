# ChessHack — Buildable Spec (AlphaZero-style, distillation-bootstrapped, uncapped self-play)

Pure Python (torch + python-chess). PUCT MCTS on CPU + policy/value ResNet on GPU. Two phases, one shared codebase. Targets: ~2000 Elo within one day (via distillation), then continuous, uncapped self-play progress past 2900. Beating full Stockfish (~3600) on one 80GB GPU is NOT promised.

> Environment verified: Python 3.14.2, torch 2.12.0+cpu (CUDA False locally — local box is correctness/dev only), python-chess 1.11.2, numpy 2.4.6, 16 cores, 31 GB RAM. Stockfish 18 at `tools/stockfish`: `UCI_Elo` 1320–3190, `UCI_LimitStrength`, `MultiPV`, `UCI_ShowWDL`, `Skill Level`. MEASURED `nodes=100000` + `MultiPV=4` + WDL = ~10 pos/s/thread. MEASURED movegen+push = ~28k pos/s/core.

## 0. Resolved decisions
1. **Policy head = `Conv1x1(C->73)` + flatten (4672 logits)**, NOT flatten→Linear (saves 9.58M dead params).
2. **Value head = WDL 3-logit softmax**, scalar `v = P(win) − P(loss)`. tanh(cp/350) fallback only if WDL absent.
3. **Net size: dev C=192/B=15 (10.7M), prod C=256/B=20 (24.5M), scale C=320/B=24 (45.5M), SE per block.** (Old killer: 109M.)
4. **Labeling = `nodes=100000`, `MultiPV=4`, `UCI_ShowWDL`, `Threads=1`, full strength.** ~10 pos/s/thread.
5. **Phase-2 value target = `0.85·z + 0.15·q_root`, FIXED (no anneal).**
6. **Anti-stall = `search_gain` monitor + AUTOMATIC sims bump + gated promotion + SF-ladder Elo.**
7. **Sims = FIXED 600 self-play from step one. NO schedule. Auto-bump 600→900→1200→1600 if `search_gain` < 55%.**
8. **No teacher in Phase-2 targets.** Stockfish = Phase-1 labels + Elo anchor only.
9. **Layout = `engine/` + `trainer/` + `bench/`** + `search_gain` + unit tests.
10. **Local box = correctness/dev ONLY (CPU torch).** Strength runs on 80GB Colab; same code, config-driven sizes.

### What would repeat the old ~1200 stall (designed against)
- Low-start sims schedule → fixed sims=600 from step 1.
- Oversized 109M net → 24.5M production net.
- GIL-bound single-stream generation → many CPU tree-search processes + one coalescing GPU inference server.
- Cold start → distillation warm start (~2000–2400 raw).
- Silent stall → `search_gain` + gate + SF-ladder Elo, with auto-bump.

## 1. Net (`engine/net.py`)
Pre-activation ResNet + Squeeze-Excite per block, two heads. One `ChessNet(NetConfig)`; dev/prod differ only by config.
```
Input float32 [N,19,8,8]
Stem:  Conv3x3(19->C, bias=False) -> BN -> ReLU
Body:  B x ResBlock:  [BN-ReLU-Conv3x3(C->C)] x2 + SE(reduction=8) + skip
Policy head: Conv1x1(C->73) -> Flatten -> 4672 logits     # NOT Linear-flatten
Value head:  Conv1x1(C->32) -> BN -> ReLU -> Flatten(2048) -> FC(2048->256) -> ReLU -> FC(256->3)  # WDL logits
Scalar value v = softmax(wdl)[win] - softmax(wdl)[loss]  in [-1,1]
```
Verified params: C=192/B=15 → 10.7M (dev), C=256/B=20 → 24.5M (prod), C=320/B=24 → 45.5M (scale). Legal-move masking (−1e9) before every softmax. Checkpoint = `{state_dict, NetConfig, format_version}`; loader asserts config. GPU: TF32 + bf16 autocast, channels_last.

## 2. Encoding (`engine/encoding.py`) — one canonical module, shared everywhere
Board → 19×8×8 float32 ALWAYS from side-to-move POV (rank-flip + color-swap when black to move). Planes: 0–5 my P,N,B,R,Q,K; 6–11 opponent; 12 stm-is-white flag; 13–16 castling (my K, my Q, opp K, opp Q); 17 en-passant one-hot; 18 halfmove clock/100. No history planes in v1. Move↔index = AlphaZero 8×8×73 = 4672 (56 queen-style + 8 knight + 9 underpromotion; queen-promo uses queen plane), POV-oriented, `index = from*73 + type`. HARD GATE: 10k-position bijection + POV-involution + mask = 100%.

## 3. Phase 1 — Stockfish distillation bootstrap (offline, parallel, no MCTS/curriculum/anneal)
- **Policy target (soft):** SF MultiPV=4; `w_i = softmax(cp_i/tau_cp)`, tau_cp=90; top-K mass 0.92, remaining 0.08 spread uniformly over other legals. Loss = KL.
- **Value target (WDL):** SF POV `score.wdl()/1000`, cross-entropy. Fallback `tanh(cp/350)` (clamp ±2000, mate→±(1−0.001n)) only if WDL absent.
- **Label settings:** Threads=1, Hash=64, MultiPV=4, ShowWDL, full strength, `Limit(nodes=100000)`. ~10 pos/s/thread → ~160 pos/s on 16 cores → ~0.58M/hr. 10% deep subset at nodes=2M.
- **Dataset + timeline:** production 8M = ~14h. FAST-START: label 1.5M slice in ~2.6h, train immediately, append shards toward 8M. Dev slice 200–500k.
- **Sourcing:** ~40% real-game FENs (1/~6 plies, cap 8/game), ~30% random-then-softmax-SF playouts, ~15% sparse-piece (3–7) endgames, ~15% tactical/ECO. FEN/Zobrist de-dup, phase stratification (35/40/25), stm-balanced 50/50, 1% held-out val.
- **On-disk:** sharded npz ~100k/shard, append-only: packed uint8 planes, sparse top-K (≤8) policy (int16 idx + fp16 wt), int16 WDL triple. `manifest.json` records label settings. mmap IterableDataset. REUSED as Phase-2 replay buffer.
- **Labeler:** `multiprocessing.Pool(16)`, one persistent SimpleEngine/worker, per-worker shard files, resumable via manifest.
- **Expected Elo:** raw ~2000–2200 at 1.5M → ~2300–2500 at 8M; +200–300 with MCTS.

## 4. MCTS (`engine/mcts.py`) — PUCT, batched, leaf-parallel
CPU tree (N,W,Q,P per child). Selection `argmax_a [ Q + c_puct·P·sqrt(ΣN_b)/(1+N_a) ]`. Expand via one net eval; negamax backup; exact terminal scoring. Leaf-parallel: collect L=8–16 leaves via virtual loss → one forward → backup all. Tree reuse. Params: c_puct=1.5, Dirichlet alpha=0.3/eps=0.25 root (self-play only), fpu_reduction=0.25, virtual_loss=1. Sims: self-play 600 (fixed), bench 800 (tau=0, no Dirichlet), tournament 1600. HARD GATE: mate-in-1; batched≈serial; vs brute-force.

## 5. Inference server (`engine/inference_server.py`)
One GPU process; coalesces leaf-eval requests from ALL workers into batches (512–2048 on 80GB; ≤128 dev). Flush at 512 OR ~2ms. Workers CPU-only tree work. Hot-swap checkpoints. `spawn`; ONLY server touches GPU; never pickle CUDA tensors.

## 6. Phase 2 — Self-play RL (must KEEP PROGRESSING)
Starts FROM distilled ckpt (`--init-from data/nets/distilled.pt`; same net.py).

### 6.1 ANTI-STALL DESIGN (make-or-break) — five instrumented teeth
1. **STRONG START** — begin from ~2000–2400 distilled net so 600-sim MCTS immediately finds learnable improvements.
2. **ENOUGH SIMS FROM STEP ONE** — fixed 600, never 1→10→50.
3. **RIGHT-SIZED NET + FAST PARALLEL GEN** — 24.5M + many workers + coalescing server; ~1 step / 8–16 fresh positions.
4. **OBSERVABLE + SELF-CORRECTING** — `bench/search_gain.py` win-rate of MCTS(net,600) vs raw-net-argmax over fixed 200-game match; require ≥60%; if <55%, sims ×=1.5 (→900→1200→1600), then scale net width.
5. **GATED PROMOTION** — candidate becomes generator only if it beats current generator ≥55%/100 games (SPRT-lite).
6. **NO TEACHER CAP** — Stockfish never in Phase-2 loss.

### 6.2 Targets
- Policy: normalized MCTS visit dist `pi(a)=N(a)^(1/T)/ΣN^(1/T)`, T=1.0; KL(net‖pi).
- Value: `0.85·z + 0.15·q_root`, FIXED blend; pure-z escape hatch.

### 6.3 Exploration / buffer / ratio
Root Dirichlet alpha=0.3/eps=0.25; T=1.0 first 20 plies then T→0; ~25% games from random distill FEN; adjudicate win if root value>0.92 for 4 plies; draw on 3-fold/50-move. Replay = sliding FIFO ~1.0M positions, uniform sample. Ratio ~1 grad step / 8–16 fresh positions (reuse ~4–8×).

## 7. Shared codebase
`engine/`: encoding.py, net.py, mcts.py, inference_server.py, player.py. `trainer/`: positions.py, label.py, dataset.py (shard format shared), train.py (`--mode distill|selfplay`), selfplay.py, gate.py. `bench/`: elo.py, arena.py, search_gain.py. Root config.py. Distilled ckpt loads directly as Phase-2 init (asserted config).

## 8. Elo measurement (`bench/elo.py`, `bench/arena.py`)
Anchor: SF18 `UCI_LimitStrength` + `UCI_Elo` ladder (1320–3190), node-capped. Our engine net+MCTS@800 (tau=0). Rungs: 1320…3190; above 3190 use fixed-nodes full-strength SF (depth 8/10/12). ≥100 games/rung, alternating colors, fixed opening book. Logistic MLE fit `E=1/(1+10^((R_opp−R_us)/400))`, ±~2σ. Cross-check checkpoint-vs-checkpoint H2H. Append `(timestamp, step, version, Elo, CI, gate%, search_gain)` to `bench/elo_history.json`.

## 9. File manifest
Root config.py; engine/{encoding,net,mcts,inference_server,player}.py; trainer/{positions,label,dataset,train,selfplay,gate}.py; bench/{elo,arena,search_gain}.py; tests/{test_encoding,test_mcts}.py; data/distill/manifest.json; data/nets/; colab_train.ipynb.

## 10. Incremental build order (test + measure Elo each step)
- STEP 0 (<1h): config.py + skeletons; smoke import test.
- STEP 1 (~2h): encoding.py + test_encoding.py. HARD GATE 100% bijection.
- STEP 2 (~2h): net.py (dev C192/B15). TEST shapes/sums/ckpt; CPU forward latency.
- STEP 3 (~3h): positions.py + label.py + dataset.py. Label 5k slice; shard read-back; confirm ~160 pos/s.
- STEP 4 (~3h): train.py --mode distill on 200–500k + bench/elo.py + arena.py. TEST top-1>50%, WDL CE drops. MEASURE ELO ~1700–2000.
- STEP 5 (~3h): mcts.py + player.py + test_mcts.py + search_gain.py. GATE mate-in-1, batched≈serial, search_gain@400≥60%. MEASURE ELO +150–300.
- STEP 6 (~6–8h, Colab 80GB): C256/B20; full 8M labeling (fast-start 1.5M) + train. MEASURE ELO at 1.5M/4M/8M. TARGET ~2000 raw / ~2200+ MCTS Day 1 (M1).
- STEP 7 (~4h): inference_server.py + selfplay.py. Benchmark games/sec + GPU util; policy_KL>0.
- STEP 8 (~4h): gate.py; self-play from distilled @600. GATE first promotion >55%. MEASURE ELO each gate.
- STEP 9 (days→weeks): sustained Phase-2; climb past 2500→2900+; auto-bump sims.

## 11. Milestones
- M0 Pipeline green (dev/CPU): correctness. Day 1.
- M1 Strong warm start (Phase-1, 80GB): ~2000–2400 raw / ~2200–2700 +MCTS@800; END OF DAY 1.
- M2 Ratchet turning: +50–150 over M1, first 1–3 gates; Day 2–4.
- M3 Surpass distillation level: ~2700–2900; Day ~5–14.
- M4 Continued uncapped progress: >2900 and climbing (NOT full-SF ~3600); weeks of GPU.

Honest ceiling: ~2000 in a day well-supported; continuous progress past 2900 is the realistic promise; beating full Stockfish (~3600) on one 80GB GPU is NOT promised.
