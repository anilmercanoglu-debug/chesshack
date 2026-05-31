"""Play a game against the bot (net + MCTS), in the console.

    python play.py --ckpt data/nets/distilled.pt --sims 400 --color white

Enter moves as SAN (e4, Nf3, O-O, exd5, e8=Q) or UCI (e2e4, e7e8q).
Commands: 'quit', 'resign', 'moves' (list legal), 'board' (redraw).

Works locally (CPU) and in a Colab cell (input() prompts inline). More sims = stronger but
slower; on CPU keep sims modest (200-400), on GPU you can go 800+.
"""
from __future__ import annotations

import argparse
import random

import chess
import torch

from engine.net import load_checkpoint
from engine.player import Player
from engine.mcts import root_value


def render(board: chess.Board) -> str:
    try:
        return board.unicode(borders=True, empty_square=".")
    except Exception:
        return str(board)


def get_human_move(board: chess.Board):
    while True:
        try:
            s = input("your move (SAN/UCI | moves | board | resign | quit): ").strip()
        except EOFError:
            return "quit"
        if s in ("quit", "exit"):
            return "quit"
        if s == "resign":
            return "resign"
        if s == "moves":
            print("  " + " ".join(board.san(m) for m in board.legal_moves))
            continue
        if s == "board":
            print(render(board))
            continue
        if not s:
            continue
        mv = None
        try:
            mv = board.parse_san(s)
        except Exception:
            try:
                cand = chess.Move.from_uci(s.lower())
                mv = cand if cand in board.legal_moves else None
            except Exception:
                mv = None
        if mv is None or mv not in board.legal_moves:
            print("  illegal or unrecognized move — try again ('moves' to list).")
            continue
        return mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/nets/distilled.pt")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--color", choices=["white", "black", "random"], default="white")
    ap.add_argument("--leaf-batch", type=int, default=16)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, meta = load_checkpoint(args.ckpt, map_location=dev)
    net = net.to(dev)
    bot = Player(net, dev, sims=args.sims, leaf_batch=args.leaf_batch)

    human = {"white": chess.WHITE, "black": chess.BLACK}.get(
        args.color, random.choice([chess.WHITE, chess.BLACK]))
    print(f"You are {'White' if human == chess.WHITE else 'Black'}. "
          f"Bot: {args.ckpt} @ {args.sims} sims on {dev}.")
    board = chess.Board()
    print(render(board))

    while not board.is_game_over(claim_draw=True):
        if board.turn == human:
            mv = get_human_move(board)
            if mv == "quit":
                print("bye."); return
            if mv == "resign":
                print("you resigned. bot wins."); return
            board.push(mv)
        else:
            mv, root = bot.choose(board, temperature=0.0)
            if mv is None:
                break
            v = root_value(root)
            san = board.san(mv)
            board.push(mv)
            print(f"\nbot plays {san}   (bot self-eval {v:+.2f}, {int(root.N.sum())} sims)")
        print(render(board))

    print("\nresult:", board.result(claim_draw=True))


if __name__ == "__main__":
    main()
