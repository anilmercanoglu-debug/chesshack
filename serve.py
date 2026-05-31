"""Web UI to play against the bot. Self-contained: vanilla JS + Unicode pieces (no image/JS
CDNs that break behind the Colab proxy). All chess legality is server-side (python-chess);
the browser just renders the board, highlights legal targets on click, and POSTs moves.

    python serve.py --ckpt data/nets/distilled.pt --sims 400 --port 8000

Local: http://localhost:8000.  Colab: run, then proxyPort(8000) for the URL.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess
import torch

from engine.net import load_checkpoint
from engine.player import Player
from engine.mcts import root_value

BOT = None
BOT_LOCK = threading.Lock()
DEFAULT_SIMS = 400

INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>ChessHack — play the bot</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font-family:system-ui,Arial,sans-serif;background:#262421;color:#e8e6e3;margin:0;
   padding:18px;display:flex;gap:24px;flex-wrap:wrap;justify-content:center}
 #board{display:grid;grid-template-columns:repeat(8,60px);grid-template-rows:repeat(8,60px);
   border:3px solid #111;box-shadow:0 4px 20px #0008}
 .sq{width:60px;height:60px;display:flex;align-items:center;justify-content:center;
   font-size:44px;line-height:1;cursor:pointer;position:relative;user-select:none}
 .light{background:#eeeed2}.dark{background:#769656}
 .p.w{color:#fbfbfb;text-shadow:0 0 2px #000,0 0 2px #000,0 0 3px #000}
 .p.b{color:#1a1a1a;text-shadow:0 0 1px #999}
 .sel{box-shadow:inset 0 0 0 4px #f6f669a0}
 .last{background:#bbcb44 !important}
 .tgt::after{content:"";position:absolute;width:22px;height:22px;border-radius:50%;
   background:#20202055}
 .tgtcap::after{content:"";position:absolute;width:54px;height:54px;border-radius:50%;
   box-shadow:inset 0 0 0 5px #20202055}
 .chk{background:#e06666 !important}
 .panel{min-width:240px;max-width:340px}
 h1{font-size:20px;margin:0 0 12px}
 button,select{font-size:15px;padding:7px 12px;margin:4px 4px 4px 0;background:#4a4844;
   color:#e8e6e3;border:1px solid #5a5854;border-radius:6px;cursor:pointer}
 button:hover{background:#5a5854}
 #status{margin-top:14px;font-size:15px;line-height:1.6;min-height:70px}
 .eval{font-weight:bold}
 label{display:block;margin-top:10px;font-size:14px;color:#bdbab5}
 input[type=range]{width:100%}
 #promo{display:none;margin-top:10px}#promo button{font-size:30px;padding:2px 10px}
</style></head><body>
<div id="board"></div>
<div class="panel">
 <h1>&#9823; ChessHack</h1>
 <button id="new">New game</button>
 <label>Play as
  <select id="color"><option value="random" selected>Random</option><option value="white">White</option><option value="black">Black</option></select></label>
 <div id="who" style="margin-top:8px;font-weight:bold"></div>
 <label>Bot strength (sims): <span id="simsv">400</span>
  <input id="sims" type="range" min="50" max="1600" step="50" value="400"></label>
 <div id="promo">Promote to:
  <button data-p="q">&#9819;</button><button data-p="r">&#9820;</button>
  <button data-p="b">&#9821;</button><button data-p="n">&#9822;</button></div>
 <div id="status">Click a piece to move.</div>
</div>
<script>
const GLYPH={K:'♚',Q:'♛',R:'♜',B:'♝',N:'♞',P:'♟'};
const $=id=>document.getElementById(id);
let st=null, humanColor='white', sel=null, busy=false, pendingPromo=null, lastMove=null;

function sqName(file,rank){return 'abcdefgh'[file]+(rank+1);}
function idxOf(name){return (7-(+name[1]-1))*8+'abcdefgh'.indexOf(name[0]);}
async function api(path,body){
 const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify(body)}:{}); return r.json();
}
function movable(name){return !!(st&&st.legal[name])&&!busy&&!st.gameover;}
function render(){
 const b=$('board'); b.innerHTML='';
 const ranks=humanColor==='white'?[7,6,5,4,3,2,1,0]:[0,1,2,3,4,5,6,7];
 const files=humanColor==='white'?[0,1,2,3,4,5,6,7]:[7,6,5,4,3,2,1,0];
 const targets=sel&&st.legal[sel]?st.legal[sel]:[];
 for(const rank of ranks)for(const file of files){
   const name=sqName(file,rank), i=(7-rank)*8+file, piece=st.cells[i];
   const d=document.createElement('div');
   d.className='sq '+((file+rank)%2?'light':'dark'); d.dataset.sq=name;
   if(piece){const g=document.createElement('span');g.className='p '+piece[0];
     g.textContent=GLYPH[piece[1]];d.appendChild(g);}
   if(name===sel)d.classList.add('sel');
   if(lastMove&&(name===lastMove[0]||name===lastMove[1]))d.classList.add('last');
   if(targets.includes(name))d.classList.add(piece?'tgtcap':'tgt');
   if(st.check&&piece&&piece[1]==='K'&&((piece[0]==='w')===(st.turn==='w')))d.classList.add('chk');
   d.onclick=()=>onClick(name);
   d.draggable=movable(name);
   d.ondragstart=e=>{if(!movable(name)){e.preventDefault();return;}sel=name;render();
     e.dataTransfer.setData('text/plain',name);e.dataTransfer.effectAllowed='move';};
   d.ondragover=e=>e.preventDefault();
   d.ondrop=e=>{e.preventDefault();const from=e.dataTransfer.getData('text/plain')||sel;
     if(from&&st.legal[from]&&st.legal[from].includes(name))attemptMove(from,name);
     else{sel=null;render();}};
   b.appendChild(d);
 }
}
function setStatus(h){$('status').innerHTML=h;}
function gameOverMsg(){return st.gameover?('<br><b>Game over: '+st.result+'</b>'):'';}
function needPromo(from,to){const p=st.cells[idxOf(from)];
 return !!p&&p[1]==='P'&&(to[1]==='8'||to[1]==='1');}
function attemptMove(from,to){
 if(busy||st.gameover)return;
 if(needPromo(from,to)){pendingPromo=[from,to];$('promo').style.display='block';return;}
 doMove(from,to,null);
}
async function doMove(from,to,promo){
 busy=true; sel=null; setStatus('bot is thinking&hellip;');
 const res=await api('/api/move',{fen:st.fen,from,to,promotion:promo||null,sims:+$('sims').value});
 if(res.error){busy=false; st=res.state||st; render(); setStatus('error: '+res.error); return;}
 lastMove=res.bot?[res.bot.from,res.bot.to]:[from,to]; st=res.state; render();
 if(res.bot){const s=res.bot.eval>=0?'+':'';
   setStatus('bot played <b>'+res.bot.san+'</b><br><span class="eval">eval '+s+
     res.bot.eval.toFixed(2)+'</span> (bot POV), '+res.bot.sims+' sims'+gameOverMsg());}
 else setStatus('your move.'+gameOverMsg());
 busy=false;
}
function onClick(name){
 if(busy||!st||st.gameover)return;
 const targets=sel&&st.legal[sel]?st.legal[sel]:[];
 if(sel&&targets.includes(name)){attemptMove(sel,name);return;}
 if(st.legal[name]){sel=name;render();} else {sel=null;render();}
}
async function botFirst(){
 busy=true; setStatus('bot is thinking&hellip;');
 const res=await api('/api/botmove',{fen:st.fen,sims:+$('sims').value});
 lastMove=res.bot?[res.bot.from,res.bot.to]:null; st=res.state; render();
 if(res.bot){const s=res.bot.eval>=0?'+':'';
   setStatus('bot played <b>'+res.bot.san+'</b><br><span class="eval">eval '+s+
     res.bot.eval.toFixed(2)+'</span>, '+res.bot.sims+' sims'+gameOverMsg());}
 busy=false;
}
async function newGame(){
 const choice=$('color').value;
 humanColor=choice==='random'?(Math.random()<0.5?'white':'black'):choice;
 sel=null; lastMove=null; $('promo').style.display='none';
 st=await api('/api/new'); render();
 $('who').textContent='You play '+humanColor.toUpperCase()+(choice==='random'?' (random)':'');
 setStatus(humanColor==='white'?'Your move. Click or drag a piece.':'Bot moves first&hellip;');
 if(humanColor==='black')botFirst();
}
$('new').onclick=newGame;
$('sims').oninput=e=>$('simsv').textContent=e.target.value;
document.querySelectorAll('#promo button').forEach(btn=>btn.onclick=()=>{
 $('promo').style.display='none'; const [f,t]=pendingPromo; doMove(f,t,btn.dataset.p);});
newGame();
</script></body></html>"""


def _state(board: chess.Board) -> dict:
    cells = []
    for rank in range(7, -1, -1):
        for file in range(8):
            p = board.piece_at(chess.square(file, rank))
            cells.append((("w" if p.color else "b") + chess.piece_symbol(p.piece_type).upper())
                         if p else None)
    legal: dict = {}
    for m in board.legal_moves:
        legal.setdefault(chess.square_name(m.from_square), set()).add(chess.square_name(m.to_square))
    legal = {k: sorted(v) for k, v in legal.items()}
    over = board.is_game_over(claim_draw=True)
    return {"fen": board.fen(), "cells": cells, "turn": "w" if board.turn else "b",
            "legal": legal, "gameover": over,
            "result": board.result(claim_draw=True) if over else None,
            "check": board.is_check()}


def _bot_move(board: chess.Board, sims: int) -> dict:
    with BOT_LOCK:
        mv, root = BOT.choose(board, temperature=0.0, sims=sims)
    san = board.san(mv)
    info = {"from": chess.square_name(mv.from_square), "to": chess.square_name(mv.to_square),
            "san": san, "eval": root_value(root), "sims": int(root.N.sum())}
    board.push(mv)
    return info


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode())

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/new":
            self._json(_state(chess.Board()))
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            sims = int(req.get("sims", DEFAULT_SIMS))
            if self.path == "/api/botmove":
                board = chess.Board(req["fen"])
                bot = None if board.is_game_over(claim_draw=True) else _bot_move(board, sims)
                self._json({"bot": bot, "state": _state(board)})
            elif self.path == "/api/move":
                board = chess.Board(req["fen"])
                promo = req.get("promotion")
                mv = chess.Move(chess.parse_square(req["from"]), chess.parse_square(req["to"]),
                                promotion=chess.Piece.from_symbol(promo).piece_type if promo else None)
                if mv not in board.legal_moves:
                    self._json({"error": "illegal move", "state": _state(board)}); return
                board.push(mv)
                bot = None if board.is_game_over(claim_draw=True) else _bot_move(board, sims)
                self._json({"bot": bot, "state": _state(board)})
            else:
                self._send(404, b'{"error":"not found"}')
        except Exception as e:
            self._json({"error": str(e)})


def main():
    global BOT, DEFAULT_SIMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/nets/distilled.pt")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--leaf-batch", type=int, default=16)
    args = ap.parse_args()
    DEFAULT_SIMS = args.sims
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, _ = load_checkpoint(args.ckpt, map_location=dev)
    net = net.to(dev)
    BOT = Player(net, dev, sims=args.sims, leaf_batch=args.leaf_batch)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[serve] {args.ckpt} @ {args.sims} sims on {dev}")
    print(f"[serve] open http://localhost:{args.port}  (Colab: proxyPort({args.port}))")
    srv.serve_forever()


if __name__ == "__main__":
    main()
