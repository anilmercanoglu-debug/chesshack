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
 .sq span{pointer-events:none}
 .sel{box-shadow:inset 0 0 0 4px #f6f669a0}
 .last{background:#bbcb44 !important}
 .tgt::after{content:"";position:absolute;width:22px;height:22px;border-radius:50%;
   background:#20202055}
 .tgtcap::after{content:"";position:absolute;width:54px;height:54px;border-radius:50%;
   box-shadow:inset 0 0 0 5px #20202055}
 .chk{background:#e06666 !important}
 .left{display:flex;flex-direction:column;align-items:center}
 #nav{margin-top:8px;width:486px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
 #nav button{font-size:16px;padding:4px 9px;margin:0}
 #navInfo{font-size:13px;color:#bdbab5;min-width:96px;text-align:center}
 #navSlider{flex:1;min-width:120px}
 .panel{min-width:240px;max-width:340px}
 h1{font-size:20px;margin:0 0 12px}
 button,select{font-size:15px;padding:7px 12px;margin:4px 4px 4px 0;background:#4a4844;
   color:#e8e6e3;border:1px solid #5a5854;border-radius:6px;cursor:pointer}
 button:hover{background:#5a5854}
 #status{margin-top:14px;font-size:15px;line-height:1.6;min-height:56px}
 #moves{margin-top:10px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:14px;
   max-height:260px;overflow-y:auto;background:#1f1d1b;padding:8px 10px;border-radius:6px}
 #moves div{padding:1px 0}#moves div:nth-child(odd){background:#2a2825}
 #moves span{cursor:pointer;padding:0 3px;border-radius:3px}
 #moves span:hover{background:#3a3a37}#moves span.cur{background:#bbcb44;color:#111}
 .eval{font-weight:bold}
 label{display:block;margin-top:10px;font-size:14px;color:#bdbab5}
 input[type=range]{width:100%}
 #promo{display:none;margin-top:10px}#promo button{font-size:30px;padding:2px 10px}
</style></head><body>
<div class="left">
 <div id="board"></div>
 <div id="nav">
  <button id="navFirst" title="basa">&#9198;</button>
  <button id="navPrev" title="geri (sol ok)">&#9664;</button>
  <span id="navInfo">live</span>
  <button id="navNext" title="ileri (sag ok)">&#9654;</button>
  <button id="navLast" title="canliya don">&#9197; live</button>
  <input id="navSlider" type="range" min="0" max="0" value="0">
 </div>
</div>
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
 <div id="moves"></div>
</div>
<script>
const GLYPH={K:'♚',Q:'♛',R:'♜',B:'♝',N:'♞',P:'♟'};
const $=id=>document.getElementById(id);
let positions=[], viewIdx=0, humanColor='white', sel=null, busy=false, pendingPromo=null, history=[];
const live=()=>positions[positions.length-1];
const atLive=()=>viewIdx===positions.length-1;

function sqName(file,rank){return 'abcdefgh'[file]+(rank+1);}
function idxOf(name){return (7-(+name[1]-1))*8+'abcdefgh'.indexOf(name[0]);}
async function api(path,body){
 const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify(body)}:{}); return r.json();
}
function movable(name){return atLive()&&!!live().legal[name]&&!busy&&!live().gameover;}
function highlightTargets(){
 const tgts=sel&&live().legal[sel]?live().legal[sel]:[];
 document.querySelectorAll('#board .sq').forEach(d=>{
   const nm=d.dataset.sq, isT=tgts.includes(nm), piece=live().cells[idxOf(nm)];
   d.classList.toggle('sel', nm===sel);
   d.classList.toggle('tgt', isT&&!piece);
   d.classList.toggle('tgtcap', isT&&!!piece);
 });
}
function render(){
 const view=positions[viewIdx]; if(!view)return;
 const b=$('board'); b.innerHTML='';
 const ranks=humanColor==='white'?[7,6,5,4,3,2,1,0]:[0,1,2,3,4,5,6,7];
 const files=humanColor==='white'?[0,1,2,3,4,5,6,7]:[7,6,5,4,3,2,1,0];
 const targets=(atLive()&&sel&&live().legal[sel])?live().legal[sel]:[];
 for(const rank of ranks)for(const file of files){
   const name=sqName(file,rank), i=(7-rank)*8+file, piece=view.cells[i];
   const d=document.createElement('div');
   d.className='sq '+((file+rank)%2?'light':'dark'); d.dataset.sq=name;
   if(piece){const g=document.createElement('span');g.className='p '+piece[0];
     g.textContent=GLYPH[piece[1]];d.appendChild(g);}
   if(atLive()&&name===sel)d.classList.add('sel');
   if(view.last&&(name===view.last[0]||name===view.last[1]))d.classList.add('last');
   if(targets.includes(name))d.classList.add(piece?'tgtcap':'tgt');
   if(view.check&&piece&&piece[1]==='K'&&((piece[0]==='w')===(view.turn==='w')))d.classList.add('chk');
   d.onclick=()=>onClick(name);
   d.draggable=movable(name);
   d.ondragstart=e=>{if(!movable(name)){e.preventDefault();return;}
     e.dataTransfer.setData('text/plain',name);e.dataTransfer.effectAllowed='move';
     sel=name; highlightTargets();
     const sp=d.querySelector('span'); if(sp)setTimeout(()=>{sp.style.visibility='hidden';},0);};
   d.ondragover=e=>e.preventDefault();
   d.ondrop=e=>{e.preventDefault();if(!atLive())return;const from=e.dataTransfer.getData('text/plain')||sel;
     if(from&&live().legal[from]&&live().legal[from].includes(name))attemptMove(from,name);
     else{sel=null;render();}};
   d.ondragend=e=>{if(!busy){sel=null;render();}};
   b.appendChild(d);
 }
 renderNav();
}
function renderNav(){
 const n=positions.length-1, s=$('navSlider');
 s.max=n; s.value=viewIdx;
 $('navInfo').textContent=atLive()?('live '+viewIdx+'/'+n):('inceleme '+viewIdx+'/'+n);
 renderMoves();
}
function setView(i){viewIdx=Math.max(0,Math.min(positions.length-1,i));sel=null;render();}
function pushPos(state,lastMove){state.last=lastMove||null;positions.push(state);viewIdx=positions.length-1;}
function setStatus(h){$('status').innerHTML=h;}
function gameOverMsg(){return live().gameover?('<br><b>Game over: '+live().result+'</b>'):'';}
function needPromo(from,to){const p=live().cells[idxOf(from)];
 return !!p&&p[1]==='P'&&(to[1]==='8'||to[1]==='1');}
function attemptMove(from,to){
 if(busy||live().gameover)return;
 if(needPromo(from,to)){pendingPromo=[from,to];$('promo').style.display='block';return;}
 doMove(from,to,null);
}
function botMsg(bot){const s=bot.eval>=0?'+':'';
 return 'bot played <b>'+bot.san+'</b><br><span class="eval">eval '+s+bot.eval.toFixed(2)+
   '</span> (bot POV), '+bot.sims+' sims'+gameOverMsg();}
function renderMoves(){
 let h='';
 for(let i=0;i<history.length;i+=2){const n=i/2+1;
   h+='<div>'+n+'. '+mv(i)+(history[i+1]?' '+mv(i+1):'')+'</div>';}
 $('moves').innerHTML=h;
 if(atLive()){const m=$('moves');m.scrollTop=m.scrollHeight;}
 function mv(k){return '<span'+(viewIdx===k+1?' class="cur"':'')+' onclick="setView('+(k+1)+')">'+history[k]+'</span>';}
}
async function doMove(from,to,promo){
 busy=true; sel=null;
 const r1=await api('/api/move',{fen:live().fen,from,to,promotion:promo||null});
 if(r1.error){busy=false; render(); setStatus('error: '+r1.error); return;}
 history.push(r1.san); pushPos(r1.state,[from,to]); render();
 if(live().gameover){busy=false; setStatus('move played.'+gameOverMsg()); return;}
 setStatus('bot is thinking&hellip;');
 const r2=await api('/api/botmove',{fen:live().fen,sims:+$('sims').value});
 if(r2.bot){history.push(r2.bot.san); pushPos(r2.state,[r2.bot.from,r2.bot.to]);}
 else pushPos(r2.state,null);
 render();
 setStatus(r2.bot?botMsg(r2.bot):('your move.'+gameOverMsg()));
 busy=false;
}
function onClick(name){
 if(!atLive())return;            // reviewing past moves -> board read-only; use nav to return
 if(busy||live().gameover)return;
 const targets=sel&&live().legal[sel]?live().legal[sel]:[];
 if(sel&&targets.includes(name)){attemptMove(sel,name);return;}
 if(live().legal[name]){sel=name;render();} else {sel=null;render();}
}
async function botFirst(){
 busy=true; setStatus('bot is thinking&hellip;');
 const res=await api('/api/botmove',{fen:live().fen,sims:+$('sims').value});
 if(res.bot){history.push(res.bot.san); pushPos(res.state,[res.bot.from,res.bot.to]);}
 else pushPos(res.state,null);
 render();
 setStatus(res.bot?botMsg(res.bot):'your move.'); busy=false;
}
async function newGame(){
 const choice=$('color').value;
 humanColor=choice==='random'?(Math.random()<0.5?'white':'black'):choice;
 sel=null; history=[]; positions=[]; viewIdx=0; $('promo').style.display='none';
 const st0=await api('/api/new'); st0.last=null; positions=[st0]; viewIdx=0; render();
 $('who').textContent='You play '+humanColor.toUpperCase()+(choice==='random'?' (random)':'');
 setStatus(humanColor==='white'?'Your move. Click or drag a piece.':'Bot moves first&hellip;');
 if(humanColor==='black')botFirst();
}
$('new').onclick=newGame;
$('sims').oninput=e=>$('simsv').textContent=e.target.value;
$('navFirst').onclick=()=>setView(0);
$('navPrev').onclick=()=>setView(viewIdx-1);
$('navNext').onclick=()=>setView(viewIdx+1);
$('navLast').onclick=()=>setView(positions.length-1);
$('navSlider').oninput=e=>setView(+e.target.value);
document.addEventListener('keydown',e=>{
 if(e.key==='ArrowLeft')setView(viewIdx-1);
 else if(e.key==='ArrowRight')setView(viewIdx+1);});
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
                san = board.san(mv)
                board.push(mv)
                self._json({"san": san, "state": _state(board)})
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
