"""Web UI to play against the bot. Drag-and-drop board in the browser; the server runs the
net + MCTS and replies with its move.

    python serve.py --ckpt data/nets/distilled.pt --sims 400 --port 8000

Local: open http://localhost:8000.  Colab: run, then
    from google.colab.output import eval_js; print(eval_js('google.colab.kernel.proxyPort(8000)'))
and open the printed URL. No extra deps (stdlib http.server + CDN board).
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

INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>ChessHack — play the bot</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<style>
 body{font-family:system-ui,sans-serif;background:#262421;color:#e8e6e3;margin:0;padding:20px;display:flex;gap:24px;flex-wrap:wrap;justify-content:center}
 #board{width:480px}
 .panel{min-width:240px;max-width:320px}
 h1{font-size:20px;margin:0 0 12px}
 button,select{font-size:15px;padding:7px 12px;margin:4px 0;background:#4a4844;color:#e8e6e3;border:1px solid #5a5854;border-radius:6px;cursor:pointer}
 button:hover{background:#5a5854}
 #status{margin-top:14px;font-size:15px;line-height:1.5;min-height:80px}
 .eval{font-weight:bold}
 label{display:block;margin-top:10px;font-size:14px;color:#bdbab5}
 input[type=range]{width:100%}
</style></head><body>
<div id="board"></div>
<div class="panel">
 <h1>♟ ChessHack</h1>
 <button id="new">New game</button>
 <label>Play as
  <select id="color"><option value="white">White</option><option value="black">Black</option></select></label>
 <label>Bot strength (sims): <span id="simsv">400</span>
  <input id="sims" type="range" min="50" max="1600" step="50" value="400"></label>
 <div id="status">Drag a piece to move.</div>
</div>
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
<script>
const PIECES='https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/img/chesspieces/wikipedia/{piece}.png';
let game=new Chess(), board, human='white', thinking=false;
const $st=document.getElementById('status');
function setStatus(h){$st.innerHTML=h;}
function over(){
 if(!game.game_over())return false;
 let m='Game over — ';
 if(game.in_checkmate())m+=(game.turn()==='w'?'Black':'White')+' wins by checkmate';
 else if(game.in_draw())m+='draw';
 else m+='over';
 setStatus(m); return true;
}
async function botMove(){
 thinking=true; setStatus('bot is thinking…');
 const sims=+document.getElementById('sims').value;
 const r=await fetch('/api/move',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({fen:game.fen(),sims})});
 const d=await r.json(); thinking=false;
 if(d.error){setStatus('error: '+d.error);return;}
 game.move({from:d.from,to:d.to,promotion:d.promotion||undefined});
 board.position(game.fen());
 const sign=(d.eval>=0?'+':'');
 setStatus('bot played <b>'+d.san+'</b><br><span class="eval">eval '+sign+d.eval.toFixed(2)+
   '</span> (its own POV), '+d.sims+' sims');
 over();
}
function onDrop(src,tgt){
 if(thinking)return 'snapback';
 const mv=game.move({from:src,to:tgt,promotion:'q'});
 if(mv===null)return 'snapback';
 board.position(game.fen());
 if(over())return;
 setTimeout(botMove,150);
}
function newGame(){
 game.reset(); human=document.getElementById('color').value;
 board=Chessboard('board',{draggable:true,position:'start',pieceTheme:PIECES,
   orientation:human,onDrop:onDrop});
 board.position(game.fen());
 setStatus('your move.');
 if(human==='black')setTimeout(botMove,300);
}
document.getElementById('new').onclick=newGame;
document.getElementById('sims').oninput=e=>document.getElementById('simsv').textContent=e.target.value;
newGame();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/move":
            self._send(404, b'{"error":"not found"}'); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            board = chess.Board(req["fen"])
            sims = int(req.get("sims", DEFAULT_SIMS))
            if board.is_game_over(claim_draw=True):
                self._send(200, json.dumps({"error": "game over"}).encode()); return
            with BOT_LOCK:
                mv, root = BOT.choose(board, temperature=0.0, sims=sims)
            ev = root_value(root)
            san = board.san(mv)
            promo = chess.piece_symbol(mv.promotion) if mv.promotion else None
            out = {"from": chess.square_name(mv.from_square),
                   "to": chess.square_name(mv.to_square), "promotion": promo,
                   "san": san, "eval": ev, "sims": int(root.N.sum())}
            self._send(200, json.dumps(out).encode())
        except Exception as e:
            self._send(200, json.dumps({"error": str(e)}).encode())


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
    print(f"[serve] open http://localhost:{args.port}  (Colab: use proxyPort({args.port}))")
    srv.serve_forever()


if __name__ == "__main__":
    main()
