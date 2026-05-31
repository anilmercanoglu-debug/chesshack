#!/usr/bin/env bash
# One-shot environment setup: venv + python deps + Stockfish (prebuilt, else build).
# Idempotent-ish; logs everything. Non-fatal on optional pieces (torch).
set -u
ROOT=/home/casper/chesshack
cd "$ROOT" || exit 1
echo "=== [1/5] venv ==="
if [ ! -d .venv ]; then
    python3.14 -m venv .venv || { echo "venv FAILED"; exit 1; }
fi
. .venv/bin/activate
python -m pip install --quiet --upgrade pip 2>&1 | tail -2

echo "=== [2/5] core deps (python-chess, numpy) ==="
python -m pip install --quiet python-chess numpy 2>&1 | tail -3
python -c "import chess, numpy; print('python-chess', chess.__version__, '| numpy', numpy.__version__)" 2>&1

echo "=== [3/5] torch (CPU) — may fail on py3.14, that's OK ==="
if python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu 2>err_torch.log; then
    python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())" 2>&1
    echo "TORCH_OK=1" > .torch_status
else
    echo "torch install FAILED (no cp314 wheel?) — will use a numpy/C++ trainer fallback"
    tail -3 err_torch.log
    echo "TORCH_OK=0" > .torch_status
fi

echo "=== [4/5] Stockfish ==="
mkdir -p tools
SF="$ROOT/tools/stockfish"
if [ ! -x "$SF" ]; then
    echo "trying prebuilt download..."
    URL=https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar
    if curl -fsSL "$URL" -o /tmp/sf.tar 2>/dev/null && tar -xf /tmp/sf.tar -C /tmp 2>/dev/null; then
        found=$(find /tmp/stockfish -type f -name 'stockfish*' -perm -u+x 2>/dev/null | head -1)
        if [ -n "$found" ]; then cp "$found" "$SF" && chmod +x "$SF"; fi
    fi
fi
if [ ! -x "$SF" ]; then
    echo "prebuilt failed; building from source (a few min)..."
    rm -rf /tmp/Stockfish
    if git clone --depth 1 https://github.com/official-stockfish/Stockfish.git /tmp/Stockfish 2>&1 | tail -1; then
        ( cd /tmp/Stockfish/src && make -j16 build ARCH=x86-64-avx2 2>&1 | tail -3 && cp stockfish "$SF" )
    fi
fi
if [ -x "$SF" ]; then
    echo "stockfish ready:"; echo -e "uci\nquit" | "$SF" 2>/dev/null | grep -E "^id (name|author)" | head -2
else
    echo "STOCKFISH SETUP FAILED — will need manual install"
fi

echo "=== [5/5] done ==="
echo "torch_status: $(cat .torch_status 2>/dev/null)"
echo "SETUP_COMPLETE"
