"""Coalescing inference server for Phase-2 self-play.

A broker thread in the MAIN process holds the generator net on the GPU. Worker PROCESSES
(CPU-only tree search) ship encoded leaf planes over a multiprocessing queue and block for
the result; the broker concatenates pending requests from ALL workers into ONE forward and
scatters results back. This fills a big GPU while many cores search in parallel.

Returns FULL softmax over 4672 (no mask): mcts._expand selects the legal indices and
renormalizes, which is identical to a legal-masked softmax. So no masks cross the IPC.

Rules (avoid CUDA+mp crashes): 'spawn' start method; ONLY this broker touches the GPU;
workers never import torch-on-GPU or pickle CUDA tensors.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from engine.encoding import board_to_planes
from engine.net import value_from_wdl


class InferenceServer:
    def __init__(self, net, device: str, n_workers: int, ctx,
                 max_batch: int = 256, max_wait_ms: float = 2.0):
        self.net = net.eval()
        self.device = device
        self.max_batch = int(max_batch)
        self.max_wait = float(max_wait_ms) / 1000.0
        self.request_q = ctx.Queue()
        self.result_qs = [ctx.Queue() for _ in range(n_workers)]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.batches = 0
        self.rows = 0
        self.thread = threading.Thread(target=self._loop, name="infer-server", daemon=True)
        self.thread.start()

    @torch.no_grad()
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self.request_q.get(timeout=0.1)
            except queue.Empty:
                continue
            reqs: List[Tuple[int, np.ndarray]] = [first]
            total = first[1].shape[0]
            deadline = time.monotonic() + self.max_wait
            while total < self.max_batch:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    break
                try:
                    r = self.request_q.get(timeout=rem)
                    reqs.append(r)
                    total += r[1].shape[0]
                except queue.Empty:
                    break

            planes = np.concatenate([r[1] for r in reqs], axis=0)
            x = torch.from_numpy(planes).to(self.device)
            with self._lock:
                with torch.autocast(self.device, dtype=torch.bfloat16,
                                    enabled=(self.device == "cuda")):
                    p_logits, wdl = self.net(x)
            probs = F.softmax(p_logits.float(), dim=1).cpu().numpy()
            vals = value_from_wdl(wdl.float()).cpu().numpy()

            self.batches += 1
            self.rows += total
            off = 0
            for wid, pl in reqs:
                b = pl.shape[0]
                self.result_qs[wid].put((probs[off:off + b], vals[off:off + b]))
                off += b

    def update_net(self, state_dict) -> None:
        """Hot-swap the generator weights (called on promotion)."""
        with self._lock:
            self.net.load_state_dict(state_dict)
            self.net.eval()

    @property
    def avg_batch(self) -> float:
        return self.rows / max(self.batches, 1)

    def stop(self) -> None:
        self._stop.set()


class ServerEvaluator:
    """Worker-side evaluate_fn: boards -> (full-softmax probs[B,4672], values[B]) via the broker."""

    def __init__(self, worker_id: int, request_q, result_q):
        self.worker_id = worker_id
        self.request_q = request_q
        self.result_q = result_q

    def __call__(self, boards):
        planes = np.stack([board_to_planes(b) for b in boards]).astype(np.float32)
        self.request_q.put((self.worker_id, planes))
        return self.result_q.get()
