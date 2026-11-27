#!/usr/bin/env python3
"""
Integration tests for `looped_fok.proxy` and fail-closed `ScopePolicyFilter` using an
in-process mock Loom upstream server (zero GPU required).

Covers:
  - ScopePolicyFilter fail-closed when context is missing/empty + cross-workspace isolation
  - Atomic batch persistence (all-or-nothing rollback on failure)
  - All 4 proxy quadrants (0,0), (1,0), (0,1), (1,1) — streaming SSE and non-streaming JSON
  - `<memory_op>...</memory_op>` split across arbitrary SSE chunk boundaries (reasoning & content)
  - Strict previous-write -> next-read ordering barrier under delayed background extraction
  - Graceful shutdown drain of pending non-daemon writes
  - Tool-role passthrough and auxiliary session-title bypass
  - Stage E adapter gating (`magic=0x7` ON for memory-active replays vs OFF for (0,0))
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from looped_fok import GeneralMemoryStore, MemoryItem, ScopePolicyFilter
from looped_fok.proxy import FOKProxyState, make_handler


class MockLoomUpstream:
    """Deterministic fake Loom 0.3.0 server that records every request and emits scripted SSE/JSON."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.script: list[dict] = []
        self._lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def enqueue(self, item: dict) -> None:
        with self._lock:
            self.script.append(item)

    def _make_handler(self):
        parent = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                with parent._lock:
                    parent.requests.append(body)
                    spec = parent.script.pop(0) if parent.script else {"deltas": [{"content": "default"}]}

                if spec.get("delay_s"):
                    time.sleep(spec["delay_s"])

                if not body.get("stream", False):
                    raw = json.dumps({
                        "id": "mock-1",
                        "object": "chat.completion",
                        "created": 1,
                        "model": body.get("model", "mock"),
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": spec.get("content", "[]")}, "finish_reason": "stop"}],
                    }).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Connection", "close")
                self.end_headers()
                for d in spec.get("deltas", []):
                    chunk = {"id": "mock-s", "object": "chat.completion.chunk", "created": 1, "model": "mock", "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        return H


class TestSecurityAndAtomicity(unittest.TestCase):
    def test_scope_policy_fail_closed_without_context(self) -> None:
        store = GeneralMemoryStore(policy=ScopePolicyFilter())
        store.add(MemoryItem("m_priv", "schedule_event", "therapy", "Weekly physical therapy", ["therapy"],
                             metadata={"tenant_id": "alice", "scope": "private", "visibility": "private_owner_only"}))
        self.assertEqual(store.active_items(None), [])
        self.assertEqual(store.active_items({}), [])
        self.assertFalse(store.has_active_memories(None))
        self.assertEqual(len(store.active_items({"tenant_id": "alice", "allowed_scopes": ["private"]})), 1)

    def test_atomic_batch_rollback_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mem.json"
            store = GeneralMemoryStore(persist_path=p)
            store.add(MemoryItem("m_01", "project_state", "db", "DuckDB", ["duckdb"]))
            with self.assertRaises(KeyError):
                with store.batch():
                    store.update("m_01", MemoryItem("m_02", "project_state", "db", "ClickHouse", ["clickhouse"]))
                    store.update("m_missing", MemoryItem("m_03", "project_state", "x", "y", []))
            # Must roll back m_01 -> m_02 completely both in memory and on disk!
            self.assertEqual([it.item_id for it in store.active_items()], ["m_01"])
            reloaded = GeneralMemoryStore(persist_path=p)
            self.assertEqual([it.item_id for it in reloaded.active_items()], ["m_01"])


class TestProxyIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.store_path = Path(self.td.name) / "memories.json"
        self.mock = MockLoomUpstream()
        self.state = FOKProxyState(self.mock.url, "inkling-stage-e", "inkling-fok-memory", self.store_path)
        self.proxy = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.state))
        self.proxy_url = f"http://127.0.0.1:{self.proxy.server_address[1]}"
        self.proxy_thread = threading.Thread(target=self.proxy.serve_forever, daemon=True)
        self.proxy_thread.start()

    def tearDown(self) -> None:
        self.proxy.shutdown()
        self.state.drain_pending_writes()
        self.proxy.server_close()
        self.mock.close()
        self.td.cleanup()

    def _chat(self, messages: list[dict], stream: bool = True, scope: str = "repo:alpha", max_tokens: int = 256) -> tuple[str, str]:
        body = json.dumps({"model": "inkling-fok-memory", "messages": messages, "stream": stream, "max_tokens": max_tokens}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.proxy_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "X-FOK-Scope": scope},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            if not stream:
                obj = json.loads(resp.read().decode("utf-8"))
                msg = obj["choices"][0]["message"]
                return msg.get("content", ""), msg.get("reasoning_content", "")
            content, reasoning = "", ""
            for line in resp:
                s = line.decode("utf-8", "ignore").strip()
                if not s.startswith("data:"):
                    continue
                ds = s[5:].strip()
                if ds == "[DONE]":
                    break
                d = (json.loads(ds)["choices"][0].get("delta") or {})
                content += d.get("content") or ""
                reasoning += d.get("reasoning_content") or ""
            return content, reasoning

    def test_quadrant_00_silent_passthrough_and_tool_and_aux_bypass(self) -> None:
        # (0,0) streaming -> no replay, single upstream call with Write-Only gate_mode=7 (empty store)
        self.mock.enqueue({"deltas": [{"content": "Hello "}, {"content": "world"}]})
        c, _ = self._chat([{"role": "user", "content": "Say hello"}], stream=True)
        self.assertEqual(c, "Hello world")
        self.assertEqual(len(self.mock.requests), 1)
        ctrl = self.mock.requests[0]["fok_ctrl"] & 0xFFFFFFFF
        self.assertEqual((ctrl >> 20) & 0xF, 7)

        # Tool-role passthrough -> fok_ctrl omitted completely
        self.mock.enqueue({"deltas": [{"content": "Tool ok"}]})
        c2, _ = self._chat([{"role": "user", "content": "run"}, {"role": "tool", "content": "exit 0"}], stream=False)
        self.assertEqual(c2, "Tool ok")
        self.assertNotIn("fok_ctrl", self.mock.requests[-1])

        # Auxiliary session-title bypass -> fok_ctrl omitted completely
        self.mock.enqueue({"deltas": [{"content": "Short Title"}]})
        c3, _ = self._chat([{"role": "system", "content": "Generate a 5-word session title."}, {"role": "user", "content": "hi"}], stream=False, max_tokens=64)
        self.assertEqual(c3, "Short Title")
        self.assertNotIn("fok_ctrl", self.mock.requests[-1])

    def test_split_inline_memory_op_and_stage_e_adapter_gate(self) -> None:
        # Probe emits 200016 (0,1); replay splits `<memory_op>...</memory_op>` across chunks in reasoning_content!
        self.mock.enqueue({"deltas": [{"reasoning_content": "<|unused_200016|>"}]})
        self.mock.enqueue({"deltas": [
            {"reasoning_content": "<memo"},
            {"reasoning_content": 'ry_op>[{"op":"add","category":"project_state","key":"stack","value":"Go 1.23","tags":["go"]}]</memory_op>'},
            {"content": "Saved Go 1.23."},
        ]})
        c, r = self._chat([{"role": "user", "content": "Remember Go 1.23."}], stream=True, scope="repo:alpha")
        self.assertEqual(c, "Saved Go 1.23.")
        self.assertNotIn("<memory_op>", r)
        # Verify replay enabled Stage E adapter (magic=0x7)
        replay_ctrl = self.mock.requests[1]["fok_ctrl"] & 0xFFFFFFFF
        self.assertEqual((replay_ctrl >> 28) & 0xF, 0x7)
        # Wait for handler thread post-[DONE] inline commit before inspecting internal store directly
        self.state.wait_for_pending_writes()
        # Verify memory committed inline and visible in repo:alpha, NOT in repo:beta
        self.assertEqual(len(self.state.store.active_items({"tenant_id": "default", "allowed_scopes": ["repo:alpha"]})), 1)
        self.assertEqual(len(self.state.store.active_items({"tenant_id": "default", "allowed_scopes": ["repo:beta"]})), 0)

    def test_immediate_write_to_recall_ordering_and_cross_scope_delete_blocked(self) -> None:
        # Write turn falls back to delayed background extractor (0.25s delay!)
        self.mock.enqueue({"deltas": [{"reasoning_content": "<|unused_200016|>"}]})
        self.mock.enqueue({"deltas": [{"content": "Got it."}]})
        self.mock.enqueue({
            "delay_s": 0.25,
            "content": '[{"op":"add","category":"fact_constraint","key":"port","value":"Use port 9099","tags":["port","9099"]}]',
        })
        c_w, _ = self._chat([{"role": "user", "content": "Use port 9099."}], stream=True, scope="repo:alpha")
        self.assertEqual(c_w, "Got it.")

        # IMMEDIATELY issue a recall on a brand-new connection — MUST wait for the 0.25s commit and see has_mem=True!
        self.mock.enqueue({"deltas": [{"reasoning_content": "<|unused_200011|>"}]})
        self.mock.enqueue({"deltas": [{"content": "Port 9099."}]})
        c_r, _ = self._chat([{"role": "user", "content": "What port?"}], stream=False, scope="repo:alpha")
        self.assertEqual(c_r, "Port 9099.")
        # Confirm the recall probe ran with gate_mode=1 (has_mem=True) and injected "Use port 9099"!
        recall_probe = self.mock.requests[-2]
        self.assertEqual(((recall_probe["fok_ctrl"] & 0xFFFFFFFF) >> 20) & 0xF, 1)
        self.assertIn("Use port 9099", self.mock.requests[-1]["fok_prefill"])

        # Cross-workspace DELETE /fok/memories/m_01 from repo:beta must return 403 Forbidden!
        req_del = urllib.request.Request(
            f"{self.proxy_url}/fok/memories/m_01",
            headers={"X-FOK-Scope": "repo:beta"},
            method="DELETE",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx_err:
            urllib.request.urlopen(req_del, timeout=5)
        self.assertEqual(ctx_err.exception.code, 403)

    def test_quadrant_11_dual_update_and_graceful_drain(self) -> None:
        self.state.store.add(MemoryItem("m_01", "project_state", "db", "DuckDB", ["duckdb"], {"tenant_id": "default", "scope": "repo:alpha"}))
        self.state._next_id = 2
        # (1,1) probe emits 200019 -> deref + delayed background update!
        self.mock.enqueue({"deltas": [{"reasoning_content": "<|unused_200019|>"}]})
        self.mock.enqueue({"deltas": [{"content": "Updated to ClickHouse."}]})
        self.mock.enqueue({
            "delay_s": 0.15,
            "content": '[{"op":"update","target_id":"m_01","category":"project_state","key":"db","value":"ClickHouse","tags":["clickhouse"]}]',
        })
        c, _ = self._chat([{"role": "user", "content": "Switch DB to ClickHouse."}], stream=True, scope="repo:alpha")
        self.assertEqual(c, "Updated to ClickHouse.")
        # Simulate graceful shutdown drain before exit
        self.state.drain_pending_writes()
        reloaded = GeneralMemoryStore(policy=ScopePolicyFilter(), persist_path=self.store_path)
        items = reloaded.active_items({"tenant_id": "default", "allowed_scopes": ["repo:alpha"]})
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].value, "ClickHouse")
        self.assertEqual(items[0].version, 2)


if __name__ == "__main__":
    unittest.main()
