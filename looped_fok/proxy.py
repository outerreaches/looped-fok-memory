#!/usr/bin/env python3
"""
OpenAI-Compatible Streaming Proxy for Unified On-GPU Read+Write Looped FOK (`looped_fok.proxy`).

Sits in front of a single Loom 0.3.0 `inkling-stage-d2` server (`127.0.0.1:8189`) and exposes a
standard OpenAI Chat Completions API (`127.0.0.1:8190/v1`) for memory-less harnesses like `dsh`
(DeepSeek Harness), Aider, Cline, or OpenCode.

Features:
  - Agentic tool-loop guard: only fresh user turns (`messages[-1].role == "user"`) engage FOK;
    tool-result turns (`role == "tool"`) pass straight through untouched (`fok_ctrl = None`).
  - Zero-copy SSE streaming on `(0, 0)` turns: buffers only token 0 to inspect `200011..200021`;
    if normal text/reasoning/tool_call, flushes immediately and streams the rest straight through!
  - Background async Write-FOK extraction guarded by `write_lock` so user-facing latency stays
    unaffected while immediate follow-up `dsh headless` calls never race a pending commit.
  - Persistent JSON-backed `GeneralMemoryStore` with `ScopePolicyFilter` + admin endpoints:
      GET /health, GET /v1/models, GET /fok/memories, DELETE /fok/memories
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import signal
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .client import EXTRACTOR_SYS, PF_MM, UNIFIED_TOKENS, pack_ctrl
from .store import CATEGORY_LIST, GeneralMemoryStore, MemoryItem, ScopePolicyFilter

LOG = logging.getLogger("looped_fok.proxy")


def extract_user_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(content)


def normalize_loom_payload(req_body: dict, upstream_model: str) -> tuple[dict, bool]:
    """Normalize OpenAI/dsh request payloads for Loom 0.3.0 (developer->system, list content->str).

    Returns (normalized_payload, is_aux_title_call).
    """
    raw_msgs = req_body.get("messages") or []
    norm_msgs = []
    sys_texts = []
    for m in raw_msgs:
        nm = dict(m)
        role = nm.get("role", "user")
        if role == "developer":
            role = "system"
            nm["role"] = "system"
        if isinstance(nm.get("content"), list):
            nm["content"] = extract_user_text(nm)
        elif nm.get("content") is None and role != "assistant":
            nm["content"] = ""
        if role == "system" and isinstance(nm.get("content"), str):
            sys_texts.append(nm["content"].lower())
        norm_msgs.append(nm)

    # Harness guard (e.g. dsh): if the harness appends a synthetic runtime-context `user`
    # message AFTER the actual user message, swap it ahead of the real user message (as
    # `system` context) so `span_last` (<|message_model|>) sits directly after the real user turn!
    if (
        len(norm_msgs) >= 2
        and norm_msgs[-1].get("role") == "user"
        and norm_msgs[-2].get("role") == "user"
        and isinstance(norm_msgs[-1].get("content"), str)
        and norm_msgs[-1]["content"].lstrip().startswith("Current runtime context")
    ):
        rt_ctx = norm_msgs.pop()
        rt_ctx["role"] = "system"
        norm_msgs.insert(len(norm_msgs) - 1, rt_ctx)

    out: dict = {"model": upstream_model, "messages": norm_msgs, "stream": True}
    for k in ("temperature", "top_p", "max_tokens", "reasoning_effort", "tools", "tool_choice", "stop"):
        if k in req_body and req_body[k] is not None:
            out[k] = req_body[k]
    if "max_tokens" not in out and req_body.get("max_completion_tokens") is not None:
        out["max_tokens"] = req_body["max_completion_tokens"]

    # Detect harness auxiliary metadata calls (e.g. dsh session-title-llm: max_tokens <= 64 + title prompt)
    max_tok = out.get("max_tokens")
    is_aux = bool(max_tok is not None and int(max_tok) <= 64 and any("title" in s for s in sys_texts))
    return out, is_aux


def build_write_context(norm_msgs: list[dict], max_prev_chars: int = 2400) -> str:
    """Build recent conversation context for Write-FOK extraction (previous assistant summary + current user turn).

    Handles both direct user declarations and anaphoric requests like
    'remember those durable architecture facts we just discussed'.
    """
    user_text = extract_user_text(norm_msgs[-1]) if norm_msgs else ""
    prev_asst = ""
    for m in reversed(norm_msgs[:-1]):
        if m.get("role") == "assistant":
            txt = extract_user_text(m).strip()
            if txt:
                prev_asst = txt[:max_prev_chars]
                break
    if prev_asst:
        return (
            f"Immediately preceding assistant summary (context if referenced by user):\n{prev_asst}\n\n"
            f"Latest user message:\n{user_text}"
        )
    return f"Latest user message:\n{user_text}"


def pack_adapter_ctrl(scale: float = 1.0) -> int:
    """Pack Stage E 32-bit `fok_ctrl` word (magic=0x7) enabling the FOK-Gated Post-Coda SwiGLU Adapter."""
    aq = max(0, min(8191, int(round(scale * 256.0))))
    val = (0x7 << 28) | aq
    return val - 0x100000000 if val >= 0x80000000 else val


def parse_ops_resilient(raw: str) -> list[dict]:
    """Parse JSON array of memory ops, falling back to per-object decoding if one item has a syntax glitch."""
    s = raw.strip()
    m = re.search(r"\[.*\]", s, flags=re.DOTALL)
    candidate = m.group(0) if m else s
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    except Exception:
        pass
    # Fallback: recover any valid JSON objects inside the output individually
    dec = json.JSONDecoder()
    ops: list[dict] = []
    idx = 0
    while idx < len(s):
        pos = s.find("{", idx)
        if pos < 0:
            break
        try:
            obj, end = dec.raw_decode(s, pos)
            if isinstance(obj, dict) and "op" in obj:
                ops.append(obj)
            idx = end
        except Exception:
            idx = pos + 1
    return ops


def inspect_delta_signal(delta: dict) -> tuple[str | None, bool, bool, str | None]:
    txt = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
    if not txt:
        return None, False, False, None
    for tok_s, (dr, dw, cat) in UNIFIED_TOKENS.items():
        if tok_s in txt:
            return tok_s, dr, dw, cat
    return None, False, False, None


class FOKProxyState:
    def __init__(
        self,
        upstream_url: str,
        upstream_model: str,
        public_model: str,
        store_path: str | Path,
        alpha: float = 0.35,
        reader: str = "mlp",
    ) -> None:
        self.upstream_url = upstream_url.rstrip("/")
        self.upstream_model = upstream_model
        self.public_model = public_model
        self.store_path = Path(store_path)
        self.store = GeneralMemoryStore(policy=ScopePolicyFilter(), persist_path=self.store_path)
        is_lin = 1 if reader == "lin" else 0
        self.ctrl_rw = pack_ctrl(alpha, vec_id=1, gate_mode=1, scope=9, reader_slot=0, is_lin=is_lin)
        self.ctrl_write_only = pack_ctrl(0.0, vec_id=1, gate_mode=7, scope=9, reader_slot=0, is_lin=is_lin)
        self.ctrl_adapter = pack_adapter_ctrl(1.0)
        self.write_lock = threading.Lock()
        self._pending_cond = threading.Condition()
        self._pending_writes = 0
        self._write_threads: list[threading.Thread] = []
        self._id_lock = threading.Lock()
        # Resume monotonic ID counter above any loaded IDs
        max_num = 0
        for it in self.store._items.values():
            m = re.match(r"m_(\d+)$", it.item_id)
            if m:
                max_num = max(max_num, int(m.group(1)))
        self._next_id = max_num + 1

    def begin_pending_write(self) -> None:
        with self._pending_cond:
            self._pending_writes += 1

    def end_pending_write(self) -> None:
        with self._pending_cond:
            self._pending_writes = max(0, self._pending_writes - 1)
            if self._pending_writes == 0:
                self._pending_cond.notify_all()

    def wait_for_pending_writes(self, timeout: float = 60.0) -> bool:
        with self._pending_cond:
            return self._pending_cond.wait_for(lambda: self._pending_writes == 0, timeout=timeout)

    def drain_pending_writes(self, timeout: float = 60.0) -> None:
        self.wait_for_pending_writes(timeout=timeout)
        for th in list(self._write_threads):
            if th.is_alive():
                th.join(timeout=timeout)
        self._write_threads = [th for th in self._write_threads if th.is_alive()]

    def start_background_extractor(self, write_ctx: str, ctx: dict, write_meta: dict) -> None:
        """Spawn a non-daemon extractor thread; caller MUST have already called begin_pending_write()."""
        def _worker() -> None:
            try:
                self.run_write_extractor(write_ctx, ctx, write_meta)
            finally:
                self.end_pending_write()

        th = threading.Thread(target=_worker, daemon=False)
        self._write_threads = [t for t in self._write_threads if t.is_alive()] + [th]
        th.start()

    def next_mem_id(self) -> str:
        with self._id_lock:
            nid = f"m_{self._next_id:02d}"
            self._next_id += 1
            return nid

    def open_upstream(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.upstream_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream" if payload.get("stream") else "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(req, timeout=300.0)

    def run_write_extractor(self, write_context: str, ctx: dict, write_meta: dict) -> None:
        with self.write_lock:
            t0 = time.perf_counter()
            active_dump = self.store.dump_all_active(ctx)
            prompt = (
                f"Currently active memories:\n{json.dumps(active_dump, indent=2)}\n\n"
                f"{write_context}\n\n"
                "Rules: emit at most 4 concise, 1-sentence operations; avoid raw double quotes inside values (use single quotes or backticks). "
                "If the latest user message states durable facts OR asks to remember/update/delete facts from the "
                "immediately preceding assistant summary, output ONLY the valid JSON array of memory operations:"
            )
            payload = {
                "model": self.upstream_model,
                "messages": [
                    {"role": "system", "content": EXTRACTOR_SYS},
                    {"role": "user", "content": prompt},
                ],
                "reasoning_effort": "low",
                "temperature": 0.0,
                "max_tokens": 1536,
                "fok_prefill": PF_MM,
                "stream": False,
            }
            try:
                with self.open_upstream(payload) as resp:
                    obj = json.loads(resp.read().decode("utf-8", "ignore"))
                raw = ((obj.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                ops = parse_ops_resilient(raw)
                if not ops and "{" in raw:
                    LOG.warning("Write extractor returned unparseable JSON (%d chars): %s", len(raw), raw[:300])
            except Exception as e:
                LOG.warning("Write extractor failed: %s", e)
                return
            self.commit_ops(ops, ctx, write_meta, source="extractor", elapsed_ms=(time.perf_counter() - t0) * 1000)

    def commit_ops(self, ops: list[dict], ctx: dict, write_meta: dict, source: str = "inline", elapsed_ms: float = 0.0) -> list[str]:
        """Commit parsed memory operations as one atomic batch (caller holds `write_lock`)."""
        applied: list[str] = []
        active_ids = {it.item_id for it in self.store.active_items(ctx)}
        with self.store.batch():
            for op in ops:
                if not isinstance(op, dict):
                    continue
                kind = str(op.get("op", "")).lower().strip()
                cat = str(op.get("category", "project_state")).strip()
                if cat not in CATEGORY_LIST:
                    cat = "project_state"
                key = str(op.get("key") or "").strip()
                val = str(op.get("value") or op.get("content") or "").strip()
                if not key and val:
                    words = [w for w in re.sub(r"[^a-z0-9]+", " ", val.lower()).split() if len(w) >= 3][:4]
                    key = "_".join(words) or "memory_note"
                tags = [str(t).lower() for t in (op.get("tags") or []) if isinstance(t, str)]
                if not tags and key:
                    tags = [w for w in key.split("_") if len(w) >= 3][:5]
                tid = op.get("target_id")

                if kind == "add" and val:
                    nid = self.next_mem_id()
                    self.store.add(MemoryItem(nid, cat, key, val, tags, dict(write_meta)))
                    active_ids.add(nid)
                    applied.append(f"add:{nid}({key})")
                elif kind == "update" and tid in active_ids and val:
                    nid = self.next_mem_id()
                    self.store.update(tid, MemoryItem(nid, cat, key, val, tags, dict(write_meta)))
                    active_ids.discard(tid)
                    active_ids.add(nid)
                    applied.append(f"upd:{tid}->{nid}({key})")
                elif kind == "delete" and tid in active_ids:
                    self.store.delete(tid)
                    active_ids.discard(tid)
                    applied.append(f"del:{tid}")
        LOG.info("Write-FOK [%s] committed %s in %.1f ms (active=%d)", source, applied, elapsed_ms, len(self.store.active_items(ctx)))
        return applied


def make_handler(state: FOKProxyState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            LOG.debug("%s - " + fmt, self.address_string(), *args)

        def _send_json(self, code: int, obj: dict) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _ctx_from_headers(self) -> tuple[dict, dict]:
            scope = self.headers.get("X-FOK-Scope", "default").strip() or "default"
            tenant = self.headers.get("X-FOK-Tenant", "default").strip() or "default"
            ctx = {"tenant_id": tenant, "allowed_scopes": [scope, "global"]}
            meta = {"tenant_id": tenant, "scope": scope}
            return ctx, meta

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/health", "/v1/health", "/readyz"):
                self._send_json(200, {
                    "status": "ok",
                    "public_model": state.public_model,
                    "upstream_model": state.upstream_model,
                    "active_memories": len(state.store.all_active_unfiltered()),
                    "pending_writes": state._pending_writes,
                    "store_path": str(state.store_path),
                })
                return
            if parsed.path in ("/v1/models", f"/v1/models/{state.public_model}"):
                now = int(time.time())
                self._send_json(200, {
                    "object": "list",
                    "data": [{
                        "id": state.public_model,
                        "object": "model",
                        "created": now,
                        "owned_by": "looped-fok-memory",
                    }],
                })
                return
            if parsed.path == "/fok/memories":
                state.wait_for_pending_writes()
                qs = parse_qs(parsed.query)
                if qs.get("all", ["0"])[0] == "1":
                    mems = [
                        {"id": it.item_id, "version": it.version, "category": it.category, "key": it.key, "value": it.value, "scope": it.metadata.get("scope", "default")}
                        for it in state.store.all_active_unfiltered()
                    ]
                    self._send_json(200, {"memories": mems})
                    return
                scope = qs.get("scope", [None])[0] or self.headers.get("X-FOK-Scope")
                tenant = qs.get("tenant", [None])[0] or self.headers.get("X-FOK-Tenant")
                if not scope and not tenant:
                    # Trusted-local default: list all active memories only when explicitly queried with ?all=1 or scope
                    mems = [
                        {"id": it.item_id, "version": it.version, "category": it.category, "key": it.key, "value": it.value, "scope": it.metadata.get("scope", "default")}
                        for it in state.store.all_active_unfiltered()
                    ]
                    self._send_json(200, {"memories": mems})
                    return
                ctx = {"allowed_scopes": [scope, "global"]} if scope else {}
                if tenant:
                    ctx["tenant_id"] = tenant
                self._send_json(200, {"memories": state.store.dump_all_active(ctx)})
                return
            self._send_json(404, {"error": "not_found"})

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            state.wait_for_pending_writes()
            if parsed.path == "/fok/memories":
                qs = parse_qs(parsed.query)
                scope = qs.get("scope", [None])[0] or self.headers.get("X-FOK-Scope")
                tenant = qs.get("tenant", [None])[0] or self.headers.get("X-FOK-Tenant")
                with state.write_lock:
                    if scope or tenant:
                        ctx = {"allowed_scopes": [scope]} if scope else {}
                        if tenant:
                            ctx["tenant_id"] = tenant
                        with state.store.batch():
                            for it in list(state.store.active_items(ctx)):
                                state.store.delete(it.item_id)
                        self._send_json(200, {"cleared": True, "scope": scope, "tenant": tenant})
                    else:
                        state.store.clear_all()
                        self._send_json(200, {"cleared": True})
                return
            if parsed.path.startswith("/fok/memories/"):
                mid = parsed.path.rsplit("/", 1)[-1]
                scope = self.headers.get("X-FOK-Scope")
                tenant = self.headers.get("X-FOK-Tenant")
                with state.write_lock:
                    if scope or tenant:
                        ctx = {"allowed_scopes": [scope, "global"]} if scope else {}
                        if tenant:
                            ctx["tenant_id"] = tenant
                        visible = {it.item_id for it in state.store.active_items(ctx)}
                        if mid not in visible:
                            self._send_json(403, {"error": f"memory {mid} not accessible in caller scope"})
                            return
                    try:
                        state.store.delete(mid)
                    except KeyError:
                        self._send_json(404, {"error": f"unknown memory {mid}"})
                        return
                self._send_json(200, {"deleted": mid})
                return
            self._send_json(404, {"error": "not_found"})

        def _stream_passthrough(self, upstream_resp, prebuffered: list[bytes] | None = None) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                if prebuffered:
                    for chunk in prebuffered:
                        self.wfile.write(chunk)
                    self.wfile.flush()
                for line in upstream_resp:
                    self.wfile.write(line)
                    self.wfile.flush()
            except BrokenPipeError:
                pass

        def _collect_non_stream(self, upstream_resp) -> dict:
            """Collect an SSE upstream response into a single OpenAI non-streaming JSON response."""
            content, reasoning, finish_reason = "", "", "stop"
            tool_calls: dict[int, dict] = {}
            cid, created, model_name = f"chatcmpl-fok-{int(time.time())}", int(time.time()), state.public_model
            usage = None
            for line in upstream_resp:
                s = line.decode("utf-8", "ignore").strip()
                if not s.startswith("data:"):
                    continue
                ds = s[5:].strip()
                if ds == "[DONE]":
                    break
                obj = json.loads(ds)
                cid = obj.get("id", cid)
                created = obj.get("created", created)
                if obj.get("usage"):
                    usage = obj["usage"]
                ch = (obj.get("choices") or [None])[0]
                if not ch:
                    continue
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    reasoning += d["reasoning_content"]
                if d.get("content"):
                    content += d["content"]
                for tc in d.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(idx, {"id": tc.get("id", ""), "type": "function", "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
            msg: dict = {"role": "assistant", "content": content}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if tool_calls:
                msg["tool_calls"] = [tool_calls[k] for k in sorted(tool_calls)]
            out = {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
            }
            if usage:
                out["usage"] = usage
            return out

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/v1/chat/completions":
                self._send_json(404, {"error": "not_found"})
                return

            length = int(self.headers.get("Content-Length", "0"))
            req_body = json.loads(self.rfile.read(length).decode("utf-8"))
            want_stream = bool(req_body.get("stream", False))
            messages = req_body.get("messages") or []
            ctx, write_meta = self._ctx_from_headers()

            # Guarantee previous-write -> next-read ordering before checking active memories!
            state.wait_for_pending_writes()
            with state.write_lock:
                has_mem = state.store.has_active_memories(ctx)

            # Normalize OpenAI/dsh fields for Loom 0.3.0 (developer->system, list content->str)
            base_payload, is_aux = normalize_loom_payload(req_body, state.upstream_model)
            norm_msgs = base_payload["messages"]

            # Agentic Loop Guard: only fresh user turns (non-auxiliary) engage Read+Write FOK!
            is_user_turn = (not is_aux) and bool(norm_msgs) and (norm_msgs[-1].get("role") == "user")
            user_text = extract_user_text(norm_msgs[-1]) if is_user_turn else ""

            if not is_user_turn:
                LOG.info("Passthrough turn (last_role=%s, aux=%s)", norm_msgs[-1].get("role") if norm_msgs else None, is_aux)
                try:
                    with state.open_upstream(base_payload) as resp:
                        if want_stream:
                            self._stream_passthrough(resp)
                        else:
                            self._send_json(200, self._collect_non_stream(resp))
                except urllib.error.HTTPError as e:
                    err_body = e.read().decode("utf-8", "ignore")
                    LOG.error("Upstream passthrough %d: %s", e.code, err_body[:400])
                    self._send_json(e.code, {"error": err_body})
                return

            # Stage D.2 Single-Pass Unified On-GPU Read+Write Probe
            probe_payload = dict(base_payload)
            probe_payload["fok_prefill"] = PF_MM
            probe_payload["fok_ctrl"] = state.ctrl_rw if has_mem else state.ctrl_write_only

            t0 = time.perf_counter()
            prebuf: list[bytes] = []
            sig_tok, did_read, did_write, routed_cat = None, False, False, None

            try:
                resp1 = state.open_upstream(probe_payload)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", "ignore")
                LOG.error("Upstream %d on keys=%s roles=%s body=%s", e.code, sorted(probe_payload.keys()), [m.get("role") for m in messages], err_body[:500])
                self._send_json(e.code, {"error": err_body})
                return
            try:
                for raw_line in resp1:
                    prebuf.append(raw_line)
                    s = raw_line.decode("utf-8", "ignore").strip()
                    if not s.startswith("data:"):
                        continue
                    ds = s[5:].strip()
                    if ds == "[DONE]":
                        break
                    obj = json.loads(ds)
                    ch = (obj.get("choices") or [None])[0]
                    if not ch:
                        continue
                    d = ch.get("delta") or {}
                    has_tok = bool(d.get("reasoning_content") or d.get("content") or d.get("tool_calls"))
                    if has_tok:
                        sig_tok, did_read, did_write, routed_cat = inspect_delta_signal(d)
                        break

                sig_ms = (time.perf_counter() - t0) * 1000.0
                LOG.info(
                    "FOK signal: tok=%s R=%d W=%d cat=%s (%.1f ms, has_mem=%s, q=%r)",
                    sig_tok, int(did_read), int(did_write), routed_cat, sig_ms, has_mem, user_text[:60],
                )

                # Quadrant (0, 0): zero overhead, flush prebuffer and stream straight through!
                if not did_read and not did_write:
                    if want_stream:
                        self._stream_passthrough(resp1, prebuffered=prebuf)
                    else:
                        # Reconstruct from prebuf + remaining stream
                        def _iter_all():
                            yield from prebuf
                            yield from resp1
                        self._send_json(200, self._collect_non_stream(_iter_all()))
                    return
            finally:
                if did_read or did_write:
                    resp1.close()

            # Build replay payload for (1,0), (0,1), or (1,1) — enable Stage E FOK-Gated Post-Coda Adapter (magic=0x7)!
            replay_payload = dict(base_payload)
            replay_payload["fok_ctrl"] = state.ctrl_adapter
            if did_read:
                qres = state.store.query_by_category(routed_cat or "project_state", user_text, ctx)
                LOG.info("Read-FOK surfaced %d/%d memories: %s", qres["n_surfaced"], qres["n_total_active"], [m["id"] for m in qres["memories"]])
                write_note = " Durable updates/deletions from this turn are being saved automatically." if did_write else ""
                replay_payload["fok_prefill"] = (
                    f"<|message_model|><|content_thinking|>\n"
                    f"[Relevant user memories inserted: {json.dumps(qres)}]\n"
                    f"Use these memories where relevant,{write_note} think briefly, and answer the user directly.\n"
                )
            else:
                replay_payload["fok_prefill"] = (
                    "<|message_model|><|content_thinking|>\n"
                    "[Persistent memory note: durable facts from this turn are being saved automatically in the background.]\n"
                    "Acknowledge concisely without claiming you lack persistent memory.\n"
                )

            # Register pending write BEFORE streaming the reply so any immediate follow-up turn waits for commit!
            if did_write:
                state.begin_pending_write()

            inline_buf = ""
            write_dispatched = False
            try:
                with state.open_upstream(replay_payload) as resp2:
                    if want_stream:
                        if not did_write:
                            self._stream_passthrough(resp2)
                        else:
                            # Stream while capturing & stripping `<memory_op>...</memory_op>` from reasoning or content
                            self.send_response(200)
                            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                            self.send_header("Cache-Control", "no-cache")
                            self.send_header("Connection", "close")
                            self.end_headers()
                            in_mem_op = {"reasoning_content": False, "content": False}
                            pending_tag = {"reasoning_content": "", "content": ""}
                            try:
                                for raw_line in resp2:
                                    s = raw_line.decode("utf-8", "ignore").strip()
                                    if not s.startswith("data:") or s[5:].strip() == "[DONE]":
                                        self.wfile.write(raw_line)
                                        self.wfile.flush()
                                        continue
                                    obj = json.loads(s[5:].strip())
                                    ch = (obj.get("choices") or [None])[0]
                                    d = (ch.get("delta") or {}) if ch else {}
                                    touched = False
                                    drop_chunk = False
                                    for fld in ("reasoning_content", "content"):
                                        txt = d.get(fld)
                                        if not txt:
                                            continue
                                        touched = True
                                        inline_buf += txt
                                        if in_mem_op[fld]:
                                            end_pos = txt.find("</memory_op>")
                                            if end_pos >= 0:
                                                in_mem_op[fld] = False
                                                rem = txt[end_pos + len("</memory_op>"):]
                                                if rem:
                                                    d[fld] = rem
                                                else:
                                                    d.pop(fld, None)
                                                    drop_chunk = True
                                            else:
                                                d.pop(fld, None)
                                                drop_chunk = True
                                            continue
                                        combined = pending_tag[fld] + txt
                                        pending_tag[fld] = ""
                                        tag_pos = combined.find("<memory_op>")
                                        if tag_pos >= 0:
                                            before = combined[:tag_pos]
                                            after = combined[tag_pos + len("<memory_op>"):]
                                            end_pos = after.find("</memory_op>")
                                            if end_pos >= 0:
                                                emit = (before + after[end_pos + len("</memory_op>"):]).rstrip("\n")
                                            else:
                                                in_mem_op[fld] = True
                                                emit = before.rstrip("\n")
                                            if emit:
                                                d[fld] = emit
                                            else:
                                                d.pop(fld, None)
                                                drop_chunk = True
                                            continue
                                        hold = 0
                                        for k in range(1, min(len("<memory_op>"), len(combined)) + 1):
                                            if "<memory_op>".startswith(combined[-k:]):
                                                hold = k
                                        if hold:
                                            pending_tag[fld] = combined[-hold:]
                                            emit = combined[:-hold]
                                        else:
                                            emit = combined
                                        if emit:
                                            d[fld] = emit
                                        else:
                                            d.pop(fld, None)
                                            drop_chunk = True
                                    if not touched:
                                        self.wfile.write(raw_line)
                                        self.wfile.flush()
                                    elif not drop_chunk or d:
                                        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))
                                        self.wfile.flush()
                            except BrokenPipeError:
                                pass
                    else:
                        out_obj = self._collect_non_stream(resp2)
                        msg = (out_obj.get("choices") or [{}])[0].get("message") or {}
                        inline_buf = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
                        if did_write:
                            for fld in ("reasoning_content", "content"):
                                if msg.get(fld) and "<memory_op>" in msg[fld]:
                                    msg[fld] = re.sub(r"\s*<memory_op>.*?</memory_op>\s*", "", msg[fld], flags=re.DOTALL).strip()
                        self._send_json(200, out_obj)

                # Inline `<memory_op>` fast commit (<5 ms!) with automatic fallback to non-daemon extractor
                if did_write:
                    m_op = re.search(r"<memory_op>(.*?)</memory_op>", inline_buf, flags=re.DOTALL)
                    if m_op:
                        t_in = time.perf_counter()
                        ops = parse_ops_resilient(m_op.group(1))
                        if ops:
                            with state.write_lock:
                                state.commit_ops(ops, ctx, write_meta, source="inline_adapter", elapsed_ms=(time.perf_counter() - t_in) * 1000)
                            state.end_pending_write()
                            write_dispatched = True
                            return
                    write_ctx = build_write_context(norm_msgs)
                    state.start_background_extractor(write_ctx, ctx, write_meta)
                    write_dispatched = True
            finally:
                if did_write and not write_dispatched:
                    state.end_pending_write()

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Looped FOK OpenAI-Compatible Memory Proxy")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--upstream", default="http://127.0.0.1:8189")
    ap.add_argument("--model", default="inkling-stage-e", help="Upstream Loom model id")
    ap.add_argument("--public-model", default="inkling-fok-memory", help="Public OpenAI model id")
    ap.add_argument("--store", default=str(Path.home() / ".local/share/looped-fok/memories.json"))
    ap.add_argument("--reader", choices=["mlp", "lin"], default="mlp")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")
    state = FOKProxyState(args.upstream, args.model, args.public_model, args.store, reader=args.reader)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(state))

    def _graceful_stop(signum, _frame) -> None:
        LOG.info("Received signal %d — shutting down HTTP server and draining pending writes...", signum)
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _graceful_stop)
    signal.signal(signal.SIGTERM, _graceful_stop)

    LOG.info("Looped FOK proxy listening on http://%s:%d -> %s (model=%s, store=%s, active=%d)",
             args.host, args.port, args.upstream, args.model, args.store, len(state.store.all_active_unfiltered()))
    try:
        srv.serve_forever()
    finally:
        state.drain_pending_writes()
        srv.server_close()
        LOG.info("Looped FOK proxy stopped cleanly.")


if __name__ == "__main__":
    main()
