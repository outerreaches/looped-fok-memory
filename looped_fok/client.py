"""
Unified Single-Model On-GPU Read+Write Looped FOK Client (`UnifiedFOKClient`).

Communicates with a single Loom 0.3.0 server (`inkling-stage-d2`) using the 11-token native
quadrant alphabet (`200011..200021`):
  - No signal token  -> (Read=0, Write=0): 100% silent bit-exact reply, 0 ms extra overhead.
  - `200011..200015` -> (Read=1, Write=0): category-routed deref, 0 extractor calls.
  - `200016`         -> (Read=0, Write=1): normal reply + async post-turn `add` extractor.
  - `200017..200021` -> (Read=1, Write=1): category-routed deref + async post-turn `update`/`delete` extractor.

When `store.has_active_memories(context)` is False, automatically sends `gate_mode = 7`
(Write-Only mode: Read-FOK off, `eff_alpha = 0`, 0% steering perturbation, Write-FOK active).
"""

from __future__ import annotations

import json
import re
import time
import urllib.request

from .store import CATEGORY_LIST, GeneralMemoryStore, MemoryItem

NEUTRAL_SYS = "You are a helpful assistant."
PF_MM = "<|message_model|>"

# token -> (did_read, did_write, routed_category)
UNIFIED_TOKENS: dict[str, tuple[bool, bool, str | None]] = {
    **{f"<|unused_{200011 + i}|>": (True, False, cat) for i, cat in enumerate(CATEGORY_LIST)},
    "<|unused_200016|>": (False, True, None),
    **{f"<|unused_{200017 + i}|>": (True, True, cat) for i, cat in enumerate(CATEGORY_LIST)},
}

EXTRACTOR_SYS = (
    "You are a precise memory-store updater for an AI assistant. "
    "Given the user's latest message and the list of currently active memories, output ONLY a valid JSON array "
    "of memory operations (`add`, `update`, or `delete`) for durable project facts, safety constraints, "
    "schedules, or user preferences explicitly stated by the user.\n"
    "Rules:\n"
    "- Output ONLY a JSON array — no prose, no explanation.\n"
    "- Each object MUST have keys: `op` (\"add\"|\"update\"|\"delete\"), `target_id` (existing id for update/delete, else null), "
    "`category` (one of \"project_state\", \"fact_constraint\", \"user_preference\", \"schedule_event\"), "
    "`key` (short snake_case), `value` (concise exact factual summary preserving exact versions, IPs, ports, flags, tool names), "
    "`tags` (array of 3-6 lowercase keywords).\n"
    "- For `update`: set `target_id` to the exact existing memory `id` being modified and preserve unchanged details.\n"
    "- For `delete`: set `target_id` to the exact existing memory `id` to remove.\n"
    "- If the user updated multiple existing memories in one message, emit one `update` object per memory.\n"
    "- If nothing durable should be stored, output `[]`."
)


def pack_ctrl(
    alpha: float,
    vec_id: int = 1,
    gate_mode: int = 1,
    scope: int = 9,
    reader_slot: int = 0,
    is_lin: int = 0,
) -> int:
    """Pack Stage D.2 32-bit `fok_ctrl` control word (magic=0x6)."""
    aq = max(0, min(8191, int(round(alpha * 256.0))))
    val = (
        (0x6 << 28)
        | ((scope & 0xF) << 24)
        | ((gate_mode & 0xF) << 20)
        | ((vec_id & 0xF) << 16)
        | ((reader_slot & 0x3) << 14)
        | ((is_lin & 0x1) << 13)
        | aq
    )
    return val - 0x100000000 if val >= 0x80000000 else val


class UnifiedFOKClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8189",
        model: str = "inkling-stage-d2",
        alpha: float = 0.35,
        reader: str = "mlp",
        reasoning_effort: str = "low",
        max_tokens: int = 768,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        is_lin = 1 if reader == "lin" else 0
        self.ctrl_rw = pack_ctrl(alpha, vec_id=1, gate_mode=1, scope=9, reader_slot=0, is_lin=is_lin)
        self.ctrl_write_only = pack_ctrl(0.0, vec_id=1, gate_mode=7, scope=9, reader_slot=0, is_lin=is_lin)
        self._next_id = 1

    def _stream(self, payload: dict, stop_on_signal: bool = True) -> dict:
        body = json.dumps({**payload, "stream": True}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        reasoning, content = "", ""
        sig_tok, did_read, did_write, routed_cat = None, False, False, None
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=180.0) as resp:
            for line in resp:
                s = line.decode("utf-8", "ignore").strip()
                if not s.startswith("data:"):
                    continue
                ds = s[5:].strip()
                if ds == "[DONE]":
                    break
                obj = json.loads(ds)
                ch = obj.get("choices") or []
                if not ch:
                    continue
                d = ch[0].get("delta") or {}
                if d.get("reasoning_content"):
                    reasoning += d["reasoning_content"]
                if d.get("content"):
                    content += d["content"]
                if stop_on_signal:
                    for tok_s, (dr, dw, cat) in UNIFIED_TOKENS.items():
                        if tok_s in reasoning or tok_s in content:
                            sig_tok, did_read, did_write, routed_cat = tok_s, dr, dw, cat
                            break
                    if sig_tok is not None:
                        break
        return {
            "reasoning": reasoning,
            "content": content,
            "sig_tok": sig_tok,
            "did_read": did_read,
            "did_write": did_write,
            "routed_cat": routed_cat,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }

    def _extract_and_apply(
        self,
        store: GeneralMemoryStore,
        user_text: str,
        context: dict | None,
        default_metadata: dict | None,
    ) -> list[dict]:
        prompt = (
            f"Currently active memories:\n{json.dumps(store.dump_all_active(context), indent=2)}\n\n"
            f"User message:\n{user_text}\n\n"
            "Output ONLY the JSON array of memory operations:"
        )
        r = self._stream(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": EXTRACTOR_SYS},
                    {"role": "user", "content": prompt},
                ],
                "reasoning_effort": self.reasoning_effort,
                "temperature": 0.0,
                "max_tokens": 1536,
                "fok_prefill": PF_MM,
            },
            stop_on_signal=False,
        )
        m = re.search(r"\[.*\]", r["content"].strip(), flags=re.DOTALL)
        ops: list[dict] = []
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list):
                    ops = [x for x in parsed if isinstance(x, dict)]
            except Exception:
                ops = []

        applied = []
        active_ids = {it.item_id for it in store.active_items(context)}
        meta = dict(default_metadata or {})
        for op in ops:
            kind = str(op.get("op", "")).lower().strip()
            cat = str(op.get("category", "project_state")).strip()
            if cat not in CATEGORY_LIST:
                cat = "project_state"
            key = str(op.get("key", "note")).strip() or "note"
            val = str(op.get("value", "")).strip()
            tags = [str(t).lower() for t in (op.get("tags") or []) if isinstance(t, str)]
            tid = op.get("target_id")

            if kind == "add" and val:
                nid = f"m_{self._next_id:02d}"
                self._next_id += 1
                store.add(MemoryItem(item_id=nid, category=cat, key=key, value=val, tags=tags, metadata=meta))
                active_ids.add(nid)
                applied.append({"op": "add", "id": nid, "category": cat, "key": key, "value": val})
            elif kind == "update" and tid in active_ids and val:
                nid = f"m_{self._next_id:02d}"
                self._next_id += 1
                store.update(tid, MemoryItem(item_id=nid, category=cat, key=key, value=val, tags=tags, metadata=meta))
                active_ids.discard(tid)
                active_ids.add(nid)
                applied.append({"op": "update", "old_id": tid, "new_id": nid, "category": cat, "key": key, "value": val})
            elif kind == "delete" and tid in active_ids:
                store.delete(tid)
                active_ids.discard(tid)
                applied.append({"op": "delete", "target_id": tid})
        return applied

    def chat_turn(
        self,
        store: GeneralMemoryStore,
        user_text: str,
        history: list[tuple[str, str]] | None = None,
        sys_prompt: str = NEUTRAL_SYS,
        context: dict | None = None,
        write_metadata: dict | None = None,
        auto_write: bool = True,
    ) -> dict:
        """Run one conversational turn with unified on-GPU Read+Write FOK gating."""
        turns = list(history or []) + [("user", user_text)]
        msgs = [{"role": "system", "content": sys_prompt}] + [{"role": r, "content": t} for r, t in turns]
        eff_ctrl = self.ctrl_rw if store.has_active_memories(context) else self.ctrl_write_only

        r1 = self._stream(
            {
                "model": self.model,
                "messages": msgs,
                "reasoning_effort": self.reasoning_effort,
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
                "fok_prefill": PF_MM,
                "fok_ctrl": eff_ctrl,
            },
            stop_on_signal=True,
        )

        surfaced_ids: list[str] = []
        if not r1["did_read"] and not r1["did_write"]:
            content = r1["content"]
            signal_ms = 0.0
        elif not r1["did_read"] and r1["did_write"]:
            r2 = self._stream(
                {
                    "model": self.model,
                    "messages": msgs,
                    "reasoning_effort": self.reasoning_effort,
                    "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                    "fok_prefill": PF_MM,
                },
                stop_on_signal=False,
            )
            content = r2["content"]
            signal_ms = r1["elapsed_ms"]
        else:
            qres = store.query_by_category(r1["routed_cat"], user_text, context)
            surfaced_ids = [m["id"] for m in qres["memories"]]
            inject = (
                f"<|message_model|><|content_thinking|>\n"
                f"[Relevant user memories inserted: {json.dumps(qres)}]\n"
                f"Use these memories where relevant, think briefly, and answer the user directly.\n"
            )
            r2 = self._stream(
                {
                    "model": self.model,
                    "messages": msgs,
                    "reasoning_effort": self.reasoning_effort,
                    "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                    "fok_prefill": inject,
                },
                stop_on_signal=False,
            )
            content = r2["content"]
            signal_ms = r1["elapsed_ms"]

        ops_applied: list[dict] = []
        if r1["did_write"] and auto_write:
            ops_applied = self._extract_and_apply(store, user_text, context, write_metadata)

        return {
            "content": content,
            "sig_tok": r1["sig_tok"],
            "read_deref": r1["did_read"],
            "write_fire": r1["did_write"],
            "routed_cat": r1["routed_cat"],
            "surfaced_ids": surfaced_ids,
            "signal_ms": signal_ms,
            "ops_applied": ops_applied,
        }
