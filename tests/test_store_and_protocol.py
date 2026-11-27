#!/usr/bin/env python3
"""Unit tests for `GeneralMemoryStore`, `ScopePolicyFilter`, `pack_ctrl`, and the 11-token quadrant protocol."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from looped_fok import (
    CATEGORY_LIST,
    UNIFIED_TOKENS,
    GeneralMemoryStore,
    MemoryItem,
    ScopePolicyFilter,
    pack_ctrl,
)
from looped_fok.proxy import (
    normalize_loom_payload,
    pack_adapter_ctrl,
    parse_ops_resilient,
)


class TestGeneralMemoryStore(unittest.TestCase):
    def test_lifecycle_add_update_delete(self) -> None:
        store = GeneralMemoryStore()
        self.assertFalse(store.has_active_memories())

        store.add(MemoryItem("m_01", "project_state", "stack", "Go 1.23 + DuckDB", ["go", "duckdb"]))
        self.assertTrue(store.has_active_memories())
        self.assertEqual(len(store.active_items()), 1)

        store.update("m_01", MemoryItem("m_02", "project_state", "stack", "Go 1.23 + ClickHouse Local", ["go", "clickhouse"]))
        active = store.active_items()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].item_id, "m_02")
        self.assertEqual(active[0].version, 2)
        self.assertEqual(active[0].supersedes, "m_01")

        store.delete("m_02")
        self.assertFalse(store.has_active_memories())
        self.assertEqual(store.dump_all_active(), [])

    def test_engineering_safety_coupling(self) -> None:
        store = GeneralMemoryStore()
        store.add(MemoryItem("m_01", "project_state", "sentinel_stack", "Go 1.23 + ClickHouse Local", ["go", "clickhouse"]))
        store.add(MemoryItem("m_02", "fact_constraint", "podman_only", "Use Podman rootless, never Docker", ["podman", "docker"]))
        store.add(MemoryItem("m_03", "user_preference", "coffee_style", "Pour-over coffee", ["coffee"]))

        # Routing to project_state must also pull coupled fact_constraint, NOT unrelated user_preference
        res = store.query_by_category("project_state", "Give me a container run command.")
        ids = {m["id"] for m in res["memories"]}
        self.assertEqual(ids, {"m_01", "m_02"})

    def test_scope_policy_isolation_and_owner_override(self) -> None:
        store = GeneralMemoryStore(policy=ScopePolicyFilter())
        store.add(MemoryItem(
            "m_s", "schedule_event", "thu_appt", "Busy Thursday", ["thursday"],
            metadata={"tenant_id": "u1", "scope": "shared", "visibility": "shared_summary_only"},
        ))
        store.add(MemoryItem(
            "m_p", "schedule_event", "thu_appt", "Physical therapy Thursday", ["thursday"],
            metadata={"tenant_id": "u1", "scope": "private", "visibility": "private_owner_only"},
        ))

        # Shared context sees only m_s
        r_shared = store.query_by_category("schedule_event", "Thursday?", {"tenant_id": "u1", "allowed_scopes": ["shared"]})
        self.assertEqual([m["id"] for m in r_shared["memories"]], ["m_s"])

        # Owner context overrides m_s with m_p
        r_owner = store.query_by_category("schedule_event", "Thursday?", {"tenant_id": "u1", "allowed_scopes": ["shared", "private"]})
        self.assertEqual([m["id"] for m in r_owner["memories"]], ["m_p"])

        # Other repo/tenant sees 0 memories
        self.assertFalse(store.has_active_memories({"tenant_id": "u2", "allowed_scopes": ["shared", "private"]}))


class TestUnifiedProtocol(unittest.TestCase):
    def test_eleven_token_alphabet(self) -> None:
        self.assertEqual(len(UNIFIED_TOKENS), 11)
        # 200011..200015 -> (Read=True, Write=False, cat)
        for i, cat in enumerate(CATEGORY_LIST):
            self.assertEqual(UNIFIED_TOKENS[f"<|unused_{200011 + i}|>"], (True, False, cat))
        # 200016 -> (Read=False, Write=True, None)
        self.assertEqual(UNIFIED_TOKENS["<|unused_200016|>"], (False, True, None))
        # 200017..200021 -> (Read=True, Write=True, cat)
        for i, cat in enumerate(CATEGORY_LIST):
            self.assertEqual(UNIFIED_TOKENS[f"<|unused_{200017 + i}|>"], (True, True, cat))

    def test_pack_ctrl_bit_fields(self) -> None:
        c_rw = pack_ctrl(0.35, vec_id=1, gate_mode=1, scope=9, reader_slot=0, is_lin=0)
        u_rw = c_rw & 0xFFFFFFFF
        self.assertEqual((u_rw >> 28) & 0xF, 0x6)
        self.assertEqual((u_rw >> 24) & 0xF, 9)
        self.assertEqual((u_rw >> 20) & 0xF, 1)
        self.assertEqual((u_rw >> 16) & 0xF, 1)
        self.assertEqual(u_rw & 0x1FFF, round(0.35 * 256))

        c_wo = pack_ctrl(0.0, vec_id=1, gate_mode=7, scope=9, reader_slot=0, is_lin=0)
        u_wo = c_wo & 0xFFFFFFFF
        self.assertEqual((u_wo >> 20) & 0xF, 7)
        self.assertEqual(u_wo & 0x1FFF, 0)

        c_ad = pack_adapter_ctrl(1.0)
        u_ad = c_ad & 0xFFFFFFFF
        self.assertEqual((u_ad >> 28) & 0xF, 0x7)
        self.assertEqual(u_ad & 0x1FFF, 256)

    def test_disk_persistence_and_proxy_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mem.json"
            s1 = GeneralMemoryStore(persist_path=p)
            s1.add(MemoryItem("m_01", "project_state", "stack", "Go 1.23", ["go"]))
            s2 = GeneralMemoryStore(persist_path=p)
            self.assertEqual(len(s2.active_items()), 1)
            self.assertEqual(s2.active_items()[0].value, "Go 1.23")
            s2.clear_all()
            self.assertFalse(GeneralMemoryStore(persist_path=p).has_active_memories())

        # Proxy payload normalization: developer->system + trailing dsh runtime context reorder
        norm, is_aux = normalize_loom_payload(
            {
                "messages": [
                    {"role": "developer", "content": "You are a coding agent."},
                    {"role": "user", "content": [{"type": "text", "text": "Remember Go 1.23."}]},
                    {"role": "user", "content": "Current runtime context. Policy: workspace-write."},
                ],
                "max_completion_tokens": 512,
            },
            upstream_model="inkling-stage-e",
        )
        self.assertFalse(is_aux)
        self.assertEqual([m["role"] for m in norm["messages"]], ["system", "system", "user"])
        self.assertEqual(norm["messages"][-1]["content"], "Remember Go 1.23.")
        self.assertEqual(norm["max_tokens"], 512)

        # Resilient JSON op parser recovers valid object even if another object is malformed
        raw = '[{"op":"add","key":"a","value":"ok"},{"op":"add","key":"b","value":"broken "quote""}]'
        ops = parse_ops_resilient(raw)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["key"], "a")


if __name__ == "__main__":
    unittest.main()
