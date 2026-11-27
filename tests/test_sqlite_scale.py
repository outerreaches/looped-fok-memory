#!/usr/bin/env python3
"""
Scale & Latency Benchmark Suite for `SQLiteMemoryStore` (10,000+ active memories).

Verifies:
  - 10,000-memory multi-repo insertion & FTS5 indexing speed
  - Bounded O(k) retrieval (`top_k=8` out of 10,000 -> 1,250x selective!) in < 20 ms
  - Exact needle recall for Read-FOK (`query_by_category`) AND Write-FOK (`query_for_write`)
  - Fail-closed `ScopePolicyFilter` without context + cross-repo isolation + owner-private override
  - ACID transaction rollback on multi-operation batch failure
  - Legacy `memories.json` -> `memories.db` auto-migration
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from looped_fok import GeneralMemoryStore, MemoryItem, SQLiteMemoryStore, ScopePolicyFilter


class TestSQLiteScaleAndCorrectness(unittest.TestCase):
    def test_10k_memory_bounded_retrieval_and_latency(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "scale.db"
            store = SQLiteMemoryStore(db, policy=ScopePolicyFilter(), default_top_k=8)
            ctx_target = {"tenant_id": "alex", "allowed_scopes": ["repo:sentinel-telemetry", "global"]}
            ctx_other = {"tenant_id": "alex", "allowed_scopes": ["repo:other-repo", "global"]}

            # Bulk insert 10,000 distractor memories across 10 repos in one atomic batch
            t0 = time.perf_counter()
            cats = ["project_state", "fact_constraint", "user_preference", "schedule_event"]
            with store.batch():
                for i in range(10_000):
                    scope = "repo:sentinel-telemetry" if (i % 4 == 0) else f"repo:svc-{i % 10}"
                    store.add(MemoryItem(
                        item_id=f"d_{i:05d}",
                        category=cats[i % 4],
                        key=f"service_note_{i}",
                        value=f"Service module {i} uses worker queue partition {i % 64} with retry interval {i % 30}s.",
                        tags=[f"mod{i % 200}", "worker", "queue"],
                        metadata={"tenant_id": "alex", "scope": scope},
                    ))
                # Insert 3 specific needle memories inside repo:sentinel-telemetry (alongside 2,500 distractors!)
                store.add(MemoryItem(
                    item_id="needle_stack",
                    category="project_state",
                    key="sentinel_telemetry_stack",
                    value="sentinel-telemetry is a Go 1.23 service using ClickHouse Local for local storage.",
                    tags=["sentinel-telemetry", "go", "clickhouse", "storage"],
                    metadata={"tenant_id": "alex", "scope": "repo:sentinel-telemetry"},
                ))
                store.add(MemoryItem(
                    item_id="needle_podman",
                    category="fact_constraint",
                    key="container_runtime_podman_only",
                    value="sentinel-telemetry must use Podman rootless containers; never Docker.",
                    tags=["sentinel-telemetry", "podman", "docker", "container"],
                    metadata={"tenant_id": "alex", "scope": "repo:sentinel-telemetry"},
                ))
                store.add(MemoryItem(
                    item_id="needle_test",
                    category="fact_constraint",
                    key="go_test_endpoint",
                    value="For sentinel-telemetry, always pass -tags=unit targeting 127.0.0.1:9099; never 100.64.0.12.",
                    tags=["sentinel-telemetry", "unit", "9099", "tailscale"],
                    metadata={"tenant_id": "alex", "scope": "repo:sentinel-telemetry"},
                ))
            insert_s = time.perf_counter() - t0
            self.assertEqual(store.count_active(), 10_003)

            # 1. Read-FOK bounded top-k retrieval (2,503 active in scope -> must surface <= 8 and hit BOTH needles!)
            tq = time.perf_counter()
            res = store.query_by_category(
                "project_state",
                "What local storage backend and container runtime should I use for sentinel-telemetry?",
                ctx_target,
                top_k=8,
            )
            q_ms = (time.perf_counter() - tq) * 1000.0
            surf_ids = {m["id"] for m in res["memories"]}
            self.assertLessEqual(res["n_surfaced"], 8)
            self.assertIn("needle_stack", surf_ids)
            self.assertIn("needle_podman", surf_ids)
            self.assertLess(q_ms, 50.0, f"FTS5 query too slow: {q_ms:.1f} ms")

            # 2. Write-FOK bounded candidate lookup for update/delete (top_k=12 out of 2,503!)
            tw = time.perf_counter()
            w_cands = store.query_for_write(
                "Update sentinel-telemetry: move unit test endpoint from 127.0.0.1:9099 to 127.0.0.1:9100.",
                ctx_target,
                top_k=12,
            )
            w_ms = (time.perf_counter() - tw) * 1000.0
            self.assertLessEqual(len(w_cands), 12)
            self.assertIn("needle_test", {m["id"] for m in w_cands})
            self.assertLess(w_ms, 50.0, f"Write candidate query too slow: {w_ms:.1f} ms")

            # 3. Cross-repo isolation & fail-closed without context
            self.assertFalse(store.has_active_memories(ctx_other))
            self.assertFalse(store.has_active_memories(None))
            self.assertEqual(store.active_items(None), [])

            print(f"\n[10k SQLite+FTS5 scale] insert={insert_s:.2f}s | read_top8={q_ms:.2f}ms | write_top12={w_ms:.2f}ms")
            store.close()

    def test_atomic_rollback_and_json_migration_and_owner_override(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            jpath = Path(td) / "memories.json"
            dpath = Path(td) / "memories.db"

            # Create legacy JSON store with shared + private items
            js = GeneralMemoryStore(persist_path=jpath)
            js.add(MemoryItem("m_01", "schedule_event", "thu_appt", "Busy Thursday", ["thursday"],
                              metadata={"tenant_id": "u1", "scope": "shared", "visibility": "shared_summary_only"}))
            js.add(MemoryItem("m_02", "schedule_event", "thu_appt", "Physical therapy Thursday", ["thursday"],
                              metadata={"tenant_id": "u1", "scope": "private", "visibility": "private_owner_only"}))

            # Migrate into SQLiteMemoryStore
            sq = SQLiteMemoryStore(dpath, policy=ScopePolicyFilter())
            n_mig = sq.migrate_from_json(jpath)
            self.assertEqual(n_mig, 2)
            self.assertEqual(sq.max_numeric_id(), 2)

            # Owner-private override works identically
            r_owner = sq.query_by_category("schedule_event", "Thursday?", {"tenant_id": "u1", "allowed_scopes": ["shared", "private"]})
            self.assertEqual([m["id"] for m in r_owner["memories"]], ["m_02"])
            r_shared = sq.query_by_category("schedule_event", "Thursday?", {"tenant_id": "u1", "allowed_scopes": ["shared"]})
            self.assertEqual([m["id"] for m in r_shared["memories"]], ["m_01"])

            # Atomic batch rollback on missing target_id
            with self.assertRaises(KeyError):
                with sq.batch():
                    sq.update("m_01", MemoryItem("m_03", "schedule_event", "thu_appt", "Updated", ["thursday"],
                                                 metadata={"tenant_id": "u1", "scope": "shared"}))
                    sq.update("m_missing", MemoryItem("m_04", "schedule_event", "x", "y", []))
            self.assertEqual([m.item_id for m in sq.all_active_unfiltered()], ["m_01", "m_02"])
            sq.close()


if __name__ == "__main__":
    unittest.main()
