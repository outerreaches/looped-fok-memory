"""
Production-Scale Bounded Memory Store (`SQLiteMemoryStore`) with FTS5 BM25 Top-k Retrieval.

Drop-in replacement for `GeneralMemoryStore` backed by stdlib `sqlite3` (WAL mode + FTS5):
  - True O(log N) incremental ACID writes (`with store.batch():` single SQLite transaction).
  - Bounded `O(k)` retrieval (`query_by_category(..., top_k=8)`) combining on-GPU routed category,
    safety coupling (`project_state` <-> `fact_constraint`), and FTS5 BM25 lexical scoring —
    never dumps the full store into the prompt even at 100,000+ active memories!
  - Bounded `query_for_write(..., top_k=12)` so the Write-FOK extractor sees only the most relevant
    existing memories when deciding `add` / `update` / `delete` at scale.
  - Preserves the exact `PolicyFilter` / fail-closed `ScopePolicyFilter` interface with zero
    changes to the GPU kernels or FOK sidecars.
"""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterator

from .store import MemoryItem, PolicyFilter


def _fts_query_from_text(text: str, max_terms: int = 16) -> str:
    """Build a safe OR-joined FTS5 MATCH query from user text (no syntax injection)."""
    raw_toks = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9._:-]{1,}", text.lower())
    seen: set[str] = set()
    terms: list[str] = []
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "what", "which", "where",
        "when", "should", "would", "could", "have", "about", "into", "your", "our",
        "please", "remember", "update", "forget", "notes", "note", "just", "now", "use",
    }
    for tok in raw_toks:
        # Split compound identifiers like `sentinel-telemetry` or `127.0.0.1:9099` into clean FTS tokens
        for sub in re.findall(r"[a-z0-9]{2,}", tok):
            if sub in stop or sub in seen:
                continue
            seen.add(sub)
            terms.append(f'"{sub}"')
            if len(terms) >= max_terms:
                return " OR ".join(terms)
    return " OR ".join(terms)


class SQLiteMemoryStore:
    def __init__(
        self,
        db_path: str | Path,
        policy: PolicyFilter | None = None,
        default_top_k: int = 8,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._policy = policy or PolicyFilter()
        self.default_top_k = default_top_k
        self._in_batch = False
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                item_id       TEXT PRIMARY KEY,
                category      TEXT NOT NULL,
                key           TEXT NOT NULL,
                value         TEXT NOT NULL,
                tags_json     TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'active',
                version       INTEGER NOT NULL DEFAULT 1,
                supersedes    TEXT,
                tenant_id     TEXT,
                scope         TEXT NOT NULL DEFAULT 'default',
                visibility    TEXT,
                updated_at    REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_active_scope ON memories(status, tenant_id, scope, category, updated_at DESC)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_active_cat ON memories(status, category, updated_at DESC)"
        )
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                item_id UNINDEXED,
                key,
                value,
                tags,
                tokenize='unicode61'
            )
            """
        )

    @contextlib.contextmanager
    def batch(self) -> Iterator[None]:
        """Atomic multi-operation SQLite transaction (all-or-nothing)."""
        if self._in_batch:
            yield
            return
        self._conn.execute("BEGIN IMMEDIATE")
        self._in_batch = True
        try:
            yield
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")
        finally:
            self._in_batch = False

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            item_id=row["item_id"],
            category=row["category"],
            key=row["key"],
            value=row["value"],
            tags=json.loads(row["tags_json"]),
            metadata=json.loads(row["metadata_json"]),
            status=row["status"],
            version=row["version"],
            supersedes=row["supersedes"],
        )

    def _upsert_fts(self, item: MemoryItem) -> None:
        self._conn.execute("DELETE FROM memories_fts WHERE item_id = ?", (item.item_id,))
        if item.status == "active":
            self._conn.execute(
                "INSERT INTO memories_fts(item_id, key, value, tags) VALUES (?, ?, ?, ?)",
                (item.item_id, item.key.replace("_", " "), item.value, " ".join(item.tags)),
            )

    def max_numeric_id(self) -> int:
        cur = self._conn.execute("SELECT item_id FROM memories WHERE item_id LIKE 'm_%'")
        mx = 0
        for (iid,) in cur:
            m = re.match(r"m_(\d+)$", iid)
            if m:
                mx = max(mx, int(m.group(1)))
        return mx

    def migrate_from_json(self, json_path: str | Path) -> int:
        """Import items from a legacy `memories.json` file if the SQLite store is empty."""
        p = Path(json_path)
        if not p.exists():
            return 0
        count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if count > 0:
            return 0
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data.get("items", [])
        if not items:
            return 0
        now = time.time()
        with self.batch():
            for d in items:
                it = MemoryItem(**d)
                meta = it.metadata or {}
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO memories
                    (item_id, category, key, value, tags_json, metadata_json, status, version, supersedes, tenant_id, scope, visibility, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        it.item_id, it.category, it.key, it.value,
                        json.dumps(it.tags), json.dumps(meta),
                        it.status, it.version, it.supersedes,
                        meta.get("tenant_id"), meta.get("scope", "default"), meta.get("visibility"),
                        now,
                    ),
                )
                self._upsert_fts(it)
        return len(items)

    def add(self, item: MemoryItem) -> None:
        now = time.time()
        meta = item.metadata or {}
        with self.batch():
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (item_id, category, key, value, tags_json, metadata_json, status, version, supersedes, tenant_id, scope, visibility, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.item_id, item.category, item.key, item.value,
                    json.dumps(item.tags), json.dumps(meta),
                    item.status, item.version, item.supersedes,
                    meta.get("tenant_id"), meta.get("scope", "default"), meta.get("visibility"),
                    now,
                ),
            )
            self._upsert_fts(item)

    def update(self, old_id: str, new_item: MemoryItem) -> None:
        now = time.time()
        with self.batch():
            row = self._conn.execute("SELECT version FROM memories WHERE item_id = ?", (old_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown item_id {old_id}")
            new_ver = int(row["version"]) + 1
            self._conn.execute("UPDATE memories SET status = 'superseded', updated_at = ? WHERE item_id = ?", (now, old_id))
            self._conn.execute("DELETE FROM memories_fts WHERE item_id = ?", (old_id,))
            meta = new_item.metadata or {}
            ni = MemoryItem(
                item_id=new_item.item_id,
                category=new_item.category,
                key=new_item.key,
                value=new_item.value,
                tags=list(new_item.tags),
                metadata=dict(meta),
                status="active",
                version=new_ver,
                supersedes=old_id,
            )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (item_id, category, key, value, tags_json, metadata_json, status, version, supersedes, tenant_id, scope, visibility, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ni.item_id, ni.category, ni.key, ni.value,
                    json.dumps(ni.tags), json.dumps(meta),
                    ni.status, ni.version, ni.supersedes,
                    meta.get("tenant_id"), meta.get("scope", "default"), meta.get("visibility"),
                    now,
                ),
            )
            self._upsert_fts(ni)

    def delete(self, item_id: str) -> None:
        now = time.time()
        with self.batch():
            cur = self._conn.execute("UPDATE memories SET status = 'deleted', updated_at = ? WHERE item_id = ?", (now, item_id))
            if cur.rowcount == 0:
                raise KeyError(f"Unknown item_id {item_id}")
            self._conn.execute("DELETE FROM memories_fts WHERE item_id = ?", (item_id,))

    def clear_all(self) -> None:
        with self.batch():
            self._conn.execute("DELETE FROM memories")
            self._conn.execute("DELETE FROM memories_fts")

    def _scope_where(self, context: dict | None) -> tuple[str, list]:
        """Push tenant/scope predicates into SQLite index when context is provided."""
        clauses = ["m.status = 'active'"]
        params: list = []
        if context:
            tenant = context.get("tenant_id")
            scopes = list(context.get("allowed_scopes") or [])
            if tenant is not None:
                clauses.append("m.tenant_id = ?")
                params.append(tenant)
            if scopes:
                ph = ",".join("?" for _ in scopes)
                clauses.append(f"m.scope IN ({ph})")
                params.extend(scopes)
        return " AND ".join(clauses), params

    def active_items(self, context: dict | None = None, limit: int | None = None) -> list[MemoryItem]:
        where, params = self._scope_where(context)
        sql = f"SELECT m.* FROM memories m WHERE {where} ORDER BY m.updated_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        raw = [self._row_to_item(r) for r in rows]
        return self._policy.filter_candidates(raw, context)

    def all_active_unfiltered(self, limit: int | None = None) -> list[MemoryItem]:
        sql = "SELECT * FROM memories WHERE status = 'active' ORDER BY updated_at DESC"
        params: list = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [self._row_to_item(r) for r in self._conn.execute(sql, params).fetchall()]

    def count_active(self, context: dict | None = None) -> int:
        if context is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM memories WHERE status = 'active'").fetchone()[0])
        return len(self.active_items(context))

    def has_active_memories(self, context: dict | None = None) -> bool:
        # Fast O(1) indexed check (still respects PolicyFilter fail-closed semantics!)
        where, params = self._scope_where(context)
        rows = self._conn.execute(f"SELECT m.* FROM memories m WHERE {where} LIMIT 16", params).fetchall()
        if not rows:
            return False
        return len(self._policy.filter_candidates([self._row_to_item(r) for r in rows], context)) > 0

    def dump_all_active(self, context: dict | None = None, limit: int | None = None) -> list[dict]:
        return [
            {"id": it.item_id, "version": it.version, "category": it.category, "key": it.key, "value": it.value}
            for it in self.active_items(context, limit=limit)
        ]

    def query_for_write(self, query_text: str, context: dict | None = None, top_k: int = 12) -> list[dict]:
        """Bounded top-k active memories for the Write-FOK extractor (scales to 100k+ items!)."""
        if not self.has_active_memories(context):
            return []
        where, params = self._scope_where(context)
        fts_q = _fts_query_from_text(query_text)
        picked: dict[str, MemoryItem] = {}

        if fts_q:
            sql = (
                f"SELECT m.* FROM memories_fts f JOIN memories m ON m.item_id = f.item_id "
                f"WHERE memories_fts MATCH ? AND {where} ORDER BY bm25(memories_fts) LIMIT ?"
            )
            rows = self._conn.execute(sql, [fts_q, *params, top_k * 2]).fetchall()
            for it in self._policy.filter_candidates([self._row_to_item(r) for r in rows], context):
                picked[it.item_id] = it
                if len(picked) >= top_k:
                    break

        if len(picked) < top_k:
            recent = self.active_items(context, limit=top_k * 2)
            for it in recent:
                if it.item_id not in picked:
                    picked[it.item_id] = it
                    if len(picked) >= top_k:
                        break

        return [
            {"id": it.item_id, "version": it.version, "category": it.category, "key": it.key, "value": it.value}
            for it in picked.values()
        ]

    def query_by_category(
        self,
        routed_category: str,
        user_text: str = "",
        context: dict | None = None,
        top_k: int | None = None,
    ) -> dict:
        """Bounded O(k) hybrid retrieval: FTS5 BM25 + on-GPU category routing + safety coupling."""
        k = top_k if top_k is not None else self.default_top_k
        where, params = self._scope_where(context)
        n_total = int(self._conn.execute(f"SELECT COUNT(*) FROM memories m WHERE {where}", params).fetchone()[0])
        if n_total == 0:
            return {"routed_category": routed_category, "n_total_active": 0, "n_surfaced": 0, "memories": []}

        # Determine primary + coupled categories
        low = user_text.lower()
        coupled_cats = {routed_category}
        if routed_category in ("project_state", "fact_constraint"):
            coupled_cats.update(("project_state", "fact_constraint"))
        elif routed_category == "user_preference" and any(
            w in low for w in ("dinner", "lunch", "food", "eat", "recipe", "restaurant", "snack", "cook", "meal")
        ):
            coupled_cats.add("fact_constraint")
        elif routed_category == "revision_update":
            coupled_cats.update(("schedule_event", "project_state", "fact_constraint", "user_preference"))

        # 1. Gather FTS5 BM25 top candidates (fast O(log N) inverted-index scan!)
        scored: dict[str, tuple[float, MemoryItem]] = {}
        fts_q = _fts_query_from_text(user_text)
        if fts_q:
            sql = (
                f"SELECT m.*, bm25(memories_fts) AS rank_score "
                f"FROM memories_fts f JOIN memories m ON m.item_id = f.item_id "
                f"WHERE memories_fts MATCH ? AND {where} "
                f"ORDER BY rank_score LIMIT ?"
            )
            rows = self._conn.execute(sql, [fts_q, *params, max(k * 4, 32)]).fetchall()
            filtered = self._policy.filter_candidates([self._row_to_item(r) for r in rows], context)
            rank_map = {r["item_id"]: float(r["rank_score"]) for r in rows}
            for it in filtered:
                # SQLite bm25() is more negative = better match; convert to positive relevance
                rel = -rank_map.get(it.item_id, 0.0)
                bonus = 2.0 if it.category == routed_category else (1.0 if it.category in coupled_cats else 0.0)
                scored[it.item_id] = (rel + bonus, it)

        # 2. If fewer than k lexical hits, top up with most recent items in coupled categories (bounded by k!)
        if len(scored) < k:
            cat_list = sorted(coupled_cats)
            ph = ",".join("?" for _ in cat_list)
            sql = f"SELECT m.* FROM memories m WHERE {where} AND m.category IN ({ph}) ORDER BY m.updated_at DESC LIMIT ?"
            rows = self._conn.execute(sql, [*params, *cat_list, max(k * 2, 16)]).fetchall()
            for idx, it in enumerate(self._policy.filter_candidates([self._row_to_item(r) for r in rows], context)):
                if it.item_id not in scored:
                    bonus = 1.5 if it.category == routed_category else 0.5
                    scored[it.item_id] = (bonus - idx * 0.01, it)

        # 3. Final fallback (still strictly bounded to k!) if routed category was empty
        if not scored:
            for idx, it in enumerate(self.active_items(context, limit=k)):
                scored[it.item_id] = (-idx * 0.01, it)

        ranked = [it for _, it in sorted(scored.values(), key=lambda x: x[0], reverse=True)][:k]
        selected = self._policy.post_select(ranked, context)

        return {
            "routed_category": routed_category,
            "n_total_active": n_total,
            "n_surfaced": len(selected),
            "memories": [
                {"id": it.item_id, "version": it.version, "category": it.category, "key": it.key, "value": it.value}
                for it in selected
            ],
        }
