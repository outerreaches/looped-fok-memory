"""
General-Memory Looped FOK Sidecar (`GeneralMemoryStore`) with Pluggable Policy Layer.

Two-layer separation:
  1. Core (`GeneralMemoryStore`): O(1) `add`, `update` (versioned supersede), and `delete`
     (exact tombstone), empty-store short-circuit, category-routed retrieval with
     safety coupling (`project_state` <-> `fact_constraint`, food preferences -> constraints)
     and cross-category key/tag lexical union.
  2. Optional Policy Layer (`PolicyFilter`): No-op by default; subclass (e.g. `ScopePolicyFilter`)
     to enforce multi-repo/tenant isolation, visibility masks, or alignment/trust rules with
     zero changes to the FOK core or GPU kernels.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

CATEGORY_LIST: list[str] = [
    "fact_constraint",
    "project_state",
    "revision_update",
    "schedule_event",
    "user_preference",
]

CATEGORY_TOKENS: dict[str, str] = {
    f"<|unused_{200011 + i}|>": cat for i, cat in enumerate(CATEGORY_LIST)
}


@dataclass
class MemoryItem:
    item_id: str
    category: str                    # one of CATEGORY_LIST
    key: str                         # short snake_case topic, e.g. "sentinel_telemetry_stack"
    value: str                       # concise factual memory content
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # extension bag for PolicyFilter (tenant_id, scope, visibility, ...)
    status: str = "active"           # "active" | "superseded" | "deleted"
    version: int = 1
    supersedes: str | None = None


class PolicyFilter:
    """Optional policy/alignment hook (no-op by default in general personal-memory mode)."""

    def filter_candidates(self, items: list[MemoryItem], context: dict | None = None) -> list[MemoryItem]:
        return items

    def post_select(self, selected: list[MemoryItem], context: dict | None = None) -> list[MemoryItem]:
        return selected


class ScopePolicyFilter(PolicyFilter):
    """Drop-in scope/privacy plugin: hard tenant & scope masking + owner-private override."""

    def filter_candidates(self, items: list[MemoryItem], context: dict | None = None) -> list[MemoryItem]:
        if not context:
            return items
        allowed_scopes = set(context.get("allowed_scopes", []))
        tenant = context.get("tenant_id")
        out = []
        for it in items:
            if tenant is not None and it.metadata.get("tenant_id") != tenant:
                continue
            if allowed_scopes and it.metadata.get("scope", "default") not in allowed_scopes:
                continue
            out.append(it)
        return out

    def post_select(self, selected: list[MemoryItem], context: dict | None = None) -> list[MemoryItem]:
        priv_keys = {it.key.lower() for it in selected if it.metadata.get("visibility") == "private_owner_only"}
        if not priv_keys:
            return selected
        return [
            it for it in selected
            if not (it.metadata.get("visibility") == "shared_summary_only" and it.key.lower() in priv_keys)
        ]


class GeneralMemoryStore:
    def __init__(self, policy: PolicyFilter | None = None, persist_path: str | Path | None = None) -> None:
        self._items: dict[str, MemoryItem] = {}
        self._policy = policy or PolicyFilter()
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self.load_json(self._persist_path)

    def save_json(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self._persist_path
        if not target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        payload = {"items": [asdict(it) for it in self._items.values()]}
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)

    def load_json(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        self._items = {d["item_id"]: MemoryItem(**d) for d in data.get("items", [])}

    def clear_all(self) -> None:
        self._items.clear()
        self.save_json()

    def add(self, item: MemoryItem) -> None:
        self._items[item.item_id] = copy.deepcopy(item)
        self.save_json()

    def update(self, old_id: str, new_item: MemoryItem) -> None:
        """O(1) versioned update: supersedes old_id and activates new_item."""
        if old_id not in self._items:
            raise KeyError(f"Unknown item_id {old_id}")
        old = self._items[old_id]
        old.status = "superseded"
        ni = copy.deepcopy(new_item)
        ni.version = old.version + 1
        ni.supersedes = old_id
        ni.status = "active"
        self._items[ni.item_id] = ni
        self.save_json()

    def delete(self, item_id: str) -> None:
        """O(1) exact tombstone delete."""
        if item_id not in self._items:
            raise KeyError(f"Unknown item_id {item_id}")
        self._items[item_id].status = "deleted"
        self.save_json()

    def active_items(self, context: dict | None = None) -> list[MemoryItem]:
        raw = [it for it in self._items.values() if it.status == "active"]
        return self._policy.filter_candidates(raw, context)

    def has_active_memories(self, context: dict | None = None) -> bool:
        return len(self.active_items(context)) > 0

    def dump_all_active(self, context: dict | None = None) -> list[dict]:
        return [
            {"id": it.item_id, "version": it.version, "category": it.category, "key": it.key, "value": it.value}
            for it in self.active_items(context)
        ]

    def query_by_category(
        self,
        routed_category: str,
        user_text: str = "",
        context: dict | None = None,
    ) -> dict:
        """Selectively retrieve active memories matching the on-GPU routed category (+ coupled safety constraints)."""
        candidates = self.active_items(context)
        low = user_text.lower()
        q_words = {w for w in "".join(c if c.isalnum() else " " for c in low).split() if len(w) >= 4}
        selected = []
        for it in candidates:
            cat_match = (it.category == routed_category)
            rev_match = (routed_category == "revision_update") and (
                it.category in ("schedule_event", "project_state", "fact_constraint", "user_preference")
                and (any(t.lower() in low for t in it.tags) or it.key.lower() in low)
            )
            safety_match = (
                (
                    routed_category == "user_preference"
                    and it.category == "fact_constraint"
                    and any(w in low for w in ("dinner", "lunch", "food", "eat", "recipe", "restaurant", "snack", "cook", "meal"))
                )
                or (routed_category in ("project_state", "fact_constraint") and it.category in ("project_state", "fact_constraint"))
            )
            key_toks = {w for w in "".join(c if c.isalnum() else " " for c in it.key.lower()).split() if len(w) >= 5}
            tag_toks = {w for t in it.tags for w in "".join(c if c.isalnum() else " " for c in t.lower()).split() if len(w) >= 5}
            lex_match = bool(q_words & (key_toks | tag_toks))
            if cat_match or rev_match or safety_match or lex_match:
                selected.append(it)

        if not selected and candidates:
            tagged = [it for it in candidates if any(t.lower() in low for t in it.tags)]
            selected = tagged if tagged else candidates

        selected = self._policy.post_select(selected, context)
        return {
            "routed_category": routed_category,
            "n_total_active": len(candidates),
            "n_surfaced": len(selected),
            "memories": [
                {"id": it.item_id, "version": it.version, "category": it.category, "key": it.key, "value": it.value}
                for it in selected
            ],
        }
