#!/usr/bin/env python3
"""
Example: Plugging a Custom Alignment / Access-Control Policy onto `GeneralMemoryStore`.

Demonstrates the two-layer architecture: the FOK core and GPU kernels stay 100% unchanged
while a pluggable `PolicyFilter` enforces multi-repo isolation, owner-private overrides,
or custom trust/authority boundaries.
"""

from __future__ import annotations

from looped_fok import GeneralMemoryStore, MemoryItem, ScopePolicyFilter


def main() -> None:
    store = GeneralMemoryStore(policy=ScopePolicyFilter())

    # Add a shared summary and an owner-private detail for the same topic
    store.add(MemoryItem(
        item_id="m_shared",
        category="schedule_event",
        key="thursday_appointment",
        value="Thursday 14:00 blocked (personal appointment).",
        tags=["thursday", "schedule"],
        metadata={"tenant_id": "alice", "scope": "shared", "visibility": "shared_summary_only"},
    ))
    store.add(MemoryItem(
        item_id="m_private",
        category="schedule_event",
        key="thursday_appointment",
        value="Thursday 14:00 weekly physical therapy at Downtown Clinic.",
        tags=["thursday", "schedule", "therapy"],
        metadata={"tenant_id": "alice", "scope": "private", "visibility": "private_owner_only"},
    ))

    # 1. Shared context (e.g., family or team view): sees ONLY the redacted summary
    shared_ctx = {"tenant_id": "alice", "allowed_scopes": ["shared"]}
    print("Shared view:", store.query_by_category("schedule_event", "Am I free Thursday?", shared_ctx)["memories"])

    # 2. Owner-private context: private detail cleanly overrides the redundant shared summary
    owner_ctx = {"tenant_id": "alice", "allowed_scopes": ["shared", "private"]}
    print("Owner view: ", store.query_by_category("schedule_event", "Am I free Thursday?", owner_ctx)["memories"])

    # 3. Other tenant / other repo: 0 visible memories -> FOK Read gate short-circuits completely!
    other_ctx = {"tenant_id": "bob", "allowed_scopes": ["shared", "private"]}
    print("Other tenant has_active_memories:", store.has_active_memories(other_ctx))


if __name__ == "__main__":
    main()
