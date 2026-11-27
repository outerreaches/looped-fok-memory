#!/usr/bin/env python3
"""
Quickstart: Empty-Store Autonomous Coding Agent using Unified On-GPU Read+Write Looped FOK.

Starts with 0 memories in `GeneralMemoryStore` (with `ScopePolicyFilter` plugged in for
multi-repo isolation) and runs a short interactive coding flow against a single Loom 0.3.0
`inkling-stage-d2` server.
"""

from __future__ import annotations

import argparse

from looped_fok import GeneralMemoryStore, ScopePolicyFilter, UnifiedFOKClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8189")
    ap.add_argument("--model", default="inkling-stage-d2")
    args = ap.parse_args()

    store = GeneralMemoryStore(policy=ScopePolicyFilter())
    client = UnifiedFOKClient(base_url=args.url, model=args.model)

    repo_ctx = {"tenant_id": "dev", "allowed_scopes": ["repo:demo-service", "global"]}
    write_meta = {"tenant_id": "dev", "scope": "repo:demo-service"}

    turns = [
        "I'm building `demo-service` in Go 1.23 with Podman rootless containers (never Docker).",
        "Write a quick Go function to deduplicate a slice of strings.",
        "What container build and run command should I use for `demo-service` locally?",
    ]

    for i, text in enumerate(turns, 1):
        res = client.chat_turn(store, text, context=repo_ctx, write_metadata=write_meta)
        print(f"\n--- Turn {i} ---")
        print(f"User:   {text}")
        print(f"Signal: tok={res['sig_tok']} read={res['read_deref']} write={res['write_fire']} sig_ms={res['signal_ms']} ops={res['ops_applied']}")
        print(f"Reply:  {res['content'][:280]}...")

    print("\nActive memories:", store.dump_all_active(repo_ctx))


if __name__ == "__main__":
    main()
