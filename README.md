# Looped FOK Memory

**Unified Single-Pass On-GPU Read+Write Feeling-of-Knowing (FOK) Memory for Looped Transformers**

> Zero-text-token memory gating, `0.0 ms` Write-FOK overhead, `100%` silent bit-identity on non-memory turns, and a clean policy-decoupled memory sidecar validated on **Inkling 2 Mini** (`inkling2_loop`, Loom 0.3.0).

![Unified Single-Pass On-GPU Read+Write Looped FOK Architecture](docs/assets/architecture_overview.png)

---

## Why Looped FOK?

Standard RAG and always-on memory prompts dump retrieved context into every turn. Even when a user asks an unrelated coding or general question, injected memories bloat latency, shift token distributions, and cause **context hijacks**.

**Looped FOK** exploits the native 3-pass recurrent bank of a looped transformer (Pass 0 → Pass 1 → Pass 2 → Coda):
1. **Orthogonal Pass-0 Metacognition:** Inside the Pass-0 bank `[l_out-11; l_out-17]`, the model represents *"I need to look up a memory"* (`Read-FOK`) and *"The user just stated a durable fact worth remembering"* (`Write-FOK`) along **two nearly orthogonal linear axes (`cos = -0.0124`)**.
2. **Cross-Pass Internal Steering (`+0.35 · v_oracle`):** When Pass-0 Read-FOK fires, a prefill-only bank injection steers Pass 1 and Pass 2 to reverse Coda collapse, allowing a lightweight on-GPU 64-unit SiLU System-2 Coda Reader to verify whether retrieval is genuinely needed (**`91.7%` forced-FP rejection**).
3. **11-Token Native Quadrant Protocol (`200011..200021`):** In **one single prefill pass on one single model**, the GPU signals all four `(Read, Write)` quadrants with zero text scent tokens:
   - `(0, 0)` **Untouched** → `0.0 ms` overhead, **`100%` silent bit-exact** to the base model.
   - `(1, 0)` **`200011..200015`** → Category-routed memory dereference, `0` write extractor calls.
   - `(0, 1)` **`200016`** → Normal reply + async post-turn `add` memory extraction (works even when the store is empty via Write-Only `gate_mode = 7`).
   - `(1, 1)` **`200017..200021`** → Category-routed dereference + async `update`/`delete` extraction in **one token**.
4. **FOK-Gated Post-Coda SwiGLU Adapter (`14.15 MB` GGUF, `magic = 0x7`):** Applied on-GPU to `result_norm` right before `lm_head` **only on Memory-Active turns** (`Val CE: 2.9453 → 0.8313`, `Top-1: 81.34% → 91.60%`, `Val KL = 0.3523`, `||Δh||/||h|| = 4.33%`). Makes memory acknowledgments, updates, and recalls completely natural with **zero *"I don't have persistent memory"* disclaimers** while preserving `100%` silent bit-identity on `(0, 0)` turns (`19.95 MB` total drop-in sidecars).
5. **Transparent OpenAI-Compatible Proxy (`looped_fok.proxy`):** Drop-in HTTP+SSE proxy (`127.0.0.1:8190/v1`) that gives any memory-less CLI/web coding harness (e.g., `dsh` / DeepSeek Harness, Aider, Cline) persistent cross-session memory with automatic `role == "tool"` passthrough, JSON disk persistence, and per-repo `ScopePolicyFilter` isolation.

---

## Verified Benchmark Results

![Looped FOK Memory Verified Benchmarks](docs/assets/benchmarks_infographic.png)

### 1. General Personal Memory (`persona_test`, 12 Pos + 12 Hard Neg, completely unseen)

| Arm (`0` text scent tokens) | Pos Deref | Pos Grounded | Avg Memories Injected | Neg Hijacks | Silent Bit-Exact vs Base | Wall Time |
|---|---:|---:|---:|---:|---:|---:|
| `base` (no memory) | `0/12` | `7/12` (`0/5` personal) | `0.00 / 4` | `0` | `12/12` | `135.2 s` |
| `always_on_rag` (all 4 dumped) | n/a | `12/12` | `4.00 / 4` | **`2`** | `1/12` (`8.3%`) | `139.0 s` |
| **`general_fok_mlp` (Looped FOK)** | **`11/12`** | **`12/12`** | **`1.00 / 4` (`4×` selective)** | **`0`** | **`12/12` (`100%`)** | **`124.9 s`** |
| **`general_fok_fp_stress` (forced ON)** | n/a | n/a | — | **`0`** (`11/12 = 91.7%` rejected) | — | `57.7 s` |

### 2. Autonomous Coding Agent (`sentinel-telemetry`, 18-Turn Empty-Store Session, Single Model)

| Metric (`models_loaded = 1`, starts from `0` memories) | Result |
|---|---:|
| **Write-FOK Extra Tap Overhead** | **`0.0 ms`** (unified in single prefill pass) |
| **Write-FOK Gate** (`add` / `update` / `delete` vs Q&A & transient variable renames) | **`5/5` TP, `0/11` FP** |
| **Read-FOK Gate** (On-GPU System-2 MLP64) | **`6/6` TP, `0/10` FP** |
| **Grounded Coding & Agentic Recalls** (`Podman`, `-tags=unit`/`9091`, Conventional Commit, `ClickHouse Local`/`9099`) | **`4/4` (`100%`)** |
| **Autonomous Memory Lifecycle** (`0 → 4 → 3` active memories, zero stale ghosting) | **`100%` exact** |
| **Silent Bit-Identity** (7 ephemeral/negative turns vs `base`) | **`7/7` (`100%`)** |
| **Multi-Repo Isolation** (`ScopePolicyFilter`: `other-repo` vs `sentinel-telemetry`) | **`2/2` (`100%`)** |

---

## Single-Pass Execution Flow

```mermaid
sequenceDiagram
    participant U as User / Coding Agent
    participant G as Inkling 2 Mini (Single Loom Model)
    participant S as GeneralMemoryStore + PolicyFilter

    U->>G: Turn prompt (fok_ctrl: RW if store non-empty, else Write-Only gate_mode=7)
    Note over G: Pass 0 Bank [l_out-11; l_out-17]<br/>Read-FOK (probe_w[0]) ⊥ Write-FOK (probe_w[1])
    alt (Read=0, Write=0) Ephemeral / Negative
        G-->>U: Untouched base reply (0.0 ms overhead, 100% silent bit-exact)
    else (Read=1, Write=0) Pure Recall
        G-->>S: Emit 200011..200015 (category token)
        S-->>G: Selective category + safety-coupled memories
        G-->>U: Grounded reply (0 write calls)
    else (Read=0, Write=1) Pure Write
        G-->>U: Emit 200016 -> Normal reply
        G->>S: Async structured JSON `add` commit
    else (Read=1, Write=1) Update / Delete
        G-->>S: Emit 200017..200021 (category + write in 1 token)
        S-->>G: Active memories -> Grounded reply
        G->>S: Async structured JSON `update` / `delete` commit
    end
```

---

## Quickstart

```bash
pip install -e ".[dev]"
pytest
```

### Run the OpenAI-Compatible Memory Proxy (for `dsh` / CLI Coding Harnesses)

```bash
python3 -m looped_fok.proxy \
  --host 127.0.0.1 --port 8190 \
  --upstream http://127.0.0.1:8189 \
  --model inkling-stage-e \
  --public-model inkling-fok-memory \
  --store ~/.local/share/looped-fok/memories.json
```

Use [`examples/fok-dsh.patch.yml`](examples/fok-dsh.patch.yml) to test-drive with `dsh` (DeepSeek Harness) with zero edits to `dsh`:

```bash
dsh --profile headless --patch examples/fok-dsh.patch.yml "your coding task"

# Inspect or clear persisted memories anytime:
curl -sS http://127.0.0.1:8190/fok/memories | jq .
curl -sS -X DELETE http://127.0.0.1:8190/fok/memories
```

### Two-Layer Architecture: General Memory Core + Pluggable Policy

The core `GeneralMemoryStore` has **zero alignment/trust/authority coupling**. When you need multi-repo isolation, tenant boundaries, or owner-private visibility overrides, plug in a `PolicyFilter` (such as `ScopePolicyFilter`) with zero changes to the FOK core or GPU kernels:

```python
from looped_fok import GeneralMemoryStore, ScopePolicyFilter, UnifiedFOKClient

store = GeneralMemoryStore(policy=ScopePolicyFilter())
client = UnifiedFOKClient(base_url="http://127.0.0.1:8189", model="inkling-stage-d2")

ctx = {"tenant_id": "alex", "allowed_scopes": ["repo:sentinel-telemetry", "global"]}
res = client.chat_turn(
    store,
    "I'm working on sentinel-telemetry: Go 1.23 + Podman rootless (never Docker).",
    context=ctx,
    write_metadata={"tenant_id": "alex", "scope": "repo:sentinel-telemetry"},
)
```

See [`examples/quickstart_coding_agent.py`](examples/quickstart_coding_agent.py), [`examples/custom_policy_filter.py`](examples/custom_policy_filter.py), and [`docs/architecture.md`](docs/architecture.md) for full details.

---

## License

MIT — see [`LICENSE`](LICENSE).
