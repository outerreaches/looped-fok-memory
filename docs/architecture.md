# Architecture Deep Dive

This document specifies the on-GPU mechanics, GGUF bank ABI, 32-bit control word layout, and 11-token native signaling protocol behind **Looped FOK Memory**.

---

## 1. Looped Transformer Geometry (`inkling2_loop`)

Inkling 2 Mini unrolls 24 physical layers into 48 execution slots:
- **Prelude:** layers `0..5`
- **Pass 0:** layers `6..17` → bank `[l_out-11; l_out-17]` (`4608`-dim concatenation)
- **Pass 1:** layers `24..35` → bank `[l_out-29; l_out-35]`
- **Pass 2:** layers `36..47` → bank `[l_out-41; l_out-47]`
- **Coda:** layers `18..23` → `result_norm` (`2304`-dim)

### Key Empirical Findings
1. **Orthogonal Read & Write Directions:** Linear probes trained on `[l_out-11; l_out-17]` for Read-FOK (`y_read`) and Write-FOK (`y_write`) have cosine similarity **`cos(w_read_p0, w_write_p0) = -0.0124`**, enabling simultaneous conflict-free gating from a single Pass-0 read.
2. **Universal Memory-Priming Axis:** The Pass-0 oracle priming vector `v_oracle = mean(B0_oracle[pos] - B0_neutral[pos])` (norm-matched per half) transfers strongly across domains:
   - `cos(v_coding_oracle, v_general_oracle) = +0.7863`
   - `cos(v_general_oracle, v_tribe_a_oracle) = +0.5784`
3. **System-2 Verification in Coda:** Injecting prefill-only `+0.35 · v_oracle` at Pass `0→1` and Pass `1→2` reverses Coda collapse so a 64-unit SiLU MLP over `[l_out-41; l_out-47; result_norm]` (`6912` dims) rejects `86.7%–91.7%` of adversarial forced-positive false triggers.

---

## 2. 9-Tensor FOK Bank GGUF v3 ABI (`5.8 MB`, F32)

| Tensor Name | Shape | Role |
|---|---:|---|
| `fok.probe_w` | `[8, 4608]` | Row `0`: Pass-0 Read-FOK; Row `1`: Pass-0 Write-FOK; Rows `2..6`: 5-way category classifier |
| `fok.probe_b` | `[8]` | Biases for `fok.probe_w` |
| `fok.vecs` | `[16, 4608]` | Slot `1`: norm-matched `v_oracle` steering vector |
| `fok.r_w1` | `[192, 6912]` | System-2 Coda MLP layer-1 weights (3 slots × 64 hidden units, StandardScaler folded in) |
| `fok.r_b1` | `[192]` | System-2 Coda MLP layer-1 biases |
| `fok.r_w2` | `[192]` | System-2 Coda MLP layer-2 weights |
| `fok.r_b2` | `[3]` | System-2 Coda MLP layer-2 biases |
| `fok.r_lw` | `[3, 6912]` | Optional linear Coda reader weights (3 slots) |
| `fok.r_lb` | `[3]` | Optional linear Coda reader biases |

---

## 3. 32-Bit `fok_ctrl` Control Word (`magic = 0x6`)

```
Bits 31..28 : magic = 0x6 (Unified Stage D.2 / B.3+ protocol)
Bits 27..24 : scope = 9   (prefill-only Pass 0->1 + Pass 1->2)
Bits 23..20 : gate_mode   (0 = force Read ON; 1 = Pass-0 Read+Write; 7 = Write-Only when store empty)
Bits 19..16 : vec_id = 1  (v_oracle slot)
Bits 15..14 : reader_slot (0..2)
Bit  13     : is_lin      (0 = 64-unit SiLU MLP; 1 = linear reader)
Bits 12..0  : alpha_q     (unsigned 13-bit fixed-point: alpha = alpha_q / 256.0)
```

---

## 4. Per-Span GPU Gate Word (`%fok_gate[span, 1]`) & 11-Token Alphabet

`@g_fok_bank_probe` latches all Pass-0 decisions into `%fok_gate[span, 1]`:
- Bits `0..3`: `vec_id`
- Bits `8..10`: `cat_id ∈ {0..4}` (`fact_constraint`, `project_state`, `revision_update`, `schedule_event`, `user_preference`)
- Bit `11` (`0x800`): `write_fired` (`probe_w[1] · B0 + probe_b[1] >= 0`, evaluated during prefill `span_len > 1`)
- Bit `30` (`0x40000000`): `read_fired` (Pass-0 Read-FOK, forced `0` when `gate_mode == 7`)

At the end of prefill, `@g_fok_coda_reader` combines `did_read` (System-2 Coda reader) and `did_write` (bit 11):

| Quadrant `(Read, Write)` | Emitted Token ID | Host Action |
|---|---|---|
| `(0, 0)` | *(untouched)* | Stream base reply directly (`0.0 ms` extra overhead, `100%` silent bit-exact) |
| `(1, 0)` | `200011 + cat_id` | Query `GeneralMemoryStore` with category + safety coupling; stream grounded reply |
| `(0, 1)` | `200016` | Stream normal reply (`fok_ctrl=None`) + run async post-turn `add` extractor |
| `(1, 1)` | `200017 + cat_id` | Query `GeneralMemoryStore` + stream grounded reply + run async `update`/`delete` extractor |

---

## 5. Stage E.2 FOK-Gated Post-Coda SwiGLU Residual Adapter (`14.15 MB`, `magic = 0x7`)

To make memory reads, writes, updates, and deletions feel completely natural without modifying any base weights of Inkling 2 Mini, Stage E.2 mounts a `hidden = 512` SwiGLU residual adapter (`3.54M` params, 3 F32 tensors: `fok.ad_wg [512, 2304]`, `fok.ad_wu [512, 2304]`, `fok.ad_wd [2304, 512]`, zero-initialized `W_down`) directly on-GPU via `@g_fok_coda_adapter`:

- **Placement:** Runs on `result_norm` (`%h`, `[tokens, 2304]`) right before `lm_head` (`output.weight`) in both `@llm_advance` (gated on `tok == span_last` for `~0.1 ms` prefill overhead even at 65K context) and `@llm_verify` (DSpark speculative decode).
- **Strict Runtime Gating (`magic = 0x7`):**
  - `magic != 0x7` (`(0, 0)` turns & `0x6` probe passes): returns immediately with zero writes → **`100.0%` silent bit-identity preserved**.
  - `magic == 0x7` (Memory-Active replays): applies `h[tok] += scale * W_down(SiLU(W_gate(h[tok])) * W_up(h[tok]))` in workgroup LDS (`2 KiB`).
- **KL-Regularized Length-Stratified Training:** Trained on `66,139` exact `Q4_K_XL` `result_norm` tokens (`480` dialogues, lengths up to `1,157` tokens `> 1,024` SWA window, with `dsh`-style multi-turn history and general fluency anchors) using `L = CE + 0.20·KL(p_base || p_ad) + 0.02·mean(||Δh||)` → `Val CE 2.9453 → 0.8313`, `Top-1 81.34% → 91.60%`, `Val KL = 0.3523`, `||Δh||/||h|| = 4.33%`.

---

## 6. OpenAI-Compatible Harness Proxy (`looped_fok.proxy`)

`looped_fok/proxy.py` exposes a standard OpenAI Chat Completions HTTP+SSE endpoint (`127.0.0.1:8190/v1`) in front of Loom 0.3.0 (`127.0.0.1:8189`):
- **Harness Normalization:** Converts `role: "developer"` → `"system"`, list-structured text content → string, bypasses auxiliary `session-title-llm` calls (`max_tokens <= 64`), and reorders trailing harness runtime-context snapshots (`"Current runtime context..."`) ahead of the user message so `span_last` sits directly after the real user turn.
- **Agentic Tool-Loop Guard:** Only fresh user turns (`messages[-1].role == "user"`) engage Read+Write FOK; intermediate `role == "tool"` turns pass straight through (`fok_ctrl = None`).
- **Zero-Race Persistence:** Atomic JSON disk persistence (`GeneralMemoryStore(persist_path=...)`) serialized with `write_lock` so immediate follow-up CLI invocations never race an in-flight background commit.
