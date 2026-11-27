# Looped FOK Server Deployment & Artifact Manifest

This repository ships the **Python memory store, policy filter, client, and OpenAI-compatible proxy** (`looped_fok`). To run the full on-GPU Read+Write FOK loop, deploy a compatible **Loom 0.3.0** server hosting **Inkling 2 Mini** (`inkling2_loop`) with the two lightweight FOK sidecar GGUF artifacts below.

---

## Pinned Host Stack & Artifact Checksums

| Artifact | Size | SHA-256 | Role |
|---|---:|---|---|
| **Base model:** `unsloth/Inkling-2-Mini` (`loom-fun__unsloth-Inkling-2-Mini-loom-b5e33e1`, `Inkling-2-Mini-UD-Q4_K_XL.gguf`) | base | *(upstream Loom package)* | Frozen 3-pass looped MoE backbone |
| **Runtime:** Loom `0.3.0` (`loom-serve`) | binary | *(upstream Loom 0.3.0)* | GPU inference runtime (tested on AMD Radeon 8060S `gfx1151`) |
| **Sidecar 1:** `fok_bank_coding.gguf` | `5.84 MB` | `d991de6fe9b8e2d6f761828ab0f1c6f5e53e29c90e2155497d95a0228f12aadd` | 9-tensor F32 GGUF: Pass-0 Read/Write/Category probes, `v_oracle`, System-2 Coda MLP64 |
| **Sidecar 2:** `fok_coda_adapter.gguf` | `14.16 MB` | `e7841bdb313b89ee3f8ab74ddf2ebca650f2777583644b635b068079cb96430e` | 3-tensor F32 GGUF: Stage E.2 KL-regularized Post-Coda SwiGLU Residual Adapter (`hidden=512`, `magic=0x7`) |

*(Total sidecar download: **`19.95 MB`**, uploaded as GitHub Release assets alongside the Stage E `llm.loom` / `deploy_stage_e.json` patch.)*

---

## Minimal End-to-End Reproduction

```bash
# 1. Start the Loom 0.3.0 Stage E server on loopback :8189
loom-serve \
  --device=amdgpu://?completion_spin_ns=4294967295 \
  --pool_backing=fixed --parameter_mode=file \
  --listen=127.0.0.1:8189 \
  --model=/path/to/deploy_stage_e.json

# 2. Start the OpenAI-compatible Looped FOK proxy on loopback :8190
python3 -m looped_fok.proxy \
  --host 127.0.0.1 --port 8190 \
  --upstream http://127.0.0.1:8189 \
  --model inkling-stage-e \
  --public-model inkling-fok-memory \
  --store ~/.local/share/looped-fok/memories.json

# 3. Verify Write -> Ephemeral -> Cross-Session Recall (e.g. via dsh headless)
dsh --profile headless --patch examples/fok-dsh.patch.yml \
  "Remember for my project nova-edge: Go 1.23 with Podman rootless (never Docker), tests use -tags=unit against 127.0.0.1:9091."
dsh --profile headless --patch examples/fok-dsh.patch.yml \
  "Explain TCP FIN vs RST in 2 sentences."
dsh --profile headless --patch examples/fok-dsh.patch.yml \
  "What container runtime and test command should my coding agent use for nova-edge?"
```
