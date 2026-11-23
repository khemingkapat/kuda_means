## Summary
- **Stage / Scope:** *(e.g. V0 CPU / V1 Naive / V2 Memory / V3 Privatized / V4 MatMul / V5 Streaming / Docs)*
- **What changed:**

## Verification Checklist
### Automated (CPU / CI)
- [ ] `make cpu` builds cleanly
- [ ] `clang-format` / `ruff` / `shellcheck` pass

### GPU Verification *(required for PRs touching `src/v*`)*
- **Hardware Provenance:** *(e.g. RTX 4090 / `sm_89` / CUDA 12.2, or Tesla T4 / `sm_75`)*
- [ ] **Correctness vs. CPU baseline:**
  - `assignment_mismatches`: `0` *(or explain tie-breaking)*
  - `max_centroid_abs_diff`: `< 1e-3`
  - `sse_rel_diff`: `< 1e-3`
- [ ] **Sanitizers:** `compute-sanitizer --tool memcheck` (and `racecheck` / `synccheck` for V3/V4) — 0 errors
- [ ] **Performance & Profiling:**
  - `kernel_ms` / `e2e_ms` logged to `results/*.json`
  - Key `ncu` / `nsys` evidence summarized in `docs/` or `profiles/`
