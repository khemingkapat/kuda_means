# Execution Backends & Hardware Workflow (`kuda_means`)

This is an **experimental GPU optimization project**: we are flexible on which GPU hardware we run on, **provided** every benchmark result logs full hardware provenance so results are reproducible, explainable, and never mixed across GPUs in a single comparison.

---

## 1. Decision Flow (Cluster First, Manual Colab Fallback)

Before running a GPU job:
1. **Check KMUTT Slurm availability first:**
   ```bash
   ./.agents/scripts/slurm_ctl.sh nodes
   ```
   This runs `sinfo` and `squeue` to show which GPU partitions/nodes are currently `idle` or `mix`.
2. **If a Slurm GPU node is available:**
   - Sync code and submit to the available partition (overriding `--partition` / `--gres` if needed).
   - The `.sbatch` scripts automatically detect the allocated GPU's compute capability (`nvidia-smi --query-gpu=compute_cap`) and build with the matching `ARCH=sm_XX`.
3. **If NO Slurm GPU nodes are available (STRICT MANUAL FALLBACK):**
   - **DO NOT** automatically launch Google Colab or start a Colab session on your own.
   - **Stop and report** the `sinfo` / `squeue` status to the user.
   - Wait for the user to decide whether to queue on Slurm or manually switch to Google Colab.

---

## 2. Backend 1 (Primary): KMUTT CPE Slurm Cluster

### Automated Control Wrapper (STRICT)
Always use `./.agents/scripts/slurm_ctl.sh` (or `slurm-ctl` in `nix develop`):

```bash
# 1. Check available GPU partitions/nodes (sinfo + squeue)
./.agents/scripts/slurm_ctl.sh nodes

# 2. Stage source, headers, Makefile, and scripts/slurm/ to ~/kuda_means/
./.agents/scripts/slurm_ctl.sh sync

# 3. Submit any scripts/slurm/*.sbatch (auto-detects GPU ARCH=sm_XX)
./.agents/scripts/slurm_ctl.sh submit [--script <name.sbatch>] [--partition <p>] [--gres <g>] [-- <args...>]

# 4. Submit Nsight Compute / Nsight Systems / compute-sanitizer profiling job
./.agents/scripts/slurm_ctl.sh profile [--partition <p>] [version] [N] [D] [K]

# 5. Monitor queue & inspect logs
./.agents/scripts/slurm_ctl.sh status [job_id | --all | -p <part>]
./.agents/scripts/slurm_ctl.sh logs <job_id> [lines]

# 6. Fetch results/, profiles/*.txt, and logs/ back to local workspace
./.agents/scripts/slurm_ctl.sh fetch [job_id]

# 7. Cancel a specific numeric job ID
./.agents/scripts/slurm_ctl.sh cancel <job_id>
```

### Security & Login Node Guardrails
1. **One-Time Setup:** The user runs `./.agents/scripts/setup_slurm_ssh.sh` once manually to install `~/.ssh/id_ed25519_kuda_means`, `Host kmutt-slurm-kuda`, and `~/bin/kuda_means_gatekeeper.sh`. Agents must **never** modify `~/.ssh/*` or touch `Host kmutt-slurm-agent` (reserved for `kv_bench`).
2. **Zero Compute on Login Node (`ctl01`):** Never compile (`nvcc`) or run binaries on `ctl01`. All builds, benchmarks, and profiling run inside `scripts/slurm/*.sbatch` on compute nodes (`module load cuda/12.2`, `NVCC_CCBIN=/usr/bin/g++-12`).
3. **Safe Polling:** Never run tight `while true` loops over SSH. Poll `slurm_ctl.sh status <job_id>` at 15–30s intervals or with a scheduled timer.
4. **Data Boundaries:** Never pass `--delete` to `rsync`. Only sync source/build files outbound and fetch `results/`, `profiles/*.txt`, and `logs/slurm-*.log` inbound.

---

## 3. Backend 2 (Manual Fallback): Google Colab (`Tesla T4`, `sm_75`)

Only use **after the user approves switching to Colab** (e.g., when `sinfo` shows all cluster GPUs are occupied):

```bash
# 1. Start GPU runtime
colab-start T4 kuda-means

# 2. Build & run the full multi-file project on Colab (packages Makefile, include/, src/, tests/)
colab-run-project -- --version v1 -n 100000 -d 32 -k 16

# Or submit a standalone single .cu file
colab-submit src/v1_naive/test_single.cu -a sm_75 -- --runs 5

# 3. Always release GPU compute units when done
colab-stop kuda-means
```

---

## 4. Hardware Provenance & Explainability Rule (Mandatory)

Because experiments may run across different GPUs (e.g., RTX 4090, another Slurm GPU partition, or Colab T4):
- Every benchmark CSV/JSON record **must** include: `gpu_name`, `compute_cap`, `sm_count`, `global_mem_mb`, `cuda_version`, `git_commit`, `kernel_ms`, `e2e_ms`, `iterations`, `sse`, and `verified`.
- **Never mix timings from different GPUs** in the same speedup curve or table; group comparisons strictly by `gpu_name`.
- Always pair performance claims with `ncu`/`nsys` profiler evidence so every speedup or regression can be explained in the IEEE report and presentation Q&A.
