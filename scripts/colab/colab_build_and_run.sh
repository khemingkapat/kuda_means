#!/usr/bin/env bash
# ==============================================================================
# colab_build_and_run.sh — Multi-file Build & Run on Google Colab GPU (T4 sm_75)
# ==============================================================================
# Packages Makefile, include/, src/, and tests/, uploads to an active Colab
# session, compiles with nvcc -arch=sm_75, and executes bin/kuda_means_gpu.
#
# Usage:
#   ./scripts/colab/colab_build_and_run.sh [-a sm_75] [-s kuda-means] [-- <args...>]
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_LOCK="${REPO_ROOT}/.agents/scripts/gpu_lock.sh"
if [[ "${KUDA_MEANS_LOCK_HELD_COLAB:-0}" != "1" && -x "$GPU_LOCK" ]]; then
    SELF_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    exec "$GPU_LOCK" run colab --note "colab_build_and_run" -- "$SELF_PATH" "$@"
fi
ARCH="sm_75"
SESSION="kuda-means"
TIMEOUT="3600"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--arch)    ARCH="$2"; shift 2 ;;
        -s|--session) SESSION="$2"; shift 2 ;;
        -t|--timeout) TIMEOUT="$2"; shift 2 ;;
        --)           shift; EXTRA_ARGS+=("$@"); break ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if ! command -v colab >/dev/null 2>&1; then
    echo "ERROR: 'colab' CLI not found. Enter 'nix develop' or install google-colab-cli." >&2
    exit 1
fi

TARBALL="$(mktemp /tmp/kuda_means_src_XXXXXX.tar.gz)"
LAUNCHER="$(mktemp /tmp/kuda_means_colab_XXXXXX.py)"
trap 'rm -f "$TARBALL" "$LAUNCHER"' EXIT

echo "==> Packaging Makefile, include/, src/, tests/..."
tar -czf "$TARBALL" -C "$REPO_ROOT" \
    --exclude='*.o' --exclude='bin' --exclude='build' \
    Makefile include src tests

python3 - "$TARBALL" "$LAUNCHER" "$ARCH" "${EXTRA_ARGS[@]}" << 'PYEOF'
import base64, pathlib, sys, textwrap

tar_path   = pathlib.Path(sys.argv[1])
out_path   = pathlib.Path(sys.argv[2])
arch       = sys.argv[3]
extra_args = sys.argv[4:]

b64_data = base64.b64encode(tar_path.read_bytes()).decode("ascii")

script = textwrap.dedent(f"""\
    import base64, io, os, subprocess, sys, tarfile, pathlib

    work_dir = pathlib.Path("/content/kuda_means")
    work_dir.mkdir(parents=True, exist_ok=True)

    raw = base64.b64decode({b64_data!r})
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(work_dir)

    os.chdir(work_dir)
    print("[colab] GPU info:")
    subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv"], check=False)

    print(f"[colab] Building project (ARCH={arch})...")
    r = subprocess.run(["make", "clean"], capture_output=True, text=True)
    r = subprocess.run(["make", "cpu"], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(r.returncode)

    r = subprocess.run(["make", "gpu", f"ARCH={arch}"], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(r.returncode)

    gpu_bin = work_dir / "bin" / "kuda_means_gpu"
    cpu_bin = work_dir / "bin" / "kuda_means_cpu"
    target = gpu_bin if gpu_bin.exists() else cpu_bin

    if not target.exists():
        print("[colab] No binary built yet.")
        sys.exit(0)

    cmd = [str(target)] + {extra_args!r}
    print("[colab] Running: " + " ".join(cmd))
    sys.stdout.flush()
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(r.returncode)
""")

out_path.write_text(script)
PYEOF

echo "==> Submitting multi-file build & run to Colab session '$SESSION' (arch=$ARCH)..."
colab exec -s "$SESSION" -f "$LAUNCHER" --timeout "$TIMEOUT"
