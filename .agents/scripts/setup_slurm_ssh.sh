#!/usr/bin/env bash
# ==============================================================================
# setup_slurm_ssh.sh — Non-Destructive One-Time SSH Jail Installer (kuda_means)
# ==============================================================================
# Creates a dedicated key (~/.ssh/id_ed25519_kuda_means) and SSH alias
# (Host kmutt-slurm-kuda), and appends the restricted kuda_means_gatekeeper.sh
# entry to ~/.ssh/authorized_keys on the KMUTT Slurm cluster WITHOUT overwriting
# existing keys (such as kv_bench's kmutt-slurm-agent).
#
# Usage:
#   ./.agents/scripts/setup_slurm_ssh.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEKEEPER_SRC="$SCRIPT_DIR/kuda_means_gatekeeper.sh"
AGENT_KEY="$HOME/.ssh/id_ed25519_kuda_means"
SSH_CONFIG="$HOME/.ssh/config"
CLUSTER_TARGET="66070503408@cpe.kmutt.ac.th@portal.slurm.cpe.kmutt.ac.th"

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

# 1. Generate dedicated SSH key for kuda_means if not present
if [[ ! -f "$AGENT_KEY" ]]; then
    echo ">> [1/3] Generating dedicated SSH key at $AGENT_KEY ..."
    ssh-keygen -t ed25519 -f "$AGENT_KEY" -N "" -C "kuda_means-agent-restricted"
else
    echo ">> [1/3] Dedicated SSH key already exists at $AGENT_KEY"
fi

# 2. Cleanly configure Host kmutt-slurm-kuda in ~/.ssh/config (preserves all other Host blocks)
echo ">> [2/3] Configuring 'Host kmutt-slurm-kuda' in $SSH_CONFIG ..."
touch "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

python3 - "$SSH_CONFIG" << 'PYEOF'
import pathlib, re, sys

cfg_path = pathlib.Path(sys.argv[1])
text = cfg_path.read_text() if cfg_path.exists() else ""

# Remove only an existing Host kmutt-slurm-kuda block (never touches kmutt-slurm-agent!)
text = re.sub(
    r"\n*Host kmutt-slurm-kuda\b.*?(?=\nHost |\Z)",
    "",
    text,
    flags=re.DOTALL,
)

block = """

Host kmutt-slurm-kuda
    HostName portal.slurm.cpe.kmutt.ac.th
    User 66070503408@cpe.kmutt.ac.th
    IdentityFile ~/.ssh/id_ed25519_kuda_means
    IdentitiesOnly yes
"""

cfg_path.write_text(text.rstrip() + block)
PYEOF

# 3. Install kuda_means_gatekeeper.sh and non-destructively update ~/.ssh/authorized_keys
echo ">> [3/3] Installing gatekeeper and updating ~/.ssh/authorized_keys on cluster..."
echo "   (Enter your KMUTT cluster password once when prompted below)"

AGENT_PUBKEY="$(cat "${AGENT_KEY}.pub")"
GATEKEEPER_B64="$(base64 -w0 "$GATEKEEPER_SRC")"
AUTH_LINE="command=\"\$HOME/bin/kuda_means_gatekeeper.sh\",no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-pty ${AGENT_PUBKEY}"
AUTH_LINE_B64="$(printf '%s' "$AUTH_LINE" | base64 -w0)"

ssh -o PubkeyAuthentication=no "$CLUSTER_TARGET" \
    "mkdir -p ~/bin ~/kuda_means/results ~/kuda_means/profiles ~/kuda_means/logs ~/.ssh && \
     chmod 700 ~/bin ~/.ssh && \
     echo '$GATEKEEPER_B64' | base64 -d > ~/bin/kuda_means_gatekeeper.sh && \
     chmod 700 ~/bin/kuda_means_gatekeeper.sh && \
     touch ~/.ssh/authorized_keys && \
     grep -v 'kuda_means_gatekeeper.sh' ~/.ssh/authorized_keys > ~/.ssh/authorized_keys.tmp || true && \
     echo '$AUTH_LINE_B64' | base64 -d >> ~/.ssh/authorized_keys.tmp && \
     echo '' >> ~/.ssh/authorized_keys.tmp && \
     mv ~/.ssh/authorized_keys.tmp ~/.ssh/authorized_keys && \
     chmod 600 ~/.ssh/authorized_keys && \
     echo '>> Cluster gatekeeper & authorized_keys updated (existing keys preserved)!'"

echo ""
echo "=================================================================="
echo "Setup Complete! Testing restricted connection (squeue --me):"
echo "=================================================================="
"$SCRIPT_DIR/slurm_ctl.sh" status
