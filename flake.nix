{
  description = "kuda_means — High-Performance GPU K-Means Clustering Dev Environment";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};

      slurmUser = "66070503408@cpe.kmutt.ac.th";
      slurmHost = "portal.slurm.cpe.kmutt.ac.th";

      libPath = pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
      ];

      slurm-ssh = pkgs.writeShellScriptBin "slurm-ssh" ''
        exec ssh -l ${slurmUser} ${slurmHost} "$@"
      '';

      slurm-scp = pkgs.writeShellScriptBin "slurm-scp" ''
        REMOTE="${slurmUser}@${slurmHost}"

        opts=()
        args=()

        while [ $# -gt 0 ]; do
          case "$1" in
            -P|-i|-o|-F|-l)
              opts+=("$1" "$2")
              shift 2
              ;;
            -*)
              opts+=("$1")
              shift
              ;;
            *)
              args+=("$1")
              shift
              ;;
          esac
        done

        if [ ''${#args[@]} -lt 2 ]; then
          echo "Error: Both source and destination must be specified." >&2
          echo "Usage: slurm-scp [options] <source>... <destination>" >&2
          exit 1
        fi

        has_remote=false
        for arg in "''${args[@]}"; do
          if [[ "$arg" == :* || "$arg" == remote:* || "$arg" == slurm:* || "$arg" == "$REMOTE:"* ]]; then
            has_remote=true
            break
          fi
        done

        final_args=()
        last_idx=$(( ''${#args[@]} - 1 ))

        for i in "''${!args[@]}"; do
          a="''${args[$i]}"
          if [[ "$a" == :* ]]; then
            final_args+=("$REMOTE:''${a#:}")
          elif [[ "$a" == remote:* ]]; then
            final_args+=("$REMOTE:''${a#remote:}")
          elif [[ "$a" == slurm:* ]]; then
            final_args+=("$REMOTE:''${a#slurm:}")
          elif [ "$has_remote" = false ] && [ "$i" -eq "$last_idx" ]; then
            final_args+=("$REMOTE:$a")
          else
            final_args+=("$a")
          fi
        done

        exec scp -r "''${opts[@]}" "''${final_args[@]}"
      '';

      slurm-ctl = pkgs.writeShellScriptBin "slurm-ctl" ''
        if [ -x "./.agents/scripts/slurm_ctl.sh" ]; then
          exec ./.agents/scripts/slurm_ctl.sh "$@"
        else
          echo "Error: ./.agents/scripts/slurm_ctl.sh not found in current directory." >&2
          exit 1
        fi
      '';

      colab-start = pkgs.writeShellScriptBin "colab-start" ''
        export PATH="$HOME/.local/bin:$PATH"
        export LD_LIBRARY_PATH="${libPath}:''${LD_LIBRARY_PATH:-}"
        GPU="''${1:-T4}"
        SESSION="''${2:-kuda-means}"
        echo "==> Starting Colab session '$SESSION' (GPU: $GPU)..."
        colab new -s "$SESSION" --gpu "$GPU"
      '';

      colab-stop = pkgs.writeShellScriptBin "colab-stop" ''
        export PATH="$HOME/.local/bin:$PATH"
        export LD_LIBRARY_PATH="${libPath}:''${LD_LIBRARY_PATH:-}"
        SESSION="''${1:-kuda-means}"
        echo "==> Stopping Colab session '$SESSION'..."
        colab stop -s "$SESSION"
      '';

      colab-status = pkgs.writeShellScriptBin "colab-status" ''
        export PATH="$HOME/.local/bin:$PATH"
        export LD_LIBRARY_PATH="${libPath}:''${LD_LIBRARY_PATH:-}"
        echo "==> Active Colab sessions:"
        colab sessions
      '';

      colab-console = pkgs.writeShellScriptBin "colab-console" ''
        export PATH="$HOME/.local/bin:$PATH"
        export LD_LIBRARY_PATH="${libPath}:''${LD_LIBRARY_PATH:-}"
        exec colab console "$@"
      '';

      colab-push = pkgs.writeShellScriptBin "colab-push" ''
        export PATH="$HOME/.local/bin:$PATH"
        export LD_LIBRARY_PATH="${libPath}:''${LD_LIBRARY_PATH:-}"
        if [ $# -lt 1 ]; then
          echo "Usage: colab-push <local_file> [remote_path]"
          exit 1
        fi
        SRC="$1"
        if [ ! -f "$SRC" ]; then
          echo "Error: Local file '$SRC' not found!" >&2
          exit 1
        fi
        FILENAME="$(basename "$SRC")"
        DEST="''${2:-/content/$FILENAME}"
        echo "==> Uploading $SRC -> Colab:$DEST..."
        exec colab upload "$SRC" "$DEST"
      '';

      colab-launcher-gen = pkgs.writeText "colab_launcher_gen.py" ''
        import sys, textwrap

        cu_file    = sys.argv[1]
        py_out     = sys.argv[2]
        remote_src = sys.argv[3]
        remote_bin = sys.argv[4]
        arch       = sys.argv[5]
        extra_args = sys.argv[6:] if len(sys.argv) > 6 else []

        with open(cu_file, "r") as f:
            cu_source = f.read()

        launcher = textwrap.dedent(f"""\
            import subprocess, sys, pathlib

            src  = {remote_src!r}
            bin_ = {remote_bin!r}
            arch = {arch!r}
            args = {extra_args!r}

            pathlib.Path(src).write_text({cu_source!r})

            print(f"[colab-submit] Compiling: nvcc -O3 {{src}} -o {{bin_}} -arch={{arch}}")
            r = subprocess.run(
                ["nvcc", "-O3", src, "-o", bin_, f"-arch={{arch}}"],
                capture_output=True, text=True
            )
            sys.stdout.write(r.stdout)
            if r.returncode != 0:
                sys.stderr.write(r.stderr)
                sys.exit(r.returncode)

            cmd = [bin_] + args
            print("[colab-submit] Running: " + " ".join(cmd))
            sys.stdout.flush()
            r = subprocess.run(cmd, capture_output=True, text=True)
            sys.stdout.write(r.stdout)
            sys.stdout.flush()
            if r.returncode != 0:
                sys.stderr.write(r.stderr)
                sys.stderr.flush()
                sys.exit(r.returncode)
        """)

        with open(py_out, "w") as f:
            f.write(launcher)
      '';

      colab-submit = pkgs.writeShellScriptBin "colab-submit" ''
        export PATH="$HOME/.local/bin:$PATH"
        export LD_LIBRARY_PATH="${libPath}:''${LD_LIBRARY_PATH:-}"

        ARCH="sm_75"
        SESSION="kuda-means"
        TIMEOUT="3600"
        FILE=""
        EXTRA_ARGS=()

        while [ $# -gt 0 ]; do
          case "$1" in
            -s|--session) SESSION="$2"; shift 2 ;;
            -a|--arch)    ARCH="$2"; shift 2 ;;
            -t|--timeout) TIMEOUT="$2"; shift 2 ;;
            --)           shift; EXTRA_ARGS+=("$@"); break ;;
            *)
              if [ -z "$FILE" ]; then
                FILE="$1"; shift
              else
                EXTRA_ARGS+=("$1"); shift
              fi
              ;;
          esac
        done

        if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
          echo "Usage: colab-submit <file.cu> [-a arch] [-s session] [-t timeout] [-- <args...>]" >&2
          exit 1
        fi

        FILENAME="$(basename "$FILE")"
        BINNAME="''${FILENAME%.*}"
        REMOTE_SRC="/content/$FILENAME"
        REMOTE_BIN="/content/$BINNAME"

        GEN_SCRIPT="${colab-launcher-gen}"
        LAUNCHER="$(mktemp /tmp/colab_launch_XXXXXX.py)"
        python3 "$GEN_SCRIPT" "$FILE" "$LAUNCHER" "$REMOTE_SRC" "$REMOTE_BIN" "$ARCH" "''${EXTRA_ARGS[@]}"

        echo "==> colab-submit: $FILE (arch=$ARCH) -> session '$SESSION'"
        colab exec -s "$SESSION" -f "$LAUNCHER" --timeout "$TIMEOUT"
        STATUS=$?
        rm -f "$LAUNCHER"
        exit $STATUS
      '';

      colab-run-project = pkgs.writeShellScriptBin "colab-run-project" ''
        export PATH="$HOME/.local/bin:$PATH"
        export LD_LIBRARY_PATH="${libPath}:''${LD_LIBRARY_PATH:-}"
        if [ -x "./scripts/colab/colab_build_and_run.sh" ]; then
          exec ./scripts/colab/colab_build_and_run.sh "$@"
        else
          echo "Error: ./scripts/colab/colab_build_and_run.sh not found." >&2
          exit 1
        fi
      '';
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          gcc
          gnumake
          cmake
          rsync
          uv
          python3
          ruff
          cppcheck
          clang-tools
          bash-language-server
          shellcheck
          shfmt
          slurm-ssh
          slurm-scp
          slurm-ctl
          colab-start
          colab-stop
          colab-status
          colab-console
          colab-push
          colab-submit
          colab-run-project
        ];

        shellHook = ''
          export PATH="$HOME/.local/bin:$PATH"
          export LD_LIBRARY_PATH="${libPath}:''${LD_LIBRARY_PATH:-}"

          echo "kuda_means Dev Environment loaded (CPE 393 GPU K-Means)"
          echo "Tools: gcc, make, cmake, clangd, uv, python3, rsync, shellcheck, shfmt"
          echo ""
          echo "Cluster (KMUTT Slurm RTX 4090, sm_89):"
          echo "  slurm-ctl sync|submit|profile|status|logs|fetch|cancel"
          echo "  slurm-ssh / slurm-scp"
          echo ""
          echo "Fallback (Google Colab T4, sm_75):"
          echo "  colab-start [T4] [kuda-means]   colab-status   colab-stop [kuda-means]"
          echo "  colab-run-project [-- <args>]   colab-submit <file.cu> [-- <args>]"
          echo ""

          if ! command -v colab >/dev/null 2>&1; then
            echo "==> 'colab' CLI not found. Installing via uv tool..."
            uv tool install google-colab-cli || echo "Failed to auto-install google-colab-cli."
          fi
        '';
      };
    };
}
