#!/bin/bash
set -euo pipefail

echo "[bootstrap] Preparing runtime environment"

: "${CODE_REPO_URL:?CODE_REPO_URL is required}"

export GIT_TERMINAL_PROMPT=0
export GIT_CONFIG_GLOBAL=/workspace/.gitconfig

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    cat >/tmp/git-askpass.sh <<'EOF'
#!/bin/sh
case "$1" in
  *sername*)
    printf '%s\n' "${GITHUB_USERNAME:-x-access-token}"
    ;;
  *)
    printf '%s\n' "$GITHUB_TOKEN"
    ;;
esac
EOF
    chmod 700 /tmp/git-askpass.sh
    export GIT_ASKPASS=/tmp/git-askpass.sh
fi

git config --global user.name "Kubeflow Pipeline"
git config --global user.email "kubeflow@localhost"

rm -rf /workspace/code /workspace/venv

echo "[bootstrap] Cloning ${CODE_REPO_URL}"
git clone "${CODE_REPO_URL}" /workspace/code

cd /workspace/code

if [[ -n "${CODE_COMMIT:-}" ]]; then
    echo "[bootstrap] Checking out ${CODE_COMMIT}"
    git checkout --detach "${CODE_COMMIT}"
fi

echo "[bootstrap] Resolved code commit: $(git rev-parse HEAD)"

echo "[bootstrap] Creating virtual environment"
/usr/local/bin/python -m venv /workspace/venv

if [[ -f /workspace/code/requirements.txt ]]; then
    echo "[bootstrap] Installing requirements.txt"
    /workspace/venv/bin/python -m pip install \
        -r /workspace/code/requirements.txt
else
    echo "[bootstrap] ERROR: /workspace/code/requirements.txt does not exist" >&2
    exit 1
fi

echo "[bootstrap] Runtime ready"
