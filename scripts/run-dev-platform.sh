#!/usr/bin/env bash
set -euo pipefail

echo "[run-dev-platform] Starting core Q2 services with local fallback configs"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

start_service() {
  local name=$1
  local cmd=$2
  local log_file="$ROOT_DIR/${name}.dev.log"
  if pgrep -f "$cmd" >/dev/null 2>&1; then
    echo "[WARN] $name appears to be already running (skipping)."
    return 0
  fi
  echo "[INFO] Starting $name ..."
  nohup bash -c "$cmd" >"$log_file" 2>&1 &
  local pid=$!
  echo $pid > "$ROOT_DIR/${name}.pid"
  echo "[INFO] $name (pid=$pid) logging to $log_file"
}

start_service quantumpulse "cd QuantumPulse && QUANTUMPULSE_SKIP_VAULT=1 python -m app.main"
sleep 2
start_service managerq "cd managerQ && MANAGERQ_SKIP_VAULT=1 python -m managerQ.app.main"

echo "[run-dev-platform] Services launched. Use 'make stop-platform' to stop or kill PIDs in *.pid files."
echo "Health endpoints:"
echo "  QuantumPulse: http://localhost:8010/health"
echo "  ManagerQ:    http://localhost:8001/health"
