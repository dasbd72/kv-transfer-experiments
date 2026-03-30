#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-/home/models/Qwen3-0.6B}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
SEQ_LEN="${SEQ_LEN:-2048}"
TARGET_COUNT="${TARGET_COUNT:-16}"
SOCK_DIR="${SOCK_DIR:-/tmp/shared_sock}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [TEST]

Run KV transfer benchmarks (receiver + sender per test).

  TEST    One of: all (default), shm, memfd, socket, cuda_ipc

Environment (optional):
  MODEL         Model path (default: $MODEL)
  NUM_REQUESTS  (default: $NUM_REQUESTS)
  SEQ_LEN       (default: $SEQ_LEN)
  TARGET_COUNT  (default: $TARGET_COUNT)
  SOCK_DIR      Unix socket directory (default: $SOCK_DIR)
EOF
}

# PIDs of the current receiver/sender pair; cleared when a pair finishes so EXIT/INT/TERM only reap active children.
PAIR_RECV_PID=
PAIR_SEND_PID=

cleanup_pair_children() {
  if [[ -n "${PAIR_RECV_PID:-}" ]] && kill -0 "$PAIR_RECV_PID" 2>/dev/null; then
    kill -TERM "$PAIR_RECV_PID" 2>/dev/null || true
  fi
  if [[ -n "${PAIR_SEND_PID:-}" ]] && kill -0 "$PAIR_SEND_PID" 2>/dev/null; then
    kill -TERM "$PAIR_SEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${PAIR_RECV_PID:-}" ]]; then
    wait "$PAIR_RECV_PID" 2>/dev/null || true
  fi
  if [[ -n "${PAIR_SEND_PID:-}" ]]; then
    wait "$PAIR_SEND_PID" 2>/dev/null || true
  fi
  PAIR_RECV_PID=
  PAIR_SEND_PID=
}

run_pair() {
  local name=$1
  local script=$2
  local sock=$3
  local log="logs/${name}_results.txt"
  local recv_pid send_pid status=0

  : >"$log"
  python "$script" --receiver --socket-path "$sock" >>"$log" 2>&1 &
  recv_pid=$!
  python "$script" --sender --socket-path "$sock" --model "$MODEL" \
    --num-requests "$NUM_REQUESTS" --seq-len "$SEQ_LEN" --target-count "$TARGET_COUNT" \
    >>"$log" 2>&1 &
  send_pid=$!

  PAIR_RECV_PID=$recv_pid
  PAIR_SEND_PID=$send_pid

  if ! wait "$recv_pid"; then
    status=1
    kill -TERM "$send_pid" 2>/dev/null || true
  fi
  PAIR_RECV_PID=

  if ! wait "$send_pid"; then
    status=1
  fi
  PAIR_SEND_PID=

  return "$status"
}

run_shm()    { run_pair shm    shm_transfer.py    "$SOCK_DIR/shm.sock"; }
run_memfd()  { run_pair memfd  memfd_transfer.py  "$SOCK_DIR/memfd.sock"; }
run_socket() { run_pair socket socket_transfer.py "$SOCK_DIR/socket.sock"; }
run_cuda()   { run_pair cuda_ipc cuda_ipc_transfer.py "$SOCK_DIR/cuda_ipc.sock"; }

TEST="${1:-all}"
case "$TEST" in
  -h|--help)
    usage
    exit 0
    ;;
  all|shm|memfd|socket|cuda_ipc)
    ;;
  *)
    echo "Error: unknown test '$TEST'" >&2
    usage >&2
    exit 1
    ;;
esac

trap cleanup_pair_children EXIT
trap 'cleanup_pair_children; exit 130' INT
trap 'cleanup_pair_children; exit 143' TERM

mkdir -p logs "$SOCK_DIR"

failed=0
case "$TEST" in
  all)
    run_shm    || failed=1
    run_memfd  || failed=1
    run_socket || failed=1
    run_cuda   || failed=1
    ;;
  shm)    run_shm    || failed=1 ;;
  memfd)  run_memfd  || failed=1 ;;
  socket) run_socket || failed=1 ;;
  cuda_ipc) run_cuda || failed=1 ;;
esac

if (( failed )); then
  echo "Error: one or more transfer tests failed (see logs/*_results.txt)." >&2
  exit 1
fi
