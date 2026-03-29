# KV Transfer Experimental Code

## Quick commands


```bash
python bandwidth_bench.py
```

```bash
docker build . --tag dasbd72/experiment-kv-transfer
```

```bash
mkdir -p /tmp/shared_sock
chmod 777 /tmp/shared_sock
```

```bash
python memfd_transfer.py --receiver --socket-path /tmp/shared_sock/memfd.sock
python memfd_transfer.py --sender --socket-path /tmp/shared_sock/memfd.sock --model /home/models/Qwen3-0.6B --num-requests 32 --seq-len 2048 --target-count 16

docker run -it --rm \
  -v /tmp/shared_sock:/tmp/shared_sock \
  dasbd72/experiment-kv-transfer \
  bash -c "python3 memfd_transfer.py --receiver --socket-path /tmp/shared_sock/memfd.sock"

docker run -it --rm \
  -v /home/models:/models \
  -v /tmp/shared_sock:/tmp/shared_sock \
  --gpus 1 \
  dasbd72/experiment-kv-transfer \
  bash -c "python3 memfd_transfer.py --sender --socket-path /tmp/shared_sock/memfd.sock --model /models/Qwen3-0.6B --num-requests 32 --seq-len 2048 --target-count 16"
```

```bash
python socket_transfer.py --receiver --socket-path /tmp/shared_sock/socket.sock
python socket_transfer.py --sender --socket-path /tmp/shared_sock/socket.sock --model /home/models/Qwen3-0.6B --num-requests 32 --seq-len 2048 --target-count 16

docker run -it --rm \
  -v /tmp/shared_sock:/tmp/shared_sock \
  dasbd72/experiment-kv-transfer \
  bash -c "python3 socket_transfer.py --receiver --socket-path /tmp/shared_sock/socket.sock"

docker run -it --rm \
  -v /home/models:/models \
  -v /tmp/shared_sock:/tmp/shared_sock \
  --gpus 1 \
  dasbd72/experiment-kv-transfer \
  bash -c "python3 socket_transfer.py --sender --socket-path /tmp/shared_sock/socket.sock --model /models/Qwen3-0.6B --num-requests 32 --seq-len 2048 --target-count 16"
```

```bash
python cuda_ipc_transfer.py --receiver --socket-path /tmp/shared_sock/cuda_ipc.sock
python cuda_ipc_transfer.py --sender --socket-path /tmp/shared_sock/cuda_ipc.sock --model /home/models/Qwen3-0.6B --num-requests 32 --seq-len 2048 --target-count 16

# CUDA IPC requires both containers to share the host IPC namespace (--ipc=host)
# and both must have GPU access.  The receiver gets GPU tensors directly — no D2H/H2D copy.
docker run -it --rm \
  -v /tmp/shared_sock:/tmp/shared_sock \
  --gpus 1 \
  --ipc=host \
  dasbd72/experiment-kv-transfer \
  bash -c "python3 cuda_ipc_transfer.py --receiver --socket-path /tmp/shared_sock/cuda_ipc.sock"

docker run -it --rm \
  -v /home/models:/models \
  -v /tmp/shared_sock:/tmp/shared_sock \
  --gpus 1 \
  --ipc=host \
  dasbd72/experiment-kv-transfer \
  bash -c "python3 cuda_ipc_transfer.py --sender --socket-path /tmp/shared_sock/cuda_ipc.sock --model /models/Qwen3-0.6B --num-requests 32 --seq-len 2048 --target-count 16"
```
