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
python shm_transfer.py --receiver --socket-path /tmp/shared_sock/shm.sock
python shm_transfer.py --sender --socket-path /tmp/shared_sock/shm.sock --model /home/models/Qwen3-0.6B --num-requests 32 --seq-len 2048 --target-count 16

# Two containers must see the same /dev/shm (bind-mount host tmpfs).  Default
# per-container /dev/shm is isolated, so paths sent over the socket would not match.
docker run -it --rm \
  -v /tmp/shared_sock:/tmp/shared_sock \
  -v /dev/shm:/dev/shm \
  dasbd72/experiment-kv-transfer \
  bash -c "python3 shm_transfer.py --receiver --socket-path /tmp/shared_sock/shm.sock"

docker run -it --rm \
  -v /home/models:/models \
  -v /tmp/shared_sock:/tmp/shared_sock \
  -v /dev/shm:/dev/shm \
  --gpus 1 \
  dasbd72/experiment-kv-transfer \
  bash -c "python3 shm_transfer.py --sender --socket-path /tmp/shared_sock/shm.sock --model /models/Qwen3-0.6B --num-requests 32 --seq-len 2048 --target-count 16"
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

# Cuda IPC does not work in docker unless GPU of sender is visible to the receiver.
```
