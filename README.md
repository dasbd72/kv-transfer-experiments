# KV Transfer Experimental Code

## Quick commands

```bash
docker build . --tag dasbd72/experiment-kv-transfer
```

```bash
mkdir -p /tmp/shared_sock
chmod 777 /tmp/shared_sock

python memfd_transfer.py --receiver --socket-path /tmp/shared_sock/memfd.sock
python memfd_transfer.py --sender --socket-path /tmp/shared_sock/memfd.sock --tensor-size 1000000000

docker run -it --rm \
  -v /tmp/shared_sock:/tmp/shared_sock \
  dasbd72/experiment-kv-transfer \
  bash -c "python3 memfd_transfer.py --receiver --socket-path /tmp/shared_sock/memfd.sock"

docker run -it --rm \
  -v /tmp/shared_sock:/tmp/shared_sock \
  dasbd72/experiment-kv-transfer \
  bash -c "python3 memfd_transfer.py --sender --socket-path /tmp/shared_sock/memfd.sock --tensor-size 1000000000"


python socket_transfer.py --receiver --socket-path /tmp/shared_sock/socket.sock
python socket_transfer.py --sender --socket-path /tmp/shared_sock/socket.sock --tensor-size 1000000000

docker run -it --rm \
  -v /tmp/shared_sock:/tmp/shared_sock \
  dasbd72/experiment-kv-transfer \
  bash -c "python3 socket_transfer.py --receiver --socket-path /tmp/shared_sock/socket.sock"

docker run -it --rm \
  -v /tmp/shared_sock:/tmp/shared_sock \
  dasbd72/experiment-kv-transfer \
  bash -c "python3 socket_transfer.py --sender --socket-path /tmp/shared_sock/socket.sock --tensor-size 1000000000"
```