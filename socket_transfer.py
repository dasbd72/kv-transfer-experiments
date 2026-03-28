"""IPC transfer of a KV Cache tensor directly over a socket.

Sample usage:

python ipc_transfer.py --receiver --socket-path /tmp/ipc.sock
python ipc_transfer.py --sender --socket-path /tmp/ipc.sock --tensor-size 100000000
"""

import argparse
import os
import socket
import time
import logging
from header import Header


def recvall(sock, n):
    """Helper function to receive exactly n bytes from a socket."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def run_sender(socket_path, tensor_size):
    logger = logging.getLogger("sender")

    header = Header(tensor_size=tensor_size)

    # 1. Generate the dummy "KV Cache" data in memory
    start_write_pc = time.perf_counter()
    header_bytes = header.to_bytes()
    k_data = b"K" * (tensor_size // 2)
    v_data = b"V" * (tensor_size - (tensor_size // 2))
    payload = header_bytes + k_data + v_data
    logger.info(
        "Data prepared in %.2f seconds.",
        time.perf_counter() - start_write_pc,
    )

    # 2. Connect to UDS and stream the payload
    logger.info("Connecting to socket at %s...", socket_path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        # Wait for receiver to bind the socket
        while True:
            try:
                sock.connect(socket_path)
                break
            except FileNotFoundError:
                time.sleep(0.5)

        # Send all data directly over the socket
        start_send = time.perf_counter()
        sock.sendall(payload)
        logger.info(
            "Data sent in %.2f seconds. Waiting for receiver acknowledgment...",
            time.perf_counter() - start_send,
        )

        done = sock.recv(1)
        if done == b"D":
            logger.info("Receiver acknowledged. Terminating sender...")
        else:
            logger.error("Unexpected receiver signal: %r", done)


def run_receiver(socket_path):
    logger = logging.getLogger("receiver")

    # 1. Setup the Unix Domain Socket
    if os.path.exists(socket_path):
        os.remove(socket_path)

    logger.info("Listening on %s...", socket_path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(socket_path)
        server.listen(1)

        conn, addr = server.accept()
        with conn:
            # 2. Receive the Header first
            header_bytes = recvall(conn, Header.SIZE)
            if not header_bytes:
                logger.error("Connection closed before header could be read.")
                return

            header = Header.from_bytes(header_bytes)
            data = recvall(conn, header.tensor_size)
            conn.sendall(b"D")
            if len(data) != header.tensor_size:
                logger.error("Incomplete data transfer.")
                return
            logger.info(
                "Completed reading %d bytes in %.2f seconds.",
                header.tensor_size,
                time.time() - (header.timestamp / 1000),
            )


def main():
    parser = argparse.ArgumentParser()
    role_group = parser.add_mutually_exclusive_group(required=True)
    role_group.add_argument("--sender", action="store_true")
    role_group.add_argument("--receiver", action="store_true")
    parser.add_argument("--socket-path", type=str, default="/tmp/socket.sock")
    parser.add_argument("--tensor-size", type=int, default=100 * 1024 * 1024)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    if args.sender:
        run_sender(args.socket_path, args.tensor_size)
    else:
        run_receiver(args.socket_path)


if __name__ == "__main__":
    main()
