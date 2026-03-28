"""IPC transfer of a KV Cache tensor between a sender and a receiver."""

import argparse
import os
import mmap
import socket
import time
import logging
from header import Header


def run_sender(socket_path, tensor_size):
    logger = logging.getLogger("sender")

    header = Header(tensor_size=tensor_size)
    file_size = header.SIZE + tensor_size

    # 1. Create anonymous file in RAM
    fd = os.memfd_create("kv_cache_tensor")
    os.ftruncate(fd, file_size)

    # 2. Write dummy "KV Cache" data into memory
    start_write_pc = time.perf_counter()
    with mmap.mmap(fd, file_size) as mm:
        header_bytes = header.to_bytes()
        k_data = b"K" * (tensor_size // 2)
        v_data = b"V" * (tensor_size - (tensor_size // 2))
        payload = header_bytes + k_data + v_data
        mm.write(payload)
        logger.info(
            "Data prepared in %.2f seconds.",
            time.perf_counter() - start_write_pc,
        )

    # 3. Connect to UDS and send the File Descriptor
    logger.info("Connecting to socket at %s...", socket_path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        # Wait for receiver to bind the socket
        while True:
            try:
                sock.connect(socket_path)
                break
            except FileNotFoundError:
                time.sleep(0.5)

        # Send the FD. A dummy byte [b'1'] is required by the protocol.
        socket.send_fds(sock, [b"1"], [fd])
        logger.info("File descriptor sent. Waiting for receiver acknowledgment...")

        done = sock.recv(1)
        if done == b"D":
            logger.info("Receiver acknowledged. Terminating sender...")
        else:
            logger.error("Unexpected receiver signal: %r", done)

    os.close(fd)


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
            # 2. Receive the File Descriptor
            msg, fds, flags, addr = socket.recv_fds(conn, 1, 1)
            received_fd = fds[0]
            file_size = os.fstat(received_fd).st_size
            logger.info("Received file descriptor: %d", received_fd)
            logger.info("File size (bytes): %d", file_size)

            # 3. Zero-copy read from the shared memory
            with mmap.mmap(received_fd, file_size) as mm:
                conn.sendall(b"D")
                header = Header.from_bytes(mm.read(Header.SIZE))
                data = mm.read(header.tensor_size)
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
    parser.add_argument("--socket-path", type=str, default="/tmp/memfd.sock")
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
