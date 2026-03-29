"""Layer-wise IPC transfer of GPU-resident paged KV cache via memfd.

Sender:
  1. Allocates a real GPU paged KV pool (vLLM FlashAttention layout) and
     populates it with multiple requests.
  2. For each layer, gathers the target requests' KV on the GPU, creates
     an mmap'd memfd, pins the payload region (cudaHostRegister under the
     hood via torch pin_memory), and performs a single synchronous D2H
     copy into the pinned mmap.  No separate pinned staging buffer or
     host-to-host copy.
  3. Sends the memfd fd and waits for a lightweight ACK before proceeding
     to the next layer.

Receiver:
  1. Parses the header, sends ACK immediately.
  2. Returns zero-copy tensor views backed by the mmap'd memfd payload
     (no host-to-host clone).  The mmap objects are kept alive until
     the caller is done with the tensors.

Wire layout per layer (after v3 header):
  Contiguous tensor of shape (num_targets, seq_len, 2, num_kv_heads, head_dim)
  in the model's native dtype.
"""

import argparse
import logging
import mmap
import os
import socket
import struct
import time
import random
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from header import DTYPE_ID, ID_TO_DTYPE_NAME, Header
from kv_layout import (
    DTYPE_BYTES,
    TORCH_DTYPE,
    KVProfile,
    PagedKVPool,
    allocate_requests,
    gather_kv,
    load_kv_profile,
)

ACK_SIZE = 5  # b"A" + big-endian uint32 layer index


# -- ACK helpers --------------------------------------------------------------


def _pack_ack(layer_idx: int) -> bytes:
    return b"A" + struct.pack(">I", layer_idx)


def _unpack_ack(data: bytes) -> int:
    if len(data) != ACK_SIZE or data[0:1] != b"A":
        raise ValueError(f"Invalid ACK payload: {data!r}")
    return struct.unpack(">I", data[1:5])[0]


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly *n* bytes from *sock*."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed before all bytes received")
        buf.extend(chunk)
    return bytes(buf)


# -- sender -------------------------------------------------------------------


def run_sender(
    socket_path: str,
    profile: KVProfile,
    num_requests: int,
    seq_len: int,
    num_gpu_blocks: int,
    block_size: int,
    target_indices: list[int],
):
    logger = logging.getLogger("sender")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sender (real GPU KV cache)")

    device = torch.device("cuda")

    # 1. Allocate paged KV pool and block tables
    pool = PagedKVPool(profile, num_gpu_blocks, block_size, device)
    block_tables = allocate_requests(pool, num_requests, seq_len)

    logger.info(
        "Model %s: layers=%d kv_heads=%d head_dim=%d dtype=%s",
        profile.model_id,
        pool.num_layers,
        pool.num_kv_heads,
        pool.head_dim,
        profile.dtype_name,
    )
    logger.info(
        "GPU pool: %d blocks × block_size=%d, num_requests=%d × seq_len=%d",
        num_gpu_blocks,
        block_size,
        num_requests,
        seq_len,
    )

    num_targets = len(target_indices)
    dtype = profile.torch_dtype
    export_numel = num_targets * seq_len * 2 * pool.num_kv_heads * pool.head_dim
    export_bytes = export_numel * profile.dtype_bytes

    logger.info(
        "Exporting %d/%d requests (indices %s), per-layer payload=%.2f MB",
        num_targets,
        num_requests,
        target_indices,
        export_bytes / (1024**2),
    )

    # Pre-compute header fields that are constant across layers
    hdr_kwargs = dict(
        num_layers=pool.num_layers,
        tensor_size=export_bytes,
        block_size=block_size,
        num_kv_heads=pool.num_kv_heads,
        head_dim=pool.head_dim,
        dtype_id=DTYPE_ID[profile.dtype_name],
        seq_len=seq_len,
        num_targets=num_targets,
        target_indices=tuple(target_indices),
    )
    hdr_size = Header(layer_idx=0, **hdr_kwargs).total_header_size
    file_size = hdr_size + export_bytes

    # 2. Connect and transfer
    logger.info("Connecting to %s ...", socket_path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        while True:
            try:
                sock.connect(socket_path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.5)

        transfer_start = time.perf_counter()

        for layer_idx in range(pool.num_layers):
            t0 = time.perf_counter()
            gathered = gather_kv(
                pool.kv_caches[layer_idx],
                block_tables,
                target_indices,
                seq_len,
            )
            torch.cuda.synchronize()
            gather_elapsed = time.perf_counter() - t0

            hdr = Header(layer_idx=layer_idx, **hdr_kwargs)

            # Create memfd, size it, mmap, write header
            fd = os.memfd_create(f"kv_layer_{layer_idx}")
            os.ftruncate(fd, file_size)
            mm = mmap.mmap(fd, file_size)

            # Pin mmap region
            t1 = time.perf_counter()
            buffer = torch.frombuffer(
                mm,
                dtype=dtype,
                offset=hdr_size,
                count=export_numel,
            ).pin_memory()
            pin_elapsed = time.perf_counter() - t1

            # Then single D2H: GPU gathered → pinned mmap payload
            t2 = time.perf_counter()
            mm[0:hdr_size] = hdr.to_bytes()
            buffer.copy_(gathered.view(-1))
            copy_elapsed = time.perf_counter() - t2

            t3 = time.perf_counter()
            socket.send_fds(sock, [b"1"], [fd])
            mm.close()
            os.close(fd)
            send_elapsed = time.perf_counter() - t3

            t4 = time.perf_counter()
            ack_layer = _unpack_ack(_recv_exact(sock, ACK_SIZE))
            if ack_layer != hdr.layer_idx:
                raise RuntimeError(
                    f"ACK mismatch: expected {hdr.layer_idx}, got {ack_layer}"
                )
            ack_elapsed = time.perf_counter() - t4

            layer_elapsed = time.perf_counter() - t0
            logger.info(
                "Layer %d/%d memfd sent in %.2f ms (%.2f MB/s). Gather: %.2f ms, Copy: %.2f ms, Pin: %.2f ms, Send: %.2f ms, Ack: %.2f ms.",
                hdr.layer_idx,
                hdr.num_layers - 1,
                layer_elapsed * 1000,
                hdr.tensor_size / (1024**2) / layer_elapsed if layer_elapsed > 0 else 0,
                gather_elapsed * 1000,
                copy_elapsed * 1000,
                pin_elapsed * 1000,
                send_elapsed * 1000,
                ack_elapsed * 1000,
            )

        elapsed = time.perf_counter() - transfer_start
        total_bytes = pool.num_layers * export_bytes
        logger.info(
            "All %d layers sent in %.2f s (%.2f MB, %.2f MB/s).",
            pool.num_layers,
            elapsed,
            total_bytes / (1024**2),
            total_bytes / (1024**2) / elapsed if elapsed > 0 else 0,
        )


# -- receiver -----------------------------------------------------------------


def _process_layer(
    fd: int,
    mm: mmap.mmap,
    hdr: Header,
    logger: logging.Logger,
) -> tuple[torch.Tensor, mmap.mmap]:
    """Build a zero-copy CPU tensor view backed by the mmap payload.

    The returned mmap must stay open for the tensor's lifetime; the caller
    is responsible for closing it after the tensor is no longer needed.
    """
    t0 = time.perf_counter()

    dtype_name = ID_TO_DTYPE_NAME[hdr.dtype_id]
    dtype = TORCH_DTYPE[dtype_name]
    num_elements = hdr.tensor_size // DTYPE_BYTES[dtype_name]

    tensor = torch.frombuffer(
        mm,
        dtype=dtype,
        offset=hdr.total_header_size,
        count=num_elements,
    ).reshape(hdr.num_targets, hdr.seq_len, 2, hdr.num_kv_heads, hdr.head_dim)

    os.close(fd)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Layer %d background done in %.1f ms → %s %s",
        hdr.layer_idx,
        elapsed_ms,
        tensor.shape,
        tensor.dtype,
    )
    return tensor, mm


def run_receiver(socket_path: str):
    logger = logging.getLogger("receiver")

    if os.path.exists(socket_path):
        os.remove(socket_path)

    logger.info("Listening on %s ...", socket_path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(socket_path)
        server.listen(1)

        conn, _ = server.accept()
        with conn:
            executor = ThreadPoolExecutor(max_workers=4)
            futures: list[tuple[int, Future]] = []
            transfer_start = time.perf_counter()
            num_layers: int | None = None
            total_bytes = 0

            while True:
                _, fds, _, _ = socket.recv_fds(conn, 1, 1)
                fd = fds[0]
                file_size = os.fstat(fd).st_size

                mm = mmap.mmap(fd, file_size)
                hdr = Header.read_from(mm)

                if num_layers is None:
                    num_layers = hdr.num_layers

                total_bytes += hdr.tensor_size

                # ACK immediately — sender can proceed
                conn.sendall(_pack_ack(hdr.layer_idx))

                # Enqueue background tensor construction
                fut = executor.submit(_process_layer, fd, mm, hdr, logger)
                futures.append((hdr.layer_idx, fut))

                if hdr.layer_idx == num_layers - 1:
                    break

            recv_elapsed = time.perf_counter() - transfer_start
            logger.info("All %d layers ACKed in %.2f s.", num_layers, recv_elapsed)

            results: dict[int, torch.Tensor] = {}
            keepalives: list[mmap.mmap] = []
            for layer_idx, fut in futures:
                tensor, mm = fut.result()
                results[layer_idx] = tensor
                keepalives.append(mm)
            executor.shutdown()
            futures.clear()

            total_elapsed = time.perf_counter() - transfer_start
            logger.info(
                "Background processing complete. Received %d layer tensors. "
                "Total wall time: %.2f s (%.2f MB/s).",
                len(results),
                total_elapsed,
                total_bytes / (1024**2) / total_elapsed if total_elapsed > 0 else 0,
            )

            # Release tensor views, then close mmap backings
            results.clear()
            for mm in keepalives:
                mm.close()


# -- CLI -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise paged KV cache transfer via memfd over Unix domain sockets.",
    )
    role_group = parser.add_mutually_exclusive_group(required=True)
    role_group.add_argument("--sender", action="store_true")
    role_group.add_argument("--receiver", action="store_true")
    parser.add_argument("--socket-path", default="/tmp/memfd.sock")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace model id (sender only).",
    )
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--num-gpu-blocks",
        type=int,
        default=0,
        help="Total GPU KV blocks (0 = auto-size for num_requests × seq_len).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Tokens per KV block (must be a multiple of 16).",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=0,
        help="Export target_count requests (0 = all).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    if args.sender:
        profile = load_kv_profile(args.model)

        blocks_per_req = (args.seq_len + args.block_size - 1) // args.block_size
        num_gpu_blocks = args.num_gpu_blocks or args.num_requests * blocks_per_req
        target_indices = random.sample(range(args.num_requests), args.target_count)

        run_sender(
            args.socket_path,
            profile,
            args.num_requests,
            args.seq_len,
            num_gpu_blocks,
            args.block_size,
            target_indices,
        )
    else:
        run_receiver(args.socket_path)


if __name__ == "__main__":
    main()
