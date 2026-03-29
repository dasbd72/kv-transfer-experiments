"""Layer-wise IPC transfer of GPU-resident paged KV cache via CUDA IPC.

Sender:
  1. Allocates a real GPU paged KV pool (vLLM FlashAttention layout) and
     populates it with multiple requests.
  2. For each layer, gathers the target requests' KV on the GPU, serializes
     the CUDA IPC memory handle and an interprocess event handle into a
     small metadata blob, and sends header + blob over the Unix socket.
     No D2H copy — the bulk tensor data stays on the GPU.
  3. Waits for a lightweight ACK before proceeding to the next layer.
     Keeps all exported GPU tensors alive until the transfer completes
     so the caching allocator cannot recycle their underlying CUDA blocks.

Receiver:
  1. Reads header + IPC metadata blob from the socket, sends ACK.
  2. Reconstructs a GPU tensor in its own address space via
     cudaIpcOpenMemHandle (PyTorch _new_shared_cuda), waits on the
     sender's interprocess CUDA event for data visibility.
  3. The received tensors are on GPU — no H2D copy needed.

Requirements:
  - Linux only (CUDA IPC limitation).
  - Same physical machine.  On K8s both pods need hostIPC: true and must
    NOT shadow /dev/shm with an emptyDir volume (the host /dev/shm is
    used by the CUDA driver for IPC ref-counting shared memory).
  - Both processes must see the same GPU(s) via CUDA_VISIBLE_DEVICES.

Wire layout per layer:
  Header.to_bytes()   (60 + num_targets × 4 bytes)
  uint32 big-endian   length of IPC metadata blob
  IPC metadata blob   (pickled dict, KB-scale — not the tensor payload)
"""

import argparse
import logging
import os
import pickle
import socket
import struct
import time
import random

import torch

from header import DTYPE_ID, Header
from kv_layout import (
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
    """Receive exactly *n* bytes from *sock* (small reads; prefer recv_into for large)."""
    buf = bytearray(n)
    _recv_exact_into(sock, buf)
    return bytes(buf)


def _recv_exact_into(sock: socket.socket, buf: bytearray) -> None:
    """Fill *buf* exactly from *sock* using recv_into (no incremental realloc)."""
    mv = memoryview(buf)
    pos = 0
    n = len(buf)
    while pos < n:
        got = sock.recv_into(mv[pos:])
        if got == 0:
            raise ConnectionError("Socket closed before all bytes received")
        pos += got


# -- GPU UUID helpers ----------------------------------------------------------

_UUID_TO_INDEX: dict[str, int] = {}


def _get_device_uuid(device_index: int) -> str:
    return str(torch.cuda.get_device_properties(device_index).uuid)


def _resolve_device_index(device_uuid: str) -> int:
    """Map a GPU UUID string to the local CUDA device ordinal."""
    if not _UUID_TO_INDEX:
        for i in range(torch.cuda.device_count()):
            _UUID_TO_INDEX[_get_device_uuid(i)] = i
    idx = _UUID_TO_INDEX.get(device_uuid)
    if idx is None:
        raise RuntimeError(
            f"Device UUID {device_uuid} not found among visible GPUs. "
            "Both processes must see the same GPU (same node, hostIPC: true in K8s)."
        )
    return idx


# -- IPC metadata serialization ------------------------------------------------


def _serialize_ipc_meta(
    tensor: torch.Tensor,
    event: torch.cuda.Event,
) -> bytes:
    """Pickle the CUDA IPC handle, tensor metadata, and event handle."""
    storage = tensor.untyped_storage()
    handle = storage._share_cuda_()
    meta = {
        "handle": handle,
        "dtype": tensor.dtype,
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "device_uuid": _get_device_uuid(tensor.device.index),
        "event_handle": event.ipc_handle(),
    }
    return pickle.dumps(meta)


def _deserialize_ipc_tensor(blob: bytes) -> tuple[torch.Tensor, torch.cuda.Event]:
    """Reconstruct a GPU tensor and interprocess event from pickled IPC metadata."""
    meta = pickle.loads(blob)  # noqa: S301
    device_index = _resolve_device_index(meta["device_uuid"])
    device = torch.device(f"cuda:{device_index}")

    storage = torch.UntypedStorage._new_shared_cuda(  # noqa: SLF001
        device_index, *meta["handle"][1:]
    )

    tensor = torch.empty((), device=device, dtype=meta["dtype"])
    tensor.set_(storage, meta["storage_offset"], meta["shape"], meta["stride"])

    event = torch.cuda.Event.from_ipc_handle(device, meta["event_handle"])
    return tensor, event


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

    # 2. Connect and transfer
    logger.info("Connecting to %s ...", socket_path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        while True:
            try:
                sock.connect(socket_path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.5)

        # Keep gathered tensors alive so the caching allocator doesn't recycle
        # the underlying CUDA blocks while the receiver still has them mapped.
        keepalive: list[torch.Tensor] = []

        transfer_start = time.perf_counter()

        for layer_idx in range(pool.num_layers):
            t0 = time.perf_counter()
            gathered = gather_kv(
                pool.kv_caches[layer_idx],
                block_tables,
                target_indices,
                seq_len,
            )

            # Record interprocess event so the receiver can wait for
            # the gather kernels to finish before reading the data.
            event = torch.cuda.Event(interprocess=True)
            event.record()
            torch.cuda.synchronize()
            gather_elapsed = time.perf_counter() - t0

            hdr = Header(layer_idx=layer_idx, **hdr_kwargs)

            # Serialize IPC handles (KB-scale, not the tensor payload)
            t1 = time.perf_counter()
            ipc_blob = _serialize_ipc_meta(gathered, event)
            serialize_elapsed = time.perf_counter() - t1

            # Send header + length-prefixed IPC blob
            t2 = time.perf_counter()
            hdr_bytes = hdr.to_bytes()
            ipc_len = struct.pack(">I", len(ipc_blob))
            sock.sendall(hdr_bytes + ipc_len + ipc_blob)
            send_elapsed = time.perf_counter() - t2

            keepalive.append(gathered)

            t3 = time.perf_counter()
            ack_layer = _unpack_ack(_recv_exact(sock, ACK_SIZE))
            if ack_layer != hdr.layer_idx:
                raise RuntimeError(
                    f"ACK mismatch: expected {hdr.layer_idx}, got {ack_layer}"
                )
            ack_elapsed = time.perf_counter() - t3

            layer_elapsed = time.perf_counter() - t0
            logger.info(
                "Layer %d/%d CUDA IPC sent in %.2f ms (%.2f MB/s). "
                "Gather: %.2f ms, Serialize: %.2f ms, Send: %.2f ms, Ack: %.2f ms.",
                hdr.layer_idx,
                hdr.num_layers - 1,
                layer_elapsed * 1000,
                hdr.tensor_size / (1024**2) / layer_elapsed if layer_elapsed > 0 else 0,
                gather_elapsed * 1000,
                serialize_elapsed * 1000,
                send_elapsed * 1000,
                ack_elapsed * 1000,
            )

        ack_layer = _unpack_ack(_recv_exact(sock, ACK_SIZE))
        if ack_layer != pool.num_layers:
            raise RuntimeError(
                f"ACK mismatch: expected {pool.num_layers}, got {ack_layer}"
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

        keepalive.clear()


# -- receiver -----------------------------------------------------------------


def _process_layer(
    ipc_blob: bytes,
    hdr: Header,
    logger: logging.Logger,
) -> torch.Tensor:
    """Reconstruct GPU tensor from CUDA IPC metadata and wait for visibility."""
    t0 = time.perf_counter()
    ipc_tensor, event = _deserialize_ipc_tensor(ipc_blob)
    event.synchronize()

    expected_shape = (hdr.num_targets, hdr.seq_len, 2, hdr.num_kv_heads, hdr.head_dim)
    if tuple(ipc_tensor.shape) != expected_shape:
        raise RuntimeError(
            f"Shape mismatch: expected {expected_shape}, got {tuple(ipc_tensor.shape)}"
        )
    deserialize_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    tensor = torch.empty(expected_shape, dtype=ipc_tensor.dtype).pin_memory()
    tensor.copy_(ipc_tensor)
    copy_elapsed = time.perf_counter() - t1

    elapsed = time.perf_counter() - t0
    logger.info(
        "Layer %d background done in %.1f ms (%.2f MB/s) → %s %s (device=%s). "
        "Deserialize: %.2f ms, Copy: %.2f ms.",
        hdr.layer_idx,
        elapsed * 1000,
        hdr.tensor_size / (1024**2) / elapsed if elapsed > 0 else 0,
        tensor.shape,
        tensor.dtype,
        tensor.device,
        deserialize_elapsed * 1000,
        copy_elapsed * 1000,
    )
    return tensor


def run_receiver(socket_path: str):
    logger = logging.getLogger("receiver")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA IPC receiver")

    torch.cuda.init()

    if os.path.exists(socket_path):
        os.remove(socket_path)

    logger.info("Listening on %s ...", socket_path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(socket_path)
        server.listen(1)

        conn, _ = server.accept()
        with conn:
            # No background processing here to avoid blocking the main thread.
            pending: list[tuple[bytes, Header]] = []
            transfer_start = time.perf_counter()
            num_layers: int | None = None
            total_bytes = 0

            while True:
                # Two-phase header read: fixed portion, then variable indices
                hdr_fixed = _recv_exact(conn, Header.SIZE)
                num_targets = struct.unpack(">I", hdr_fixed[-4:])[0]
                hdr_dynamic = b""
                if num_targets > 0:
                    hdr_dynamic = _recv_exact(conn, num_targets * 4)
                hdr = Header.from_bytes(hdr_fixed + hdr_dynamic)

                if num_layers is None:
                    num_layers = hdr.num_layers

                # Read length-prefixed IPC metadata blob
                ipc_len = struct.unpack(">I", _recv_exact(conn, 4))[0]
                ipc_blob = _recv_exact(conn, ipc_len)

                total_bytes += hdr.tensor_size

                # ACK immediately — sender can proceed
                conn.sendall(_pack_ack(hdr.layer_idx))

                pending.append((ipc_blob, hdr))

                if hdr.layer_idx == num_layers - 1:
                    break

            recv_elapsed = time.perf_counter() - transfer_start
            logger.info("All %d layers ACKed in %.2f s.", num_layers, recv_elapsed)

            results: dict[int, torch.Tensor] = {}
            for ipc_blob, hdr in pending:
                results[hdr.layer_idx] = _process_layer(ipc_blob, hdr, logger)

            # Final ACK for freeing the GPU tensors
            conn.sendall(_pack_ack(num_layers))

            total_elapsed = time.perf_counter() - transfer_start
            logger.info(
                "Background processing complete. Received %d layer tensors. "
                "Total wall time: %.2f s (%.2f MB/s).",
                len(results),
                total_elapsed,
                total_bytes / (1024**2) / total_elapsed if total_elapsed > 0 else 0,
            )

            results.clear()
            torch.cuda.ipc_collect()


# -- CLI -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise paged KV cache transfer via CUDA IPC over Unix domain sockets.",
    )
    role_group = parser.add_mutually_exclusive_group(required=True)
    role_group.add_argument("--sender", action="store_true")
    role_group.add_argument("--receiver", action="store_true")
    parser.add_argument("--socket-path", default="/tmp/cuda_ipc.sock")
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
