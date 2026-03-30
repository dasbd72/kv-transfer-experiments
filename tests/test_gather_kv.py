"""Correctness tests: gather_kv (GPU reference) vs gather_kv_into_cpu (CUDA kernel → pinned CPU).

Run with:
    python test_gather_kv.py
"""

import pytest
import torch

from kv_layout import (
    KVProfile,
    PagedKVPool,
    allocate_requests,
    gather_kv,
    gather_kv_into_cpu,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _make_profile(
    num_kv_heads: int = 4,
    head_dim: int = 64,
    num_layers: int = 2,
    dtype_name: str = "float16",
) -> KVProfile:
    return KVProfile(
        model_id="test",
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes={"float32": 4, "float16": 2, "bfloat16": 2}[dtype_name],
        dtype_name=dtype_name,
    )


def _alloc_pinned(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, pin_memory=True)


def _alloc_pinned_misaligned(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Return a pinned tensor whose data_ptr is NOT 16-byte aligned.

    Allocates a flat pinned buffer with one extra leading element, then returns
    a contiguous view starting at element 1.  Because pin_memory allocations are
    page-aligned (4096 bytes), shifting by one element (2 bytes for fp16/bf16,
    4 bytes for fp32) breaks 16-byte alignment without leaving pinned memory.

    This replicates what happens in the memfd / shm transfer scripts when
    ``torch.frombuffer(mm, offset=hdr_size)`` is used and ``hdr_size % 16 != 0``.
    """
    numel = 1
    for s in shape:
        numel *= s
    flat = torch.empty(numel + 1, dtype=dtype, pin_memory=True)
    # flat.data_ptr() is page-aligned ⟹ flat[1:].data_ptr() = base + itemsize,
    # which is 2 or 4 bytes — never a multiple of 16.
    return flat[1:].reshape(shape)


def _run_both(
    pool: PagedKVPool,
    block_tables: torch.Tensor,
    target_indices: list[int],
    seq_len: int,
    layer: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (reference_cpu, kernel_cpu) for the given layer."""
    kv = pool.kv_caches[layer]
    T = len(target_indices)
    H, D = pool.num_kv_heads, pool.head_dim

    # Reference: gather_kv returns a GPU tensor; move to CPU for comparison.
    ref_gpu = gather_kv(kv, block_tables, target_indices, seq_len)
    ref_cpu = ref_gpu.cpu()

    # Kernel path: write directly into a pinned CPU buffer.
    out = _alloc_pinned((T, seq_len, 2, H, D), pool.dtype)
    gather_kv_into_cpu(kv, block_tables, target_indices, seq_len, out)
    torch.cuda.synchronize()

    return ref_cpu, out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
def test_dtype_correctness(dtype_name: str) -> None:
    """Kernel output matches reference for all supported dtypes."""
    profile = _make_profile(dtype_name=dtype_name)
    pool = PagedKVPool(profile, num_blocks=64, block_size=16)
    block_tables = allocate_requests(pool, num_requests=4, seq_len=32)

    ref, out = _run_both(pool, block_tables, [0, 1, 2, 3], seq_len=32)

    assert ref.shape == out.shape, f"Shape mismatch: {ref.shape} vs {out.shape}"
    assert torch.equal(ref, out), (
        f"[{dtype_name}] max abs diff = {(ref.float() - out.float()).abs().max()}"
    )


def test_partial_targets() -> None:
    """Exporting a subset of requests gives the same result as the reference."""
    profile = _make_profile()
    pool = PagedKVPool(profile, num_blocks=128, block_size=16)
    block_tables = allocate_requests(pool, num_requests=8, seq_len=48)

    for targets in ([2], [0, 5], [1, 3, 7]):
        ref, out = _run_both(pool, block_tables, targets, seq_len=48)
        assert torch.equal(ref, out), (
            f"Mismatch for targets={targets}: "
            f"max diff = {(ref.float() - out.float()).abs().max()}"
        )


def test_seq_len_not_multiple_of_block_size() -> None:
    """seq_len that doesn't fill the last block is handled correctly."""
    profile = _make_profile()
    pool = PagedKVPool(profile, num_blocks=64, block_size=16)
    # seq_len=30 requires ceil(30/16)=2 blocks but only uses 30 tokens.
    block_tables = allocate_requests(pool, num_requests=3, seq_len=30)

    ref, out = _run_both(pool, block_tables, [0, 1, 2], seq_len=30)
    assert torch.equal(ref, out), (
        f"seq_len=30 mismatch: max diff = {(ref.float() - out.float()).abs().max()}"
    )


def test_single_token() -> None:
    """Edge case: seq_len=1 (single token, one block)."""
    profile = _make_profile()
    pool = PagedKVPool(profile, num_blocks=16, block_size=16)
    block_tables = allocate_requests(pool, num_requests=2, seq_len=1)

    ref, out = _run_both(pool, block_tables, [0, 1], seq_len=1)
    assert torch.equal(ref, out), "Single-token mismatch"


def test_all_layers() -> None:
    """Both layers of a two-layer pool produce matching output."""
    profile = _make_profile(num_layers=2)
    pool = PagedKVPool(profile, num_blocks=64, block_size=16)
    block_tables = allocate_requests(pool, num_requests=4, seq_len=32)

    for layer in range(profile.num_layers):
        ref, out = _run_both(pool, block_tables, [0, 1, 2, 3], seq_len=32, layer=layer)
        assert torch.equal(ref, out), f"Layer {layer} mismatch"


def test_large_batch() -> None:
    """Larger batch/seq to exercise the kernel across many thread blocks."""
    profile = _make_profile(num_kv_heads=8, head_dim=128)
    pool = PagedKVPool(profile, num_blocks=512, block_size=16)
    block_tables = allocate_requests(pool, num_requests=16, seq_len=256)

    ref, out = _run_both(pool, block_tables, list(range(16)), seq_len=256)
    assert torch.equal(ref, out), (
        f"Large batch mismatch: max diff = {(ref.float() - out.float()).abs().max()}"
    )


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
def test_misaligned_output(dtype_name: str) -> None:
    """vec_aligned=False scalar fallback: output pointer is not 16-byte aligned.

    Replicates the memfd/shm transfer case where the tensor is created with
    ``torch.frombuffer(mm, offset=hdr_size)`` and ``hdr_size % 16 != 0``.
    The kernel must fall back to scalar stores instead of uint4 to avoid
    ``cudaErrorMisalignedAddress``.
    """
    dtype = DTYPE_MAP[dtype_name]
    profile = _make_profile(dtype_name=dtype_name)
    pool = PagedKVPool(profile, num_blocks=64, block_size=16)
    block_tables = allocate_requests(pool, num_requests=4, seq_len=32)
    kv = pool.kv_caches[0]
    T, SL, H, D = 4, 32, pool.num_kv_heads, pool.head_dim

    out = _alloc_pinned_misaligned((T, SL, 2, H, D), dtype)
    assert out.data_ptr() % 16 != 0, (
        f"Expected misaligned pointer, got {out.data_ptr():#x} (% 16 = {out.data_ptr() % 16})"
    )
    assert out.is_pinned(), "View of a pinned tensor must still report is_pinned()"
    assert out.is_contiguous(), "Reshape of a contiguous slice must be contiguous"

    gather_kv_into_cpu(kv, block_tables, list(range(T)), SL, out)
    torch.cuda.synchronize()

    ref = gather_kv(kv, block_tables, list(range(T)), SL).cpu()
    assert torch.equal(ref, out), (
        f"[{dtype_name}] misaligned: max diff = {(ref.float() - out.float()).abs().max()}"
    )


def test_requires_pinned_memory() -> None:
    """gather_kv_into_cpu must raise ValueError for non-pinned output."""
    profile = _make_profile()
    pool = PagedKVPool(profile, num_blocks=16, block_size=16)
    block_tables = allocate_requests(pool, num_requests=1, seq_len=16)
    kv = pool.kv_caches[0]

    out_unpinned = torch.empty(
        (1, 16, 2, pool.num_kv_heads, pool.head_dim), dtype=pool.dtype
    )
    with pytest.raises(ValueError, match="pinned"):
        gather_kv_into_cpu(kv, block_tables, [0], 16, out_unpinned)


def test_requires_cpu_tensor() -> None:
    """gather_kv_into_cpu must raise ValueError if out is on GPU."""
    profile = _make_profile()
    pool = PagedKVPool(profile, num_blocks=16, block_size=16)
    block_tables = allocate_requests(pool, num_requests=1, seq_len=16)
    kv = pool.kv_caches[0]

    out_gpu = torch.empty(
        (1, 16, 2, pool.num_kv_heads, pool.head_dim),
        dtype=pool.dtype,
        device="cuda",
    )
    with pytest.raises(ValueError, match="CPU"):
        gather_kv_into_cpu(kv, block_tables, [0], 16, out_gpu)
