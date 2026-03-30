"""Derive per-layer KV cache dimensions from a Hugging Face model config.

The sizing formula mirrors vLLM's AttentionSpec.real_page_size_bytes
(see vllm/v1/kv_cache_interface.py):

    per_token_per_layer = 2 * num_kv_heads * head_dim * dtype_bytes

MLA / FP8 / other non-standard cache formats use a different formula and
are NOT supported here; load_kv_profile() will raise for unsupported dtypes.
"""

import os
from dataclasses import dataclass

import torch
from transformers import AutoConfig

# JIT CUDA extension for gather (avoids Triton kernel compile/cold start on each process).
_gather_kv_cuda_mod: object | None = None
_gather_kv_cuda_load_failed = False


def _get_gather_kv_cuda():
    """Load ``csrc/gather_kv*.`` once; return module or None if unavailable."""
    global _gather_kv_cuda_mod, _gather_kv_cuda_load_failed
    if _gather_kv_cuda_load_failed:
        return None
    if _gather_kv_cuda_mod is not None:
        return _gather_kv_cuda_mod
    try:
        if not torch.cuda.is_available():
            _gather_kv_cuda_load_failed = True
            return None
        from torch.utils.cpp_extension import load

        _dir = os.path.dirname(os.path.abspath(__file__))
        _gather_kv_cuda_mod = load(
            name="gather_kv_cuda",
            sources=[
                os.path.join(_dir, "csrc", "gather_kv.cpp"),
                os.path.join(_dir, "csrc", "gather_kv_cuda.cu"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        return _gather_kv_cuda_mod
    except Exception:
        _gather_kv_cuda_load_failed = True
        _gather_kv_cuda_mod = None
        return None


_get_gather_kv_cuda()


DTYPE_BYTES: dict[str, int] = {
    "float16": 2,
    "float32": 4,
    "bfloat16": 2,
}

TORCH_DTYPE: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class KVProfile:
    """Per-layer KV cache sizing derived from a HuggingFace model config."""

    model_id: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int
    dtype_name: str = "float16"

    @property
    def kv_bytes_per_token(self) -> int:
        """Bytes for K+V of one token in one layer."""
        return 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes

    def layer_bytes(self, batch_size: int, seq_len: int) -> int:
        """Total bytes for one layer across the full batch."""
        return batch_size * seq_len * self.kv_bytes_per_token

    def request_layer_bytes(self, seq_len: int) -> int:
        """Bytes for one request's KV in one layer."""
        return seq_len * self.kv_bytes_per_token

    @property
    def torch_dtype(self) -> torch.dtype:
        return TORCH_DTYPE[self.dtype_name]


def load_kv_profile(model_id: str) -> KVProfile:
    """Load a HuggingFace model config and extract KV cache dimensions.

    Raises ValueError for unsupported dtypes (e.g. fp8, MLA-specific formats).
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    num_layers: int = config.num_hidden_layers

    num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = config.num_attention_heads

    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = getattr(config, "attention_head_size", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads

    torch_dtype = str(getattr(config, "torch_dtype", "float16")).replace("torch.", "")
    dtype_bytes = DTYPE_BYTES.get(torch_dtype)
    if dtype_bytes is None:
        raise ValueError(
            f"Unsupported dtype '{torch_dtype}' for model {model_id}. "
            f"Supported: {sorted(DTYPE_BYTES)}"
        )

    return KVProfile(
        model_id=model_id,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
        dtype_name=torch_dtype,
    )


# -- Paged KV cache (vLLM FlashAttention layout) ------------------------------


class PagedKVPool:
    """GPU paged KV cache pool matching vLLM FlashAttention layout.

    Per-layer tensor shape: ``(2, num_blocks, block_size, num_kv_heads, head_dim)``
    where dim 0 distinguishes K (index 0) from V (index 1).

    This matches ``FlashAttentionBackend.get_kv_cache_shape()`` with the
    default NHD stride order (vllm/v1/attention/backends/flash_attn.py).
    """

    def __init__(
        self,
        profile: KVProfile,
        num_blocks: int,
        block_size: int = 16,
        device: torch.device | str = "cuda",
    ):
        if block_size % 16 != 0:
            raise ValueError("block_size must be a multiple of 16 (vLLM constraint)")

        self.num_layers = profile.num_layers
        self.num_kv_heads = profile.num_kv_heads
        self.head_dim = profile.head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = profile.torch_dtype
        self.dtype_name = profile.dtype_name
        self.device = torch.device(device)

        self.kv_caches: list[torch.Tensor] = []
        for _ in range(self.num_layers):
            self.kv_caches.append(
                torch.randn(
                    2,
                    num_blocks,
                    block_size,
                    self.num_kv_heads,
                    self.head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

        self._next_free = 0

    def allocate_blocks(self, n: int) -> list[int]:
        """Bump allocator: return *n* consecutive block IDs."""
        if self._next_free + n > self.num_blocks:
            raise RuntimeError(
                f"Out of blocks: need {n}, have {self.num_blocks - self._next_free}"
            )
        start = self._next_free
        self._next_free += n
        return list(range(start, start + n))


def allocate_requests(
    pool: PagedKVPool,
    num_requests: int,
    seq_len: int,
) -> torch.Tensor:
    """Allocate blocks for *num_requests* and return the block table.

    Returns:
        block_tables: ``LongTensor[num_requests, blocks_per_req]`` on
        ``pool.device``.
    """
    blocks_per_req = (seq_len + pool.block_size - 1) // pool.block_size
    rows: list[list[int]] = []
    for _ in range(num_requests):
        rows.append(pool.allocate_blocks(blocks_per_req))
    return torch.tensor(rows, dtype=torch.long, device=pool.device)


def gather_kv(
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    target_indices: list[int] | torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Gather target requests' KV into a contiguous export tensor.

    Args:
        kv_cache: ``(2, num_blocks, block_size, num_kv_heads, head_dim)``
        block_tables: ``(num_requests, blocks_per_req)`` — int64 on same device
        target_indices: which request rows to export
        seq_len: tokens per request (may be < blocks_per_req * block_size)

    Returns:
        Contiguous GPU tensor of shape
        ``(num_targets, seq_len, 2, num_kv_heads, head_dim)``.
    """
    target_blocks = block_tables[target_indices]  # (T, blocks_per_req)
    num_targets = target_blocks.shape[0]
    blocks_per_req = target_blocks.shape[1]
    block_size = kv_cache.shape[2]
    H, D = kv_cache.shape[3], kv_cache.shape[4]

    flat = target_blocks.reshape(-1)  # (T * blocks_per_req,)

    # Advanced-index dim 1: (2, T*bpr, block_size, H, D)
    gathered = kv_cache[:, flat].contiguous()
    # Merge block dims → (2, T, bpr*block_size, H, D), then trim to seq_len
    gathered = gathered.reshape(2, num_targets, blocks_per_req * block_size, H, D)
    gathered = gathered[:, :, :seq_len]
    # → (T, seq_len, 2, H, D)
    return gathered.permute(1, 2, 0, 3, 4).contiguous()


def gather_kv_into_cpu(
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    target_indices: list[int] | torch.Tensor,
    seq_len: int,
    out: torch.Tensor,
) -> None:
    """Dense KV export: paged cache → preallocated pinned CPU buffer.

    Selects rows ``block_tables[target_indices]``, gathers K/V block slices for
    the first ``seq_len`` tokens per request, and writes directly into ``out``
    via ``cudaHostGetDevicePointer`` — no intermediate GPU staging buffer is
    allocated.

    Args:
        kv_cache: Paged layout
            ``(2, num_blocks, block_size, num_kv_heads, head_dim)``.
        block_tables: ``(num_requests, blocks_per_req)``, int64, same device
            as ``kv_cache``.
        target_indices: Which rows of ``block_tables`` to export.
        seq_len: Sequence length; may be less than ``blocks_per_req * block_size``
            (tail of the last block is unused).
        out: **Pinned** CPU tensor ``(num_targets, seq_len, 2, num_kv_heads, head_dim)``
            where ``num_targets`` is the leading size of ``target_indices``
            (same as ``block_tables[target_indices].shape[0]``), dtype matching
            ``kv_cache``. Must be allocated with ``pin_memory=True``.

    Raises:
        ValueError: If ``out`` is not a pinned CPU tensor, or dtype/shape do not match.
    """
    if out.device.type != "cpu":
        raise ValueError("out must be a CPU tensor")
    if not out.is_pinned():
        raise ValueError(
            "out must be pinned memory (allocate with pin_memory=True) so the "
            "CUDA kernel can write directly to it without a staging buffer"
        )

    target_blocks = block_tables[target_indices].contiguous()
    T, _ = target_blocks.shape
    *_, H, D = kv_cache.shape

    if out.dtype != kv_cache.dtype:
        raise ValueError(
            f"out dtype {out.dtype} must match kv_cache dtype {kv_cache.dtype}"
        )
    expected = (T, seq_len, 2, H, D)
    if out.shape != expected:
        raise ValueError(
            f"out shape {tuple(out.shape)} does not match expected {expected}"
        )

    mod = _get_gather_kv_cuda()
    mod.gather_kv_to_cpu(kv_cache, target_blocks, seq_len, out)
