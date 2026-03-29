import struct
import time
from dataclasses import dataclass, field

DTYPE_ID: dict[str, int] = {"float16": 0, "bfloat16": 1, "float32": 2}
ID_TO_DTYPE_NAME: dict[int, str] = {v: k for k, v in DTYPE_ID.items()}


@dataclass
class Header:
    """v3 header for paged KV cache transfer.

    Wire format (big-endian, 60-byte fixed + variable tail):
        magic_number  uint32
        version       uint32   (must be 3)
        timestamp     int64    (ms since epoch)
        layer_idx     uint32
        num_layers    uint32
        tensor_size   uint64   (payload bytes following header)
        block_size    uint32
        num_kv_heads  uint32
        head_dim      uint32
        dtype_id      uint32   (see DTYPE_ID)
        seq_len       uint32
        num_targets   uint32
      variable:
        target_indices  num_targets × uint32
    """

    magic_number: int = 0x12345678
    version: int = 3
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    layer_idx: int = 0
    num_layers: int = 0
    tensor_size: int = 0
    block_size: int = 16
    num_kv_heads: int = 0
    head_dim: int = 0
    dtype_id: int = 0
    seq_len: int = 0
    num_targets: int = 0
    target_indices: tuple[int, ...] = ()

    _FORMAT = ">IIqIIQIIIIII"
    SIZE = struct.calcsize(_FORMAT)  # 60

    @property
    def total_header_size(self) -> int:
        return self.SIZE + self.num_targets * 4

    def to_bytes(self) -> bytes:
        fixed = struct.pack(
            self._FORMAT,
            self.magic_number,
            self.version,
            self.timestamp,
            self.layer_idx,
            self.num_layers,
            self.tensor_size,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            self.dtype_id,
            self.seq_len,
            self.num_targets,
        )
        if self.num_targets > 0:
            fixed += struct.pack(f">{self.num_targets}I", *self.target_indices)
        return fixed

    @classmethod
    def from_bytes(cls, data: bytes) -> "Header":
        if len(data) < cls.SIZE:
            raise ValueError(f"Need >= {cls.SIZE} bytes, got {len(data)}")

        (
            magic,
            version,
            timestamp,
            layer_idx,
            num_layers,
            tensor_size,
            block_size,
            num_kv_heads,
            head_dim,
            dtype_id,
            seq_len,
            num_targets,
        ) = struct.unpack(cls._FORMAT, data[: cls.SIZE])

        if magic != 0x12345678:
            raise ValueError(f"Invalid magic number: {hex(magic)}")
        if version != 3:
            raise ValueError(f"Unsupported header version: {version}")
        indices_end = cls.SIZE + num_targets * 4
        if len(data) < indices_end:
            raise ValueError(
                f"Need {indices_end} bytes for header + indices, got {len(data)}"
            )
        target_indices: tuple[int, ...] = ()
        if num_targets > 0:
            target_indices = struct.unpack(
                f">{num_targets}I", data[cls.SIZE : indices_end]
            )
        return cls(
            magic_number=magic,
            version=version,
            timestamp=timestamp,
            layer_idx=layer_idx,
            num_layers=num_layers,
            tensor_size=tensor_size,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_id=dtype_id,
            seq_len=seq_len,
            num_targets=num_targets,
            target_indices=target_indices,
        )

    @classmethod
    def read_from(cls, mm) -> "Header":
        """Two-phase read from an mmap: fixed portion, then variable indices."""
        fixed = mm.read(cls.SIZE)
        if len(fixed) < cls.SIZE:
            raise ValueError(f"Short mmap read: {len(fixed)} bytes")
        vals = struct.unpack(cls._FORMAT, fixed)
        num_targets = vals[-1]
        indices = mm.read(num_targets * 4) if num_targets > 0 else b""
        return cls.from_bytes(fixed + indices)
