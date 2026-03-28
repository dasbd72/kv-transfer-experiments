import struct
import time
from dataclasses import dataclass


@dataclass
class Header:
    magic_number: int = 0x12345678
    version: int = 1
    tensor_size: int = 0
    timestamp: int = int(time.time() * 1000)  # milliseconds since epoch

    # Format: > (big-endian), I (4-byte unsigned integer) x 3
    _FORMAT = ">IIIq"
    # Pre-calculate the size of the struct (12 bytes)
    SIZE = struct.calcsize(_FORMAT)

    def to_bytes(self) -> bytes:
        """Packs the instance data into big-endian bytes."""
        return struct.pack(
            self._FORMAT,
            self.magic_number,
            self.version,
            self.tensor_size,
            self.timestamp,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "Header":
        """Unpacks big-endian bytes into a Header instance."""
        if len(data) < cls.SIZE:
            raise ValueError(f"Expected at least {cls.SIZE} bytes, got {len(data)}")

        # Unpack exactly the number of bytes we need
        (
            magic_number,
            version,
            tensor_size,
            timestamp,
        ) = struct.unpack(cls._FORMAT, data[: cls.SIZE])

        if magic_number != 0x12345678:  # Compare against the expected constant
            raise ValueError(f"Invalid magic number: {hex(magic_number)}")
        if version != 1:
            raise ValueError(f"Invalid version: {version}")

        return cls(
            magic_number=magic_number,
            version=version,
            tensor_size=tensor_size,
            timestamp=timestamp,
        )
