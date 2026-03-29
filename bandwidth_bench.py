"""Benchmark PCIe and memory-copy bandwidth.

Tests:
  1. PCIe Host-to-Device (H2D) — pinned and pageable CPU memory → GPU
  2. PCIe Device-to-Host (D2H) — GPU → pinned and pageable CPU memory
  3. Device-to-Device (D2D)    — GPU global memory copy
  4. Host-to-Host (H2H)        — CPU memory copy

Each test runs over a configurable range of buffer sizes with warmup
iterations and reports throughput in GB/s.  GPU-involved transfers are
timed with CUDA events for precision; CPU-only copies use
time.perf_counter.
"""

import argparse
import logging
import time
from dataclasses import dataclass

import torch

GB = 1 << 30
MB = 1 << 20


# -- result container ---------------------------------------------------------


@dataclass
class BenchResult:
    label: str
    size_bytes: int
    elapsed_s: float
    iters: int

    @property
    def bw_gbs(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return (self.size_bytes * self.iters) / self.elapsed_s / GB


# -- CUDA-event timing helpers ------------------------------------------------


def _cuda_timed_copy(
    dst: torch.Tensor,
    src: torch.Tensor,
    iters: int,
    warmup: int,
    stream: torch.cuda.Stream | None = None,
) -> float:
    """Return elapsed seconds for *iters* non_blocking copies timed via CUDA events."""
    s = stream or torch.cuda.current_stream()
    for _ in range(warmup):
        with torch.cuda.stream(s):
            dst.copy_(src, non_blocking=True)
    s.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(s)
    for _ in range(iters):
        with torch.cuda.stream(s):
            dst.copy_(src, non_blocking=True)
    end.record(s)
    end.synchronize()
    return start.elapsed_time(end) / 1000.0


# -- individual benchmarks ----------------------------------------------------


def bench_h2d(
    size: int,
    iters: int,
    warmup: int,
    pinned: bool,
    device: torch.device,
    stream: torch.cuda.Stream,
) -> BenchResult:
    """Host-to-Device transfer."""
    cpu = torch.empty(size, dtype=torch.uint8)
    if pinned:
        cpu = cpu.pin_memory()
    gpu = torch.empty(size, dtype=torch.uint8, device=device)

    elapsed = _cuda_timed_copy(gpu, cpu, iters, warmup, stream)
    kind = "pinned" if pinned else "pageable"
    return BenchResult(f"H2D ({kind})", size, elapsed, iters)


def bench_d2h(
    size: int,
    iters: int,
    warmup: int,
    pinned: bool,
    device: torch.device,
    stream: torch.cuda.Stream,
) -> BenchResult:
    """Device-to-Host transfer."""
    gpu = torch.empty(size, dtype=torch.uint8, device=device)
    cpu = torch.empty(size, dtype=torch.uint8)
    if pinned:
        cpu = cpu.pin_memory()

    elapsed = _cuda_timed_copy(cpu, gpu, iters, warmup, stream)
    kind = "pinned" if pinned else "pageable"
    return BenchResult(f"D2H ({kind})", size, elapsed, iters)


def bench_d2d(
    size: int,
    iters: int,
    warmup: int,
    device: torch.device,
    stream: torch.cuda.Stream,
) -> BenchResult:
    """Device-to-Device (GPU global memory) copy."""
    src = torch.empty(size, dtype=torch.uint8, device=device)
    dst = torch.empty(size, dtype=torch.uint8, device=device)

    elapsed = _cuda_timed_copy(dst, src, iters, warmup, stream)
    return BenchResult("D2D (GPU)", size, elapsed, iters)


def bench_h2h(
    size: int,
    iters: int,
    warmup: int,
) -> BenchResult:
    """Host-to-Host (CPU memory) copy via torch.Tensor.copy_."""
    src = torch.empty(size, dtype=torch.uint8)
    dst = torch.empty(size, dtype=torch.uint8)

    for _ in range(warmup):
        dst.copy_(src)

    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    elapsed = time.perf_counter() - t0
    return BenchResult("H2H (CPU)", size, elapsed, iters)


# -- size schedule ------------------------------------------------------------

DEFAULT_SIZES_MB = [1, 4, 16, 64, 256, 512, 1024]


def _parse_sizes(raw: str) -> list[int]:
    """Parse comma-separated MB values into byte counts."""
    return [int(s.strip()) * MB for s in raw.split(",")]


# -- pretty printing ----------------------------------------------------------


def _print_table(results: list[BenchResult], logger: logging.Logger):
    hdr = f"{'Test':<22} {'Size':>10} {'Iters':>6} {'Time (ms)':>12} {'BW (GB/s)':>12}"
    logger.info(hdr)
    logger.info("-" * len(hdr))
    for r in results:
        size_str = f"{r.size_bytes / MB:.0f} MB"
        time_ms = r.elapsed_s * 1000
        logger.info(
            f"{r.label:<22} {size_str:>10} {r.iters:>6} {time_ms:>12.2f} {r.bw_gbs:>12.2f}"
        )


# -- orchestrators ------------------------------------------------------------


def run_pcie_bench(
    sizes: list[int],
    iters: int,
    warmup: int,
    device: torch.device,
    logger: logging.Logger,
):
    stream = torch.cuda.Stream(device)
    results: list[BenchResult] = []

    for size in sizes:
        results.append(
            bench_h2d(size, iters, warmup, pinned=True, device=device, stream=stream)
        )
        results.append(
            bench_h2d(size, iters, warmup, pinned=False, device=device, stream=stream)
        )
        results.append(
            bench_d2h(size, iters, warmup, pinned=True, device=device, stream=stream)
        )
        results.append(
            bench_d2h(size, iters, warmup, pinned=False, device=device, stream=stream)
        )

    logger.info("")
    logger.info("=== PCIe Bandwidth ===")
    _print_table(results, logger)


def run_memcpy_bench(
    sizes: list[int],
    iters: int,
    warmup: int,
    device: torch.device,
    logger: logging.Logger,
):
    stream = torch.cuda.Stream(device)
    results: list[BenchResult] = []

    for size in sizes:
        results.append(bench_d2d(size, iters, warmup, device=device, stream=stream))
        results.append(bench_h2h(size, iters, warmup))

    logger.info("")
    logger.info("=== Memory Copy Bandwidth ===")
    _print_table(results, logger)


# -- CLI -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PCIe and memory-copy bandwidth.",
    )
    parser.add_argument(
        "--sizes",
        default=",".join(str(s) for s in DEFAULT_SIZES_MB),
        help="Comma-separated buffer sizes in MB (default: %(default)s).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Timed iterations per size (default: %(default)s).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations before timing (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device ordinal (default: %(default)s).",
    )
    parser.add_argument(
        "--test",
        choices=["all", "pcie", "memcpy"],
        default="all",
        help="Which benchmark suite to run (default: %(default)s).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("bandwidth_bench")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bandwidth benchmarks")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    sizes = _parse_sizes(args.sizes)

    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory
    logger.info("GPU %d: %s (%.1f GB)", args.device, gpu_name, gpu_mem / GB)
    logger.info(
        "Sizes (MB): %s | iters=%d warmup=%d",
        [s // MB for s in sizes],
        args.iters,
        args.warmup,
    )

    if args.test in ("all", "pcie"):
        run_pcie_bench(sizes, args.iters, args.warmup, device, logger)

    if args.test in ("all", "memcpy"):
        run_memcpy_bench(sizes, args.iters, args.warmup, device, logger)


if __name__ == "__main__":
    main()
