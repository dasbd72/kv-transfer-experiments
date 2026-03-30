#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// ----------------------------------------------------------------------------
// Grid layout — both kernels share the same X dimension convention:
//   gridDim.x  = T * seq_len * 2 * H   (one CTA row per outer element)
//   gridDim.x carries the large outer dim: CUDA caps gridDim.y/z at 65535
//   but allows gridDim.x up to 2^31-1.
//
// The two kernels differ in how they tile the head-dim axis (Y):
//
//   gather_kv_kernel_vec    — 128 threads, VEC elements each → 128*VEC per CTA
//     gridDim.y = ceil(D / (128 * VEC))
//
//   gather_kv_kernel_scalar — 256 threads, 1 element each → 256 per CTA
//     gridDim.y = ceil(D / 256)
//
// All index arithmetic uses int32_t; dimensions of a realistic KV cache fit
// well within 2^31 (head_dim ≤ 512, heads ≤ 64, seq_len ≤ 64k, T ≤ 512).
// ----------------------------------------------------------------------------

// Vectorized kernel — output pointer is 16-byte aligned.
// Grid: (outer, ceil(D / (128 * VEC))),  block: 128 threads.
// Each thread issues one 128-bit load + one 128-bit store, no masking.
template <typename scalar_t, int VEC>
__global__ void gather_kv_kernel_vec(
    const scalar_t* __restrict__ kv_ptr, const int64_t* __restrict__ tb_ptr,
    scalar_t* __restrict__ out_ptr, int32_t kv_s0, int32_t kv_s1, int32_t kv_s2,
    int32_t kv_s3, int32_t tb_s0, int32_t tb_s1, int32_t o_s0, int32_t o_s1,
    int32_t o_s2, int32_t o_s3, int32_t seq_len, int32_t H, int32_t D,
    int32_t block_size, int32_t T) {
  using VecT = uint4;
  static_assert(sizeof(scalar_t) * VEC == sizeof(VecT),
                "scalar_t * VEC must equal 16 bytes");

  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int32_t h_idx = row % H;
  const int32_t rem1 = row / H;
  const int32_t kv_idx = rem1 % 2;
  const int32_t rem2 = rem1 / 2;
  const int32_t pos = rem2 % seq_len;
  const int32_t t_idx = rem2 / seq_len;

  __shared__ int64_t s_block_id;
  if (threadIdx.x == 0) {
    const int32_t b = pos / block_size;
    s_block_id = tb_ptr[t_idx * tb_s0 + b * tb_s1];
  }
  __syncthreads();

  const int32_t d_base = (static_cast<int32_t>(blockIdx.y) * blockDim.x +
                          static_cast<int32_t>(threadIdx.x)) *
                         VEC;
  if (d_base + VEC > D) return;

  const int32_t block_id = static_cast<int32_t>(s_block_id);
  const int32_t off = pos % block_size;

  const int32_t src_base =
      kv_idx * kv_s0 + block_id * kv_s1 + off * kv_s2 + h_idx * kv_s3 + d_base;
  const int32_t dst_base =
      t_idx * o_s0 + pos * o_s1 + kv_idx * o_s2 + h_idx * o_s3 + d_base;

  VecT v = *reinterpret_cast<const VecT*>(kv_ptr + src_base);
  *reinterpret_cast<VecT*>(out_ptr + dst_base) = v;
}

// Scalar kernel — used when the output pointer is not 16-byte aligned.
// Grid: (outer, ceil(D / 256)),  block: 256 threads.
// Each thread copies exactly one element; no VEC arithmetic or loops needed.
template <typename scalar_t>
__global__ void gather_kv_kernel_scalar(
    const scalar_t* __restrict__ kv_ptr, const int64_t* __restrict__ tb_ptr,
    scalar_t* __restrict__ out_ptr, int32_t kv_s0, int32_t kv_s1, int32_t kv_s2,
    int32_t kv_s3, int32_t tb_s0, int32_t tb_s1, int32_t o_s0, int32_t o_s1,
    int32_t o_s2, int32_t o_s3, int32_t seq_len, int32_t H, int32_t D,
    int32_t block_size, int32_t T) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int32_t h_idx = row % H;
  const int32_t rem1 = row / H;
  const int32_t kv_idx = rem1 % 2;
  const int32_t rem2 = rem1 / 2;
  const int32_t pos = rem2 % seq_len;
  const int32_t t_idx = rem2 / seq_len;

  __shared__ int64_t s_block_id;
  if (threadIdx.x == 0) {
    const int32_t b = pos / block_size;
    s_block_id = tb_ptr[t_idx * tb_s0 + b * tb_s1];
  }
  __syncthreads();

  const int32_t d = static_cast<int32_t>(blockIdx.y) * blockDim.x +
                    static_cast<int32_t>(threadIdx.x);
  if (d >= D) return;

  const int32_t block_id = static_cast<int32_t>(s_block_id);
  const int32_t off = pos % block_size;

  const int32_t src =
      kv_idx * kv_s0 + block_id * kv_s1 + off * kv_s2 + h_idx * kv_s3 + d;
  const int32_t dst =
      t_idx * o_s0 + pos * o_s1 + kv_idx * o_s2 + h_idx * o_s3 + d;

  out_ptr[dst] = kv_ptr[src];
}

// ----------------------------------------------------------------------------
// Launcher
// ----------------------------------------------------------------------------

template <typename scalar_t>
void launch_gather_kv_cuda_impl(const at::Tensor& kv_cache,
                                const at::Tensor& target_blocks,
                                int64_t seq_len, const at::Tensor& cpu_out) {
  // VEC: elements per 128-bit load.  fp16/bf16 → 8,  float32 → 4.
  constexpr int VEC = sizeof(uint4) / sizeof(scalar_t);

  const int32_t T = static_cast<int32_t>(target_blocks.size(0));
  const int32_t H = static_cast<int32_t>(kv_cache.size(3));
  const int32_t D = static_cast<int32_t>(kv_cache.size(4));
  const int32_t block_size = static_cast<int32_t>(kv_cache.size(2));
  const int32_t SL = static_cast<int32_t>(seq_len);

  // Map the pinned CPU buffer into the device address space so the kernel can
  // write directly to host memory without an intermediate GPU allocation.
  void* dev_ptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer(&dev_ptr, cpu_out.data_ptr(), 0);
  TORCH_CHECK(
      err == cudaSuccess,
      "gather_kv: cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));

  const auto kv_st = kv_cache.strides();
  const auto tb_st = target_blocks.strides();
  const auto ost = cpu_out.strides();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(kv_cache.get_device());

  auto* kv_ptr = kv_cache.data_ptr<scalar_t>();
  auto* tb_ptr = target_blocks.data_ptr<int64_t>();
  auto* out_ptr = static_cast<scalar_t*>(dev_ptr);

  const int32_t outer = T * SL * 2 * H;

  // kv_ptr is GPU-allocated (256-byte aligned) and all strides are multiples
  // of VEC, so the source is always aligned. Only the destination needs
  // checking (e.g. torch.frombuffer with a non-aligned header offset).
  const bool vec_aligned =
      (reinterpret_cast<uintptr_t>(dev_ptr) % sizeof(uint4)) == 0;

  if (vec_aligned) {
    // 128 threads × VEC elements per thread → 128*VEC elements per CTA.
    constexpr int VEC_THREADS = 128;
    const dim3 vec_grid(static_cast<unsigned>(outer),
                        static_cast<unsigned>((D + VEC_THREADS * VEC - 1) /
                                              (VEC_THREADS * VEC)));
    const dim3 vec_block(VEC_THREADS);

    gather_kv_kernel_vec<scalar_t, VEC><<<vec_grid, vec_block, 0, stream>>>(
        kv_ptr, tb_ptr, out_ptr, static_cast<int32_t>(kv_st[0]),
        static_cast<int32_t>(kv_st[1]), static_cast<int32_t>(kv_st[2]),
        static_cast<int32_t>(kv_st[3]), static_cast<int32_t>(tb_st[0]),
        static_cast<int32_t>(tb_st[1]), static_cast<int32_t>(ost[0]),
        static_cast<int32_t>(ost[1]), static_cast<int32_t>(ost[2]),
        static_cast<int32_t>(ost[3]), SL, H, D, block_size, T);
  } else {
    // 256 threads × 1 element per thread → 256 elements per CTA.
    constexpr int SCALAR_THREADS = 256;
    const dim3 scalar_grid(
        static_cast<unsigned>(outer),
        static_cast<unsigned>((D + SCALAR_THREADS - 1) / SCALAR_THREADS));
    const dim3 scalar_block(SCALAR_THREADS);

    gather_kv_kernel_scalar<scalar_t><<<scalar_grid, scalar_block, 0, stream>>>(
        kv_ptr, tb_ptr, out_ptr, static_cast<int32_t>(kv_st[0]),
        static_cast<int32_t>(kv_st[1]), static_cast<int32_t>(kv_st[2]),
        static_cast<int32_t>(kv_st[3]), static_cast<int32_t>(tb_st[0]),
        static_cast<int32_t>(tb_st[1]), static_cast<int32_t>(ost[0]),
        static_cast<int32_t>(ost[1]), static_cast<int32_t>(ost[2]),
        static_cast<int32_t>(ost[3]), SL, H, D, block_size, T);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void launch_gather_kv_cuda(const at::Tensor& kv_cache,
                           const at::Tensor& target_blocks, int64_t seq_len,
                           const at::Tensor& cpu_out) {
  switch (kv_cache.scalar_type()) {
    case at::ScalarType::Float:
      launch_gather_kv_cuda_impl<float>(kv_cache, target_blocks, seq_len,
                                        cpu_out);
      break;
    case at::ScalarType::Half:
      launch_gather_kv_cuda_impl<c10::Half>(kv_cache, target_blocks, seq_len,
                                            cpu_out);
      break;
    case at::ScalarType::BFloat16:
      launch_gather_kv_cuda_impl<c10::BFloat16>(kv_cache, target_blocks,
                                                seq_len, cpu_out);
      break;
    default:
      TORCH_CHECK(
          false, "gather_kv_cuda: expected float16, bfloat16, or float32, got ",
          kv_cache.scalar_type());
  }
}
