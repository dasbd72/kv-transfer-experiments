#include <torch/extension.h>

void launch_gather_kv_cuda(const at::Tensor& kv_cache,
                           const at::Tensor& target_blocks, int64_t seq_len,
                           const at::Tensor& cpu_out);

void gather_kv_to_cpu(const at::Tensor& kv_cache,
                      const at::Tensor& target_blocks, int64_t seq_len,
                      at::Tensor& cpu_out) {
  TORCH_CHECK(kv_cache.is_cuda(), "gather_kv: kv_cache must be a CUDA tensor");
  TORCH_CHECK(target_blocks.is_cuda(),
              "gather_kv: target_blocks must be a CUDA tensor");
  TORCH_CHECK(!cpu_out.is_cuda(), "gather_kv: cpu_out must be a CPU tensor");
  TORCH_CHECK(cpu_out.is_pinned(),
              "gather_kv: cpu_out must be pinned (page-locked) memory");
  TORCH_CHECK(
      kv_cache.dim() == 5,
      "gather_kv: kv_cache must be 5D (2, num_blocks, block_size, H, D)");
  TORCH_CHECK(target_blocks.dim() == 2, "gather_kv: target_blocks must be 2D");
  TORCH_CHECK(cpu_out.dim() == 5, "gather_kv: cpu_out must be 5D");
  TORCH_CHECK(target_blocks.scalar_type() == at::ScalarType::Long,
              "gather_kv: target_blocks must be int64");

  const int64_t T = target_blocks.size(0);
  const int64_t H = kv_cache.size(3);
  const int64_t D = kv_cache.size(4);
  TORCH_CHECK(cpu_out.sizes() == at::IntArrayRef({T, seq_len, 2, H, D}),
              "gather_kv: cpu_out shape must be (T, seq_len, 2, H, D)");
  TORCH_CHECK(cpu_out.scalar_type() == kv_cache.scalar_type(),
              "gather_kv: cpu_out dtype must match kv_cache");
  TORCH_CHECK(kv_cache.is_contiguous(),
              "gather_kv: kv_cache must be contiguous");
  TORCH_CHECK(target_blocks.is_contiguous(),
              "gather_kv: target_blocks must be contiguous");
  TORCH_CHECK(cpu_out.is_contiguous(), "gather_kv: cpu_out must be contiguous");

  const int64_t numel = T * seq_len * 2 * H * D;
  if (numel == 0) {
    return;
  }

  launch_gather_kv_cuda(kv_cache, target_blocks, seq_len, cpu_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_kv_to_cpu", &gather_kv_to_cpu,
        "Gather paged KV directly into a pinned CPU tensor");
}
