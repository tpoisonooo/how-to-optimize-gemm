#define SPDLOG_ACTIVE_LEVEL 5
#include "Shader.hpp"
#include "kompute/Kompute.hpp"
#include <cassert>
#include <chrono>
#include <iostream>

// MY_MMult = [
// 64 36.24 0.000000e+00 
// 128 48.75 0.000000e+00 
// 192 51.88 0.000000e+00 
// 256 52.91 0.000000e+00 
// 320 53.30 0.000000e+00 
// 384 53.51 0.000000e+00 
// 448 53.64 0.000000e+00 
// 512 53.72 0.000000e+00 
// ];
float kompute(const std::string &comp, uint32_t m, uint32_t k,
              uint32_t n, float *a, float *b, float *c) {
  constexpr uint32_t local_size = 16;
  kp::Manager mgr;

  // Create and initialise Kompute Tensors through manager
  auto dtype = kp::Tensor::TensorDataTypes::eFloat;
  auto tensorInA = mgr.tensor(a, m * k, sizeof(float), dtype);
  auto tensorInB = mgr.tensor(b, k * n, sizeof(float), dtype);
  auto tensorInC = mgr.tensor(c, m * n, sizeof(float), dtype);

  std::vector<std::shared_ptr<kp::Tensor>> params = {tensorInA, tensorInB,
                                                     tensorInC};

  // Create algorithm based on shader (supports buffers & push/spec constants)
  kp::Workgroup workgroup({m / local_size, n / local_size, 1});

  auto algorithm =
      mgr.algorithm(params, compileFile(comp), workgroup, {k * 1.f});

    mgr.sequence()->record<kp::OpTensorSyncDevice>(params)->eval();
    // use weired vk timestamps
    auto seq = mgr.sequence(0, 1);
    seq->record<kp::OpAlgoDispatch>(algorithm)->eval();

    mgr.sequence()->record<kp::OpTensorSyncLocal>(params)->eval();

    auto timestamps = seq->getTimestamps();
    auto computecost = timestamps[1] - timestamps[0];
     memcpy(c, tensorInC->data<float>(), m * n * sizeof(float));
    return computecost/1e6f;
}

float MY_MMult(int m, int n, int k, float *a, float *b, float *c) {
  return kompute("MMult_vk_2.comp", static_cast<uint32_t>(m),
                 static_cast<uint32_t>(n), static_cast<uint32_t>(k), a, b, c);
}
