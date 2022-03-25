#define SPDLOG_ACTIVE_LEVEL 5
#include "Shader.hpp"
#include "kompute/Kompute.hpp"
#include <cassert>
#include <chrono>
#include <iostream>

// MY_MMult = [
// 64 20.13 2.861023e-06 
// 128 22.85 5.722046e-06 
// 192 23.48 1.144409e-05 
// 256 23.24 1.716614e-05 
// 320 23.26 2.098083e-05 
// 384 23.29 2.288818e-05 
// 448 23.30 2.861023e-05 
// 512 23.31 3.433228e-05 
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

    auto seq = mgr.sequence(0, 3);
    seq->record<kp::OpTensorSyncDevice>(params)
      ->record<kp::OpAlgoDispatch>(algorithm)
      ->record<kp::OpTensorSyncLocal>(params)->eval();

    auto timestamps = seq->getTimestamps();
    auto computecost = timestamps[2] - timestamps[1];
     memcpy(c, tensorInC->data<float>(), m * n * sizeof(float));
    return computecost/1e6f;
}

float MY_MMult(int m, int n, int k, float *a, float *b, float *c) {
  return kompute("MMult_vk_3.comp", static_cast<uint32_t>(m),
                 static_cast<uint32_t>(n), static_cast<uint32_t>(k), a, b, c);
}
