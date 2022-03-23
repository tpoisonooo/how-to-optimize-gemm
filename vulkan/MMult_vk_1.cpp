#define SPDLOG_ACTIVE_LEVEL 5
#include "Shader.hpp"
#include "kompute/Kompute.hpp"
#include <cassert>
#include <chrono>
#include <iostream>

// MY_MMult = [
// 64 36.12 0.000000e+00
// 128 46.69 0.000000e+00
// 192 50.05 0.000000e+00
// 256 49.41 0.000000e+00
// 320 49.32 0.000000e+00
// 384 49.68 0.000000e+00
// 448 49.54 0.000000e+00
// 512 50.14 0.000000e+00
// ];
float kompute(const std::string &shader_template, uint32_t m, uint32_t k,
              uint32_t n, float *a, float *b, float *c) {
  // build real .comp shader

  constexpr int local_size = 16;
  constexpr int block = 16;
  char shader[2048] = {0};
  sprintf(shader, shader_template.c_str(), local_size, local_size, local_size,
          local_size, local_size, local_size, block);
  // fprintf(stdout, "%s\n", shader);

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
      mgr.algorithm(params, compileSource(shader), workgroup, {k * 1.f});

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
  std::string shader_template = (R"(
        #version 450

        layout (local_size_x = %d, local_size_y = %d) in;

        layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 { float in_tensor_1[]; };
        layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 { float in_tensor_2[]; };
        layout (set = 0, binding = 2) writeonly buffer buf_out_tensor { float out_tensor[]; };

        layout (constant_id = 0) const float tensor_size_f = 0;

        shared float sub_tensor_1[%d][%d];
        shared float sub_tensor_2[%d][%d];

        void main() {
            uint block = %d;
            uint tensor_size = uint(tensor_size_f);
            uint loop = tensor_size / block;

            uint threadIdx = gl_LocalInvocationID.x;
            uint threadIdy = gl_LocalInvocationID.y;

            uint globalCol = gl_WorkGroupID.y * block +threadIdy;
            uint globalRow = gl_WorkGroupID.x * block + threadIdx;

            float acc = 0.0;
            for (uint i = 0u; i < loop; ++i) {
                sub_tensor_1[threadIdy][threadIdx] = in_tensor_1[tensor_size * globalCol + i * block + threadIdx];
                sub_tensor_2[threadIdy][threadIdx] = in_tensor_2[tensor_size * (i * block + threadIdy) + globalRow];

                memoryBarrierShared();
                barrier();

                for (uint k = 0u; k < block; ++k) {
                    acc += sub_tensor_1[threadIdy][k] * sub_tensor_2[k][threadIdx];
                }
                barrier();
            }

            out_tensor[(globalCol * tensor_size) + globalRow] = acc;
        }
    )");

  return kompute(shader_template, static_cast<uint32_t>(m),
                 static_cast<uint32_t>(n), static_cast<uint32_t>(k), a, b, c);
}
