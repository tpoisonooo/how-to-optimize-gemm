#define SPDLOG_ACTIVE_LEVEL 5
#include "Shader.hpp"
#include "kompute/Kompute.hpp"
#include <cassert>
#include <chrono>

// MY_MMult = [                                                                    
// 64 0.41 0.000000e+00                                                            
// 96 1.25 0.000000e+00                                                            
// 128 2.46 0.000000e+00                                                           
// 160 4.48 0.000000e+00                                                           
// 192 6.28 0.000000e+00
// 224 7.84 0.000000e+00
// 256 8.51 0.000000e+00
// 288 8.14 0.000000e+00
// 320 9.30 0.000000e+00
// 352 9.90 0.000000e+00
// 384 10.71 0.000000e+00
// 416 11.09 0.000000e+00
// 448 11.38 0.000000e+00
// 480 11.94 0.000000e+00
// 512 12.27 0.000000e+00
// ];
float kompute(const std::string &shader_template, uint32_t m, uint32_t k,
              uint32_t n, float *a, float *b, float *c) {
  // build real .comp shader

  constexpr int local_size = 32;
  constexpr int block = 32;
  char shader[2048] = {0};
  sprintf(shader, shader_template.c_str(), local_size, local_size, block);
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

#if 0
    // use weired vk timestamps
    auto seq = mgr.sequence(0, 3);
    seq->record<kp::OpTensorSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpTensorSyncLocal>(params)
        ->eval();

    auto timestamps = seq->getTimestamps();
    assert(timestamps.size() == 4);
    auto computecost = timestamps[2] - timestamps[1];
     memcpy(c, tensorInC->data<float>(), m * n * sizeof(float));
     // return milliseconds ?
    return computecost/1000.f;
#else

  auto seq = mgr.sequence();
  seq->record<kp::OpTensorSyncDevice>(params)->eval();

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  seq->record<kp::OpAlgoDispatch>(algorithm)->eval();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  seq->record<kp::OpTensorSyncLocal>(params)->eval();

  auto count =
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
          .count();

  memcpy(c, tensorInC->data<float>(), m * n * sizeof(float));
  return count / 1e3f;
#endif
}

float MY_MMult(int m, int n, int k, float *a, float *b, float *c) {
  std::string shader_template = (R"(
        #version 450

        layout (local_size_x = %d, local_size_y = %d) in;

        layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 { float in_tensor_1[]; };
        layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 { float in_tensor_2[]; };
        layout (set = 0, binding = 2) writeonly buffer buf_out_tensor { float out_tensor[]; };

        layout (constant_id = 0) const float tensor_size_f = 0;

        void main()
        {
            uint block = %d;

            uint globalRow = gl_WorkGroupID.x * block + gl_LocalInvocationID.x;
            uint globalCol = gl_WorkGroupID.y * block +gl_LocalInvocationID.y;
            uint tensor_size = uint(tensor_size_f);
            
            float acc = 0.0;
            for(uint k = 0u; k < tensor_size; k++) {
                acc += in_tensor_1[(globalCol * tensor_size) + k] * in_tensor_2[(k * tensor_size) + globalRow];
            }
            out_tensor[(globalCol * tensor_size) + globalRow] = acc;
        }
    )");

  return kompute(shader_template, static_cast<uint32_t>(m),
                 static_cast<uint32_t>(n), static_cast<uint32_t>(k), a, b, c);
}
