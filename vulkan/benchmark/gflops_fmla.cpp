#include "Shader.hpp"
#include "kompute/Kompute.hpp"
#include <iostream>
#include <cassert>
#include "types.h"

// gflops_fmla: 3795.930664 
void kompute(const std::string &shader) {
  kp::Manager mgr;
  kp::Workgroup workgroup({1, 1, 1});
  
  AlignVector data(32, 1.0f);
  auto dtype = kp::Tensor::TensorDataTypes::eFloat;
  auto tensorIn = mgr.tensor(data.data(), data.size(), sizeof(float), dtype);

  std::vector<std::shared_ptr<kp::Tensor>> params = {tensorIn};
  constexpr float LOOP = 10000;
  auto algorithm = mgr.algorithm(params, compileSource(shader), workgroup, {LOOP});
  auto seq = mgr.sequence(0, 2);

  seq->record<kp::OpTensorSyncDevice>(params)
      ->record<kp::OpAlgoDispatch>(algorithm)
      ->eval();

  auto timestamps = seq->getTimestamps();
  auto time_second = (timestamps[2] - timestamps[1]);

  fprintf(stdout, "gflops_fmla: %lf \n",  LOOP * 256 * 8/ time_second);
}

int main() {

  std::string shader = (R"(
        #version 450

        layout (local_size_x = 256) in;
        layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 { float in_tensor_1[]; };
        layout (constant_id = 0) const float loopf = 0;

        void main() {
            float v0 = float(gl_GlobalInvocationID.x);
            float v1, v2, v3, v4, v5, v6, v7;
            v1 = v2 = v3 = v4 = v5 = v6 = v7 = v0;

            for (float i = 0.0; i < loopf; i += 1.0) {
              v1 = v0 + v0 * 0.01;
              v2 = v1 + v1 * 0.01;
              v3 = v2 + v2 * 0.01;
              v4 = v3 + v3 * 0.01;
              v5 = v4 + v4 * 0.01;
              v6 = v5 + v5 * 0.01;
              v7 = v6 + v6 * 0.01;
              v0 = v7 + v7 * 0.01;
            }

            v0 = v1;
            v1 = v2;
            v2 = v3;
            v3 = v4;
            v4 = v5;
            v5 = v6;
            v6 = v7;
            v7 = v0;
        }
    )");

  kompute(shader);
}
