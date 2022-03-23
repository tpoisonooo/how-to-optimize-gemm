#include "Shader.hpp"
#include "kompute/Kompute.hpp"
#include <iostream>
#include <cassert>
#include "types.h"

// gmem2smem 80.194374 ns ~  72 cycle  0.899 GHz
void kompute(const std::string &shader) {
  kp::Manager mgr;

  constexpr uint32_t SIZE = 128; // 128B
  constexpr uint32_t COUNT = SIZE/ sizeof(float); //  cannot exceed `vulkaninfo | grep maxComputeWorkGroupCount`
  AlignVector data(COUNT, 3.14f);

  auto dtype = kp::Tensor::TensorDataTypes::eFloat;
  auto tensorIn = mgr.tensor(data.data(), data.size(), sizeof(float), dtype);

  std::vector<std::shared_ptr<kp::Tensor>> params = {tensorIn};
  kp::Workgroup workgroup({1, 1, 1});
  constexpr float LOOP = 10000000.f;
  auto algorithm = mgr.algorithm(params, compileSource(shader), workgroup, {LOOP});

  auto seq = mgr.sequence(0, 2);

  seq->record<kp::OpTensorSyncDevice>(params)
      ->record<kp::OpAlgoDispatch>(algorithm)
      ->eval();

  auto timestamps = seq->getTimestamps();
  assert(timestamps.size() == 3);
  auto gmem2smem = (timestamps[2] - timestamps[1]);

  const float ns = gmem2smem / LOOP;
  constexpr float GHz = 921/ 1024.f; // jetson nano max_frequency.
  const int cycle = ns * GHz;
  fprintf(stdout, "***** gmem2smem %f ns ~ %d cycle  %0.3f GHz \n",  ns, cycle, GHz);
}

int main() {

  std::string shader = (R"(
        #version 450

        layout (local_size_x = 32) in;

        // The input tensors bind index is relative to index in parameter passed
        layout(set = 0, binding = 0) buffer buf_in_a { float in_a[]; };
        layout (constant_id = 0) const float tensor_size_f = 0;

        shared float sub_tensor_1[32];

        void main() {
            uint index = gl_GlobalInvocationID.x;
            uint loop = uint(tensor_size_f);
            for (uint x = 0; x < loop; ++x) {
              sub_tensor_1[index] = in_a[index];
              barrier();
            }
        }
    )");

  kompute(shader);
}
