#include "Shader.hpp"
#include "kompute/Kompute.hpp"
#include <iostream>
#include <cassert>
#include "types.h"

void kompute(const std::string &shader) {
  kp::Manager mgr;

  constexpr uint32_t MB = 256;
  constexpr uint32_t SIZE = MB * 1024 * 1024;
  constexpr uint32_t COUNT = SIZE/ sizeof(float); //  cannot exceed `vulkaninfo | grep maxComputeWorkGroupCount`
  assert(COUNT <= 2147483647);
  AlignVector data(COUNT, 3.14f);

  auto dtype = kp::Tensor::TensorDataTypes::eFloat;
  auto tensorIn = mgr.tensor(data.data(), data.size(), sizeof(float), dtype);
  auto tensorOut = mgr.tensor(data.data(), data.size(), sizeof(float), dtype);

  std::vector<std::shared_ptr<kp::Tensor>> params = {tensorIn, tensorOut};
  kp::Workgroup workgroup({COUNT / 256, 1, 1});

  auto algorithm = mgr.algorithm(params, compileSource(shader), workgroup);

  auto seq = mgr.sequence(0, 3);

  // h2d
  seq->record<kp::OpTensorSyncDevice>(params)
  // d2d
      ->record<kp::OpAlgoDispatch>(algorithm)
  // d2h
      ->record<kp::OpTensorSyncLocal>(params)
      ->eval();

  auto timestamps = seq->getTimestamps();
  assert(timestamps.size() == 4);
  auto h2d = (timestamps[1] - timestamps[0]) / 1e9f;
  auto d2d = (timestamps[2] - timestamps[1]) / 1e9f;
  auto d2h = (timestamps[3] - timestamps[2]) / 1e9f;

  fprintf(stdout, "h2d: %f MB/s, \nd2d %f MB/s, \nd2h: %f MB/s \n",  MB/h2d, MB/d2d,  MB/d2h);
}

int main() {

  std::string shader = (R"(
        #version 450

        layout (local_size_x = 256) in;

        // The input tensors bind index is relative to index in parameter passed
        layout(set = 0, binding = 0) buffer buf_in_a { float in_a[]; };
        layout(set = 0, binding = 1) buffer buf_out_a { float out_a[]; };

        void main() {
            uint index = gl_GlobalInvocationID.x;
            out_a[index] = in_a[index];
        }
    )");

  kompute(shader);
}
