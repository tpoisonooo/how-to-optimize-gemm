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
  auto tensorIn1 = mgr.tensor(data.data(), data.size(), sizeof(float), dtype);
  auto tensorIn2 = mgr.tensor(data.data(), data.size(), sizeof(float), dtype);
  auto tensorOut = mgr.tensor(data.data(), data.size(), sizeof(float), dtype);

  std::vector<std::shared_ptr<kp::Tensor>> params = {tensorIn1, tensorIn2, tensorOut};
  kp::Workgroup workgroup({COUNT / 256, 1, 1});

  auto algorithm = mgr.algorithm(params, compileFile(shader), workgroup);

  auto seq = mgr.sequence(0, 3);

  seq->record<kp::OpTensorSyncDevice>(params)
      ->record<kp::OpAlgoDispatch>(algorithm)
      ->record<kp::OpTensorSyncLocal>(params)
      ->eval();

  auto timestamps = seq->getTimestamps();
  for (int i = 0; i < timestamps.size() -1; ++i) {
    auto cost = timestamps[i+1] - timestamps[i];
    fprintf(stdout, "time cost %ld  %0.4f GB/s \n", cost, MB / (cost/1e9f) / 1000.f);
  }
  // auto h2d = (timestamps[1] - timestamps[0]) / 1e9f;
  // auto d2d = (timestamps[2] - timestamps[1]) / 1e9f;
  // auto d2h = (timestamps[3] - timestamps[2]) / 1e9f;

  // fprintf(stdout, "h2d: %f MB/s, \nd2d %f MB/s, \nd2h: %f MB/s \n",  MB/h2d, MB/d2d,  MB/d2h);
}

int main() {
  kompute("gmem_bandwidth.comp");
}
