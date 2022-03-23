#include "Shader.hpp"
#include "kompute/Kompute.hpp"
#include <iostream>
#include <cassert>
#include "types.h"

// gmem2smem bandwidth 11.395 GB/s
void kompute(const std::string &shader) {
  kp::Manager mgr;

  constexpr uint32_t SIZE_IN_BYTES = 32768;
  constexpr uint32_t BLOCK = 32;
  AlignVector data(SIZE_IN_BYTES/ sizeof(float), 3.14f);

  auto dtype = kp::Tensor::TensorDataTypes::eFloat;
  auto tensorIn = mgr.tensor(data.data(), data.size(), sizeof(float), dtype);

  std::vector<std::shared_ptr<kp::Tensor>> params = {tensorIn};
  kp::Workgroup workgroup({SIZE_IN_BYTES / BLOCK, 1, 1});
  constexpr float LOOP = 1000000.f;
  auto algorithm = mgr.algorithm(params, compileFile(shader), workgroup, {LOOP});

  auto seq = mgr.sequence(0, 2);

  seq->record<kp::OpTensorSyncDevice>(params)
      ->record<kp::OpAlgoDispatch>(algorithm)
      ->eval();

  auto timestamps = seq->getTimestamps();
  assert(timestamps.size() == 3);
  auto gmem2smem = (timestamps[2] - timestamps[1]);

  const float sec = gmem2smem * 1.0 / LOOP / 1e9f;
  fprintf(stdout, "***** gmem2smem bandwidth %0.3f GB/s \n",  SIZE_IN_BYTES / 1024. / 1024. / 1024. / sec);
}

int main() {
  kompute("smem_bandwidth.comp");
}
