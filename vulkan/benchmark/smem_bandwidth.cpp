#include "Shader.hpp"
#include "kompute/Kompute.hpp"
#include <iostream>
#include <cassert>
#include "types.h"

void kompute(const std::string &shader) {
  kp::Manager mgr;

  constexpr uint32_t SIZE_IN_BYTES = 32768;
  constexpr uint32_t BLOCK = 256;
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
  fprintf(stdout, "***** %s bandwidth %0.3f GB/s \n", shader.c_str(),SIZE_IN_BYTES / 1024. / 1024. / 1024. / sec);
}

int main() {
  // smem_bandwidth.comp bandwidth 10.665 GB/s
  kompute("smem_bandwidth.comp");
  // smem_bandwidth1.comp bandwidth 18.663 GB/s
  kompute("smem_bandwidth1.comp");
// sampler_bandwidth.comp bandwidth 35.502 GB/s
  kompute("sampler_bandwidth.comp");
}
