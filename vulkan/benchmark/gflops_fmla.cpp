#include "../Shader.hpp"
#include "../kompute/Kompute.hpp"
#include <iostream>
#include <cassert>
#include "types.h"

// gflops_fmla: 184.358588 
constexpr uint32_t COUNT = 16384;
constexpr uint32_t BLOCK = 256;

constexpr float LOOP = 3000000.0;

uint64_t kompute(const std::string &filename) {

  kp::Manager mgr;
  kp::Workgroup workgroup({COUNT / BLOCK, 1, 1});
  
  AlignVector data1(COUNT, 1.0f);
  AlignVector data2(COUNT, 0.0f);
  AlignVector data3(COUNT, 0.0f);

  auto dtype = kp::Tensor::TensorDataTypes::eFloat;
  auto tensorIn1 = mgr.tensor(data1.data(), data1.size(), sizeof(float), dtype);
  auto tensorIn2 = mgr.tensor(data2.data(), data2.size(), sizeof(float), dtype);
  auto tensorOut = mgr.tensor(data3.data(), data3.size(), sizeof(float), dtype);

  std::vector<std::shared_ptr<kp::Tensor>> params = {tensorIn1, tensorIn2, tensorOut};
  auto algorithm = mgr.algorithm(params, compileFile(filename), workgroup, {LOOP});
  auto seq = mgr.sequence(0, 3);

  seq->record<kp::OpTensorSyncDevice>(params)
      ->record<kp::OpAlgoDispatch>(algorithm)
      ->record<kp::OpTensorSyncLocal>(params)
      ->eval();

  float* pResult = static_cast<float*>(tensorOut->rawData());
  for (int  i =0;  i< 10; ++i) {
    fprintf(stdout, "%f ", pResult[i]);
  }

  auto timestamps = seq->getTimestamps();
  return (timestamps[3] - timestamps[0]);
}

int main() {
  auto rw_compute_cost = kompute("gflops_fmla_1.comp");
  auto rw_cost = kompute("gflops_fmla_2.comp");
  fprintf(stdout, "gflops_fmla: %lf \n",  LOOP * COUNT * 10.0/ (rw_compute_cost - rw_cost));

  return 0;
}
