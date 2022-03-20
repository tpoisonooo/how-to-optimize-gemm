#include "Shader.hpp"
#include "kompute/Kompute.hpp"
#include <iostream>
#include "types.h"

void kompute(const std::string &shader) {

  // 1. Create Kompute Manager with default settings (device 0, first queue and
  // no extensions)
  kp::Manager mgr;

  constexpr uint32_t SIZE = 512 * 1024 * 1024; // 512MB
  constexpr uint32_t count = SIZE/ sizeof(float);
  std::vector<float> data(SIZE / sizeof(float), 0);

  auto dtype = kp::Tensor::TensorDataTypes::eFloat;
  auto tensorIn = mgr.tensor(data.data(), count, sizeof(float), dtype);
  auto tensorOut = mgr.tensor(data.data(), count, sizeof(float), dtype);

  std::vector<std::shared_ptr<kp::Tensor>> params = {tensorIn, tensorout};

  // 3. Create algorithm based on shader (supports buffers & push/spec
  // constants)
  kp::Workgroup workgroup({1, 1, 1});

  auto algorithm =
      mgr.algorithm(params,
                    // See documentation shader section for compileSource
                    compileSource(shader), workgroup);

  // 4. Run operation synchronously using sequence
  auto seq = mgr.sequence(0, 3);
  seq->record<kp::OpTensorSyncDevice>(params)
      ->record<kp::OpAlgoDispatch>(algorithm) // Binds default push consts
      ->record<kp::OpTensorSyncLocal>(params)
      ->eval(); // Evaluates only last recorded operation

} // Manages / releases all CPU and GPU memory resources

int main() {

  // Define a raw string shader (or use the Kompute tools to compile to SPIRV /
  // C++ header files). This shader shows some of the main components including
  // constants, buffers, etc
  std::string shader = (R"(
        #version 450

        layout (local_size_x = 16) in;

        // The input tensors bind index is relative to index in parameter passed
        layout(set = 0, binding = 0) buffer buf_in_a { float in_a[]; };
        layout(set = 0, binding = 2) buffer buf_out_a { uint out_a[]; };

        // Kompute supports push constants updated on dispatch
        layout(push_constant) uniform PushConstants {
            float val;
        } push_const;

        // Kompute also supports spec constants on initalization
        layout(constant_id = 0) const float const_one = 0;

        void main() {
            uint index = gl_GlobalInvocationID.x;
            out_a[index] += uint( in_a[index] * in_b[index] );
            out_b[index] += uint( const_one * push_const.val );
        }
    )");

  // Run the function declared above with our raw string shader
  kompute(shader);
}
