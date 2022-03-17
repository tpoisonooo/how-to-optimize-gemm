#include "kompute/Kompute.hpp"
#include "Shader.hpp"
#include "types.h"

std::vector<float> kompute(const std::string& shader, unsigned int m, unsigned int k, unsigned int n) {

    // 1. Create Kompute Manager with default settings (device 0, first queue and no extensions)
    kp::Manager mgr;

    auto dtype = kp::Tensor::TensorDataTypes::eFloat;

    // define Input & Output
    AlignVector matrixA(m * k, 2.f);
    // AlignVector matrixB(k * n, 2.f);
    // std::vector<float> matrixC;
    // matrixC.resize(k * n);

    // std::vector<float> matrixC(m*n, 3.f);

    // Create and initialise Kompute Tensors through manager
    // auto tensorInA = mgr.tensor({2., 2., 2., 2.});
    auto tensorInA = mgr.tensor(matrixA.data(), matrixA.size(), sizeof(float), dtype);
    auto tensorInB = mgr.tensor({2., 2., 2., 2.});
    auto tensorInC = mgr.tensorT<float>({3., 3., 3., 3.});

    std::vector<std::shared_ptr<kp::Tensor>> params = {tensorInA, tensorInB, tensorInC};

    // Create algorithm based on shader (supports buffers & push/spec constants)
    constexpr int local_size = 2;
    // kp::Workgroup workgroup({m / local_size, n / local_size});
    kp::Workgroup workgroup({1, 1, 1});


    auto algorithm = mgr.algorithm(params,
                                   // See documentation shader section for compileSource
                                   compileSource(shader),
                                   workgroup,
                                   {2.});

    // Run operation synchronously using sequence
    mgr.sequence()
        ->record<kp::OpTensorSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->eval();

    auto sq = mgr.sequence();
    sq->record<kp::OpTensorSyncLocal>(params)
        ->eval();

    return tensorInC->vector();
}

int main() {

    // Define a raw string shader (or use the Kompute tools to compile to SPIRV / C++ header
    // files). This shader shows some of the main components including constants, buffers, etc
    std::string shader = (R"(
        #version 450

        layout (local_size_x = 2, local_size_y = 2) in;

        layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 { float in_tensor_1[]; };
        layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 { float in_tensor_2[]; };
        layout (set = 0, binding = 2) writeonly buffer buf_out_tensor { float out_tensor[]; };

        layout (constant_id = 0) const float tensor_size_f = 0;


        void main()
        {
            uint globalRow = gl_GlobalInvocationID.x;
            uint globalCol = gl_GlobalInvocationID.y;
            uint tensor_size = uint(tensor_size_f);
            
            float acc = 0.0;
            for(uint k = 0u; k < tensor_size; k++) {
                acc += in_tensor_1[(k * tensor_size) + globalRow] * in_tensor_2[(globalCol * tensor_size) + k];
            }
            out_tensor[(globalCol * tensor_size) + globalRow] = acc;
        }
    )");

    // Run the function declared above with our raw string shader
    auto result = kompute(shader, 2, 2, 2);
    constexpr int LEN = 4;
    for (int i = 0; i < LEN; ++i) {
        fprintf(stdout, "%d: %f\n", i, result[i]);
    }
}