#include "kompute/Kompute.hpp"
#include "Shader.hpp"
#include "types.h"

float kompute(const std::string& shader_template, unsigned int m, unsigned int k, unsigned int n) {
    constexpr int local_size = 32;
    char shader[2048] = {0};
    sprintf(shader, shader_template.c_str(), local_size, local_size);
    fprintf(stdout, "%s\n", shader);

    kp::Manager mgr;

    auto dtype = kp::Tensor::TensorDataTypes::eFloat;

    // define Input & Output
    AlignVector matrixA(m * k, 1.f);
    AlignVector matrixB(k * n, 1.f);
    AlignVector matrixC(m * n);

    // Create and initialise Kompute Tensors through manager
    auto tensorInA = mgr.tensor(matrixA.data(), matrixA.size(), sizeof(float), dtype);
    auto tensorInB = mgr.tensor(matrixB.data(), matrixB.size(), sizeof(float), dtype);
    auto tensorInC = mgr.tensor(matrixC.data(), matrixC.size(), sizeof(float), dtype);

    std::vector<std::shared_ptr<kp::Tensor>> params = {tensorInA, tensorInB, tensorInC};

    // Create algorithm based on shader (supports buffers & push/spec constants)
    kp::Workgroup workgroup({m / local_size, n / local_size, 1});

    auto algorithm = mgr.algorithm(params,
                                   compileSource(shader),
                                   workgroup,
                                   { k * 1.f});

    mgr.sequence()
        ->record<kp::OpTensorSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpTensorSyncLocal>(params)
        ->eval();

    float* result = tensorInC->data<float>();
    constexpr int LEN = 4;
    for (int i = 0; i < LEN; ++i) {
        fprintf(stdout, "tensorInC raw %d: %f\n", i, result[i]);
    }
}

    // diff = compare_matrices(m, n, cold, ldc, cref, ldc);
    // if (diff > 0.5f || diff < -0.5f) {
    //   printf("diff too big !\n");
    //   exit(-1);
    // }

int main() {
    std::string shader_template = (R"(
        #version 450

        layout (local_size_x = %d, local_size_y = %d) in;

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
    constexpr int SIZE = 256;
    auto gflops = kompute(shader_template, SIZE, SIZE, SIZE);
}