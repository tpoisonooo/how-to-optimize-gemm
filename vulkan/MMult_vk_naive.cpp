#define SPDLOG_ACTIVE_LEVEL 5
#include "kompute/Kompute.hpp"
#include "Shader.hpp"

float kompute(const std::string& shader_template, uint32_t m, uint32_t k, uint32_t n, float* a, float *b, float *c) {
    // build real .comp shader
    
    constexpr int local_size = 32;
    char shader[2048] = {0};
    sprintf(shader, shader_template.c_str(), local_size, local_size);
    // fprintf(stdout, "%s\n", shader);

    kp::Manager mgr;

    // Create and initialise Kompute Tensors through manager
    auto dtype = kp::Tensor::TensorDataTypes::eFloat;
    auto tensorInA = mgr.tensor(a, m*k, sizeof(float), dtype);
    auto tensorInB = mgr.tensor(b, k*n, sizeof(float), dtype);
    auto tensorInC = mgr.tensor(c, m*n, sizeof(float), dtype);

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

    auto ptr = tensorInC->data<float>();
    for (int i = 0; i < 4; ++i) {
        fprintf(stdout, "%d = %f \n", i, ptr[i]);
    }
    memcpy(c, tensorInC->data<float>(), m * n * sizeof(float));
    return 100;
}

float MY_MMult(int m, int n, int k, float * a, float * b, float * c) {
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

    return kompute(shader_template, static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k), a, b, c);
}