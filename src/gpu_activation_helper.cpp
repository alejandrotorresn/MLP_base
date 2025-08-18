//
// Created by zephyr on 8/18/25.
//

#include "gpu_activation_helper.hpp"
#include "gpu_error_checking.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

GPUActivationHelper::GPUActivationHelper(const int size, cudnnActivationDescriptor_t act_desc, cudnnTensorDescriptor_t tensor_desc)
    : input_size(size), act_desc(act_desc), tensor_desc(tensor_desc),
      d_input(nullptr), d_output(nullptr), d_grad(nullptr), d_input_grad(nullptr) {
    allocate();
}

GPUActivationHelper::~GPUActivationHelper() {
    release();
}

void GPUActivationHelper::allocate() {
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_grad, input_size * sizeof(float)));
}

void GPUActivationHelper::release() const {
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_input_grad));
}

std::vector<float> GPUActivationHelper::forward(const std::vector<float>& input, cudnnHandle_t handle) const {
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    constexpr float alpha = 1.0;
    constexpr float beta = 0.0;

    CUDNN_CHECK(
        cudnnActivationForward(handle, act_desc,
            &alpha, tensor_desc, d_input,
            &beta, tensor_desc, d_output)
        );

    std::vector<float> output(input_size);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, input.size() * sizeof(float), cudaMemcpyDeviceToHost));

    return output;
}

std::vector<float> GPUActivationHelper::backward(const std::vector<float>& input, const std::vector<float>& grad, cudnnHandle_t handle) const {
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad, grad.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));

    constexpr float alpha = 1.0;
    constexpr float beta = 0.0;

    CUDNN_CHECK(
        cudnnActivationForward(handle, act_desc,
            &alpha, tensor_desc, d_input,
            &beta, tensor_desc, d_grad)
        );

    CUDNN_CHECK(
        cudnnActivationBackward(handle, act_desc,
            &alpha, tensor_desc, d_output,
            tensor_desc, d_grad,
            tensor_desc, d_input,
            &beta, tensor_desc, d_input_grad)
        );

    std::vector<float> grad_input(input_size);
    CUDA_CHECK(cudaMemcpy(grad_input.data(), d_input_grad, input.size() * sizeof(float), cudaMemcpyDeviceToHost));

    return grad_input;
}






