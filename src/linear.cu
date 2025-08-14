//
// Created by zephyr on 8/12/25.
//
#include "linear.h"
#include <mkl.h>
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*
 * @brief Constructor: inicializa pesos y sesgos con valores constantes
 */
Linear::Linear(const std::string &name, int in, int out)
    : Layer(name, in, out),
    weights(out * in, 0.01f),
    bias(out, 0.0f),
    last_input(in, 0.0f) {}

/*
 * @brief Forward en CPU usando MKL: y = Wx + b
 */
std::vector<float> Linear::forward_cpu(const std::vector<float> &x) {
    if (x.size() != input_size)
        throw std::invalid_argument("Dimension de entrada incorrecta en forward_cpu");

    last_input = x;
    std::vector<float> output(output_size, 0.0f);

    // y = Wx + b usando MKL: y = alpha*A*x + beta*y
    const float alpha = 1.0f;
    const float beta = 1.0f;

    // Copiar bias en output
    std::copy(bias.begin(), bias.end(), output.begin());

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                output_size, input_size,
                alpha,
                weights.data(), input_size,
                x.data(), 1,
                beta,
                output.data(), 1);

    return output;
}

/*
 * @brief Backward en CPU::
 */
std::vector<float> Linear::backward_cpu(const std::vector<float>& grad) {
    if (grad.size() != output_size)
        throw std::invalid_argument("Dimensión de gradiente incorrecta en backward_cpu");

    std::vector<float> grad_input(input_size, 0.0f);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cblas_sgemv(CblasRowMajor, CblasTrans,
                output_size, input_size,
                alpha,
                weights.data(), input_size,
                grad.data(), 1,
                beta,
                grad_input.data(), 1);

    return grad_input;
}

/*
 * @brief Forward en GPU usando cuBLAS: y = Wx + b
 */
std::vector<float> Linear::forward_gpu(const std::vector<float>& x, void* handle_void) {
    if (x.size() != input_size)
        throw std::invalid_argument("Dimensión de entrada incorrecta en forward_gpu");

    cublasHandle_t handle = static_cast<cublasHandle_t>(handle_void);

    last_input = x;
    std::vector<float> output(output_size, 0.0f);

    // Reservar memoria en GPU
    float* d_weights;
    float* d_input;
    float* d_output;

    cudaMalloc(&d_weights, weights.size() * sizeof(float));
    cudaMalloc(&d_input, x.size() * sizeof(float));
    cudaMalloc(&d_output, output.size() * sizeof(float));

    cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice); // y = b

    const float alpha = 1.0f;
    const float beta = 1.0f;

    // y = Wx + b → cuBLAS usa column-major, así que usamos Wᵗ
    cublasSgemv(handle,
                CUBLAS_OP_T,
                input_size, output_size,
                &alpha,
                d_weights, input_size,
                d_input, 1,
                &beta,
                d_output, 1);

    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

/*
 * @brief Backward en GPU usando cuBLAS: grad_input = Wᵗ · grad
 */
std::vector<float> Linear::backward_gpu(const std::vector<float>& grad, void* handle_void) {
    if (grad.size() != output_size)
        throw std::invalid_argument("Dimensión de gradiente incorrecta en backward_gpu");

    cublasHandle_t handle = static_cast<cublasHandle_t>(handle_void);

    std::vector<float> grad_input(input_size, 0.0f);

    // Reservar memoria en GPU
    float* d_weights;
    float* d_grad;
    float* d_grad_input;

    cudaMalloc(&d_weights, weights.size() * sizeof(float));
    cudaMalloc(&d_grad, grad.size() * sizeof(float));
    cudaMalloc(&d_grad_input, grad_input.size() * sizeof(float));

    cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, grad.data(), grad.size() * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // grad_input = W · grad
    cublasSgemv(handle,
                CUBLAS_OP_N,
                input_size, output_size,
                &alpha,
                d_weights, input_size,
                d_grad, 1,
                &beta,
                d_grad_input, 1);

    cudaMemcpy(grad_input.data(), d_grad_input, grad_input.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_grad);
    cudaFree(d_grad_input);

    return grad_input;
}