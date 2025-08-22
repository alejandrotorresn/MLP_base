//
// Created by zephyr on 8/21/25.
//
#include <iostream>
#include <cmath>
#include <cassert>
#include "test_loss.hpp"

#include <iomanip>

/*
 * @brief Compara dos vectores con tolerancia absoluta
 */
bool vectors_close(const std::vector<float>& a, const std::vector<float>& b,
                  float tol = 1e-5) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

/*
 * @brief Compara dos floatantes con tolerancia
 */
bool floats_close(const float a, const float b, const float tol = 1e-5) {
    return std::fabs(a - b) < tol;
}


void test_mse_loss_consistency(cublasHandle_t cublas) {
    std::cout << "[TEST] MSELoss CPU vs GPU\n";

    std::vector<float> y_pred = {0.1f, 0.6f, 0.3f};
    std::vector<float> y_true = {0.0f, 1.0f, 0.0f};
    size_t size = y_pred.size();

    MSELoss loss;

    // CPU
    float cpu_loss = loss.compute(y_pred, y_true);
    std::vector<float> cpu_grad = loss.gradient(y_pred, y_true);

    // GPU
    float* d_pred;
    float* d_true;
    float* d_grad;

    cudaMalloc(reinterpret_cast<void**>(&d_pred), size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_true), size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_grad), size * sizeof(float));


    cudaMemcpy(d_pred, y_pred.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_true, y_true.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    float gpu_loss = loss.compute_gpu(d_pred, d_true, size, cublas);
    loss.gradient_gpu(d_pred, d_true, d_grad, size, cublas);

    std::vector<float> gpu_grad(size);
    cudaMemcpy(gpu_grad.data(), d_grad, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_grad);
    cudaFree(d_true);
    cudaFree(d_pred);

    std::cout << "CPU Loss: " << cpu_loss << "\n";
    std::cout << "GPU Loss: " << gpu_loss << "\n";
    assert(floats_close(cpu_loss, gpu_loss));

    std::cout << "CPU Gradient: ";
    for (const float g : cpu_grad) std::cout << std::fixed << std::setprecision(6) << g << " ";
    std::cout << "\nGPU Gradient: ";
    for (const float g : gpu_grad) std::cout << std::fixed << std::setprecision(6) << g << " ";
    std::cout << "\n";

    assert(vectors_close(cpu_grad, gpu_grad));

    std::cout << "MSELoss CPU/GPU outputs match.\n\n";
}

void test_cross_entropy_gpu(cudnnHandle_t cudnn, cublasHandle_t cublas) {
    std::cout << "[TEST] CrossEntropyLoss GPU\n";

    std::vector<float> y_pred = {0.1f, 0.6f, 0.3f}; // logits
    std::vector<float> y_true = {0.0f, 1.0f, 0.0f}; // one-hot
    size_t size = y_pred.size();

    CrossEntropyLoss loss;

    float* d_pred;
    float* d_true;
    float* d_grad;

    cudaMalloc(reinterpret_cast<void**>(&d_pred), size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_true), size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_grad), size * sizeof(float));

    cudaMemcpy(d_pred, y_pred.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_true, y_true.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    float gpu_loss = loss.compute_gpu(d_pred, d_true, size, cudnn);
    loss.gradient_gpu(d_pred, d_true, d_grad, size, cudnn);

    std::vector<float> gpu_grad(size);
    cudaMemcpy(gpu_grad.data(), d_grad, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_pred);
    cudaFree(d_true);
    cudaFree(d_grad);

    std::cout << "GPU Loss: " << gpu_loss << "\n";
    std::cout << "GPU Gradient: ";
    for (const float g : gpu_grad) std::cout << g << " ";
    std::cout << "\n";

    std::cout << "CrossEntropyLoss GPU outputs computed.\n\n";
}