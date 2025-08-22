#include "test_loss.hpp"

#include "loss.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

constexpr float EPSILON = 1e-4f;

bool compare_vectors_loss(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

void test_cross_entropy_loss(size_t size) {
    std::vector<float> logits(size);
    std::vector<float> labels(size);

    // Simulación de logits y etiquetas one-hot
    for (size_t i = 0; i < size; ++i) {
        logits[i] = static_cast<float>(std::rand()) / RAND_MAX * 4.0f - 2.0f; // [-2, 2]
        labels[i] = 0.0f;
    }
    labels[size / 2] = 1.0f; // Etiqueta one-hot

    CrossEntropyLoss loss;

    // CPU
    float cpu_loss = loss.compute(logits, labels);
    std::vector<float> cpu_grad = loss.gradient(logits, labels);

    // GPU
    float* d_logits;
    float* d_labels;
    float* d_grad;
    cudaMalloc(&d_logits, size * sizeof(float));
    cudaMalloc(&d_labels, size * sizeof(float));
    cudaMalloc(&d_grad, size * sizeof(float));

    cudaMemcpy(d_logits, logits.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    float gpu_loss = loss.compute_gpu(d_logits, d_labels, size, nullptr);
    loss.gradient_gpu(d_logits, d_labels, d_grad, size, nullptr);

    std::vector<float> gpu_grad(size);
    cudaMemcpy(gpu_grad.data(), d_grad, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_logits);
    cudaFree(d_labels);
    cudaFree(d_grad);

    std::cout << "[CrossEntropyLoss] CPU: " << cpu_loss << " | GPU: " << gpu_loss << "\n";
    std::cout << "Gradiente consistente: " << (compare_vectors_loss(cpu_grad, gpu_grad) ? "✅" : "❌") << "\n";
}

void test_mse_loss(size_t size) {
    std::vector<float> y_pred(size);
    std::vector<float> y_true(size);

    for (size_t i = 0; i < size; ++i) {
        y_pred[i] = static_cast<float>(std::rand()) / RAND_MAX;
        y_true[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    MSELoss loss;

    // CPU
    float cpu_loss = loss.compute(y_pred, y_true);
    std::vector<float> cpu_grad = loss.gradient(y_pred, y_true);

    // GPU
    float* d_pred;
    float* d_true;
    float* d_grad;
    cudaMalloc(&d_pred, size * sizeof(float));
    cudaMalloc(&d_true, size * sizeof(float));
    cudaMalloc(&d_grad, size * sizeof(float));

    cudaMemcpy(d_pred, y_pred.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_true, y_true.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float gpu_loss = loss.compute_gpu(d_pred, d_true, size, handle);
    loss.gradient_gpu(d_pred, d_true, d_grad, size, handle);

    std::vector<float> gpu_grad(size);
    cudaMemcpy(gpu_grad.data(), d_grad, size * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(d_pred);
    cudaFree(d_true);
    cudaFree(d_grad);

    std::cout << "[MSELoss] CPU: " << cpu_loss << " | GPU: " << gpu_loss << "\n";
    std::cout << "Gradiente consistente: " << (compare_vectors_loss(cpu_grad, gpu_grad) ? "✅" : "❌") << "\n";
}