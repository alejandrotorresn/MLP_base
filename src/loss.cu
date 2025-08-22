//
// Created by zephyr on 8/20/25.
//
#include "loss.hpp"
#include <cublas_v2.h>
#include <cassert>
#include <stdexcept>
#include <cfloat>
#include <cmath>

/*
 * @brief Calcula la perdida MSE en GPU usando cuBLAS.
 *
 * Realiza la operacion: MSE = ||y_pred - y_true||² / N
 * sin transferencias entre host y dispositivo
 *
 * @param y_pred Puntero a predicciones en GPU.
 * @param y_true Puntero a etiquetas verdaderas en GPU
 * @param size Numero de elementos
 * @param handle Puntero a cublasHandle_t
 * @return Perdida MSE como flotante
 */
float MSELoss::compute_gpu(const float* y_pred,
                           const float* y_true,
                           const size_t size,
                           void* handle) const {
    assert(y_pred && y_true && handle);

    const auto cublas = static_cast<cublasHandle_t>(handle);

    float* y_diff;
    cudaMalloc(&y_diff, size * sizeof(float));

    constexpr float alpha = 1.0f;
    constexpr float beta = -1.0f;

    // y_diff = y_pred - y_true
    cublasScopy(cublas, static_cast<int>(size), y_pred, 1, y_diff, 1);
    cublasSaxpy(cublas, static_cast<int>(size), &beta, y_true, 1, y_diff, 1);

    // MSE = dot(y_diff, y_diff) / size
    float dot = 0.0f;
    cublasSdot(cublas, static_cast<int>(size), y_diff, 1, y_diff, 1, &dot);
    cudaFree(y_diff);

    return dot / static_cast<float>(size);
}

/*
 * @brief Calcula el gradiente de MSE en GPU usando cuBLAS.
 *
 * Realiza la operacion: grad = 2 * (y_pred - y_true) / N
 * sin transferencias entre host y dispositivo.
 *
 * @param y_pred Puntero a predicciones en GPU.
 * @param y_true Puntero a etiquetas verdaderas en GPU.
 * @param grad_out Puntero de salida para el gradiente.
 * @param size Numero de elementos
 * @param handle Puntero a cublasHandle_t
 */
void MSELoss::gradient_gpu(const float* y_pred,
                           const float* y_true,
                           float* grad_out,
                           size_t size,
                           void* handle) const {
    assert(y_pred && y_true && grad_out && handle);

    const auto cublas = static_cast<cublasHandle_t>(handle);

    float alpha = 1.0f;
    float beta = -1.0f;

    // grad_out = y_pred - y_true
    cublasSaxpy(cublas, static_cast<int>(size), &beta, y_true, 1, grad_out, 1);
    cublasSaxpy(cublas, static_cast<int>(size), &alpha, y_pred, 1, grad_out, 1);

    // grad_out *= 2 / size
    const float scale = 2.0f / static_cast<float>(size);
    cublasSscal(cublas, static_cast<int>(size), &scale, grad_out, 1);
}

//
// CrossEntropyLoss - GPU (CUDA puro)
//

__global__ void log_softmax_kernel(const float* logits, float* output,
                                   float max_val, float log_sum_exp, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        output[idx] = logits[idx] - max_val - log_sum_exp;
}

__global__ void cross_entropy_grad_kernel(const float* logits, const float* y_true,
                                          float* grad_out, float max_val, float sum_exp, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float softmax = expf(logits[idx] - max_val) / sum_exp;
        grad_out[idx] = softmax - y_true[idx];
    }
}

/*
 * @brief Calcula la perdidad de entropia cruzada en GPU usando cuDNN.
 *
 * Aplica softmax logaritmico y luego realiza la reduccion:
 * CE = -∑(y_true * log_softmax(y_pred))
 *
 * @param y_pred Puntero a logits en GPU
 * @param y_true Puntero a etiquetas one-hot en GPU
 * @param size Numero de clases
 * @param handle Puntero a cudnnHandle_t
 * @return Perdida CrossEntropy como flotante.
 */
float CrossEntropyLoss::compute_gpu(const float* y_pred,
                                    const float* y_true,
                                    size_t size,
                                    void* handle) const {
    assert(y_pred && y_true);

    // 1. Calcular max(logits)
    float* d_max;
    cudaMalloc(&d_max, sizeof(float));
    cudaMemcpy(d_max, y_pred, sizeof(float), cudaMemcpyDeviceToDevice);

    float max_val = -FLT_MAX;
    for (size_t i = 0; i < size; ++i) {
        float val;
        cudaMemcpy(&val, y_pred + i, sizeof(float), cudaMemcpyDeviceToHost);
        if (val > max_val) max_val = val;
    }

    // 2. Calcular sum(exp(logits - max))
    float sum_exp = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float val;
        cudaMemcpy(&val, y_pred + i, sizeof(float), cudaMemcpyDeviceToHost);
        sum_exp += expf(val - max_val);
    }
    float log_sum_exp = logf(sum_exp);

    // 3. Calcular log_softmax
    float* log_softmax;
    cudaMalloc(&log_softmax, size * sizeof(float));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    log_softmax_kernel<<<blocks, threads>>>(y_pred, log_softmax, max_val, log_sum_exp, size);

    // 4. Calcular dot(y_true, log_softmax)
    float loss = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float log_val, true_val;
        cudaMemcpy(&log_val, log_softmax + i, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&true_val, y_true + i, sizeof(float), cudaMemcpyDeviceToHost);
        loss += true_val * log_val;
    }

    cudaFree(log_softmax);
    cudaFree(d_max);

    return -loss;
}

/*
 * @brief Calcula el gradiente de entropia cruzada en GPU usando cuDNN.
 *
 * Realiza la operacion: grad = softmax(y_pred) - y_true
 *
 * @param y_pred Puntero a logits en GPU
 * @param y_true Puntero a etiquetas one-hot en GPU
 * @param grad_out Puntero de salida para el gradiente
 * @param size Numero de clases
 * @param handle Puntero a cudnnHandle_t
 */
void CrossEntropyLoss::gradient_gpu(const float* y_pred,
                                    const float* y_true,
                                    float* grad_out,
                                    size_t size,
                                    void* handle) const {
    assert(y_pred && y_true && grad_out);

    // 1. Calcular max(logits)
    float max_val = -FLT_MAX;
    for (size_t i = 0; i < size; ++i) {
        float val;
        cudaMemcpy(&val, y_pred + i, sizeof(float), cudaMemcpyDeviceToHost);
        if (val > max_val) max_val = val;
    }

    // 2. Calcular sum(exp(logits - max))
    float sum_exp = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float val;
        cudaMemcpy(&val, y_pred + i, sizeof(float), cudaMemcpyDeviceToHost);
        sum_exp += expf(val - max_val);
    }

    // 3. Calcular grad = softmax - y_true
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cross_entropy_grad_kernel<<<blocks, threads>>>(y_pred, y_true, grad_out, max_val, sum_exp, size);
}



