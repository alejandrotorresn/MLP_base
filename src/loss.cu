//
// Created by zephyr on 8/20/25.
//
#include "loss.hpp"
#include <cublas_v2.h>
#include <cudnn.h>
#include <cassert>
#include <stdexcept>

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
    cublasSdot(cublas, static_cast<int>(size), y_diff, 1, y_diff,1, &dot);
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
void MSELoss::gradient_gpu(const float* y_pred, const float* y_true, float* grad_out, size_t size, void* handle) const {
    assert(y_pred && y_true && grad_out && handle);

    const auto cublas = static_cast<cublasHandle_t>(handle);

    // grad_out = y_pred - y_true
    float alpha = 1.0f;
    float beta = -1.0f;
    cublasSaxpy(cublas, static_cast<int>(size), &beta, y_true, 1, grad_out, 1);     // grad_out = -y_true
    cublasSaxpy(cublas, static_cast<int>(size), &alpha, y_pred, 1, grad_out, 1);        // grad_out += y_pred

    // grad_out *= 2 / size
    const float scale = 2.0f / static_cast<float>(size);
    cublasSscal(cublas, static_cast<int>(size), &scale, grad_out, 1);
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
    assert(y_pred && y_true && handle);

    const auto cudnn = static_cast<cudnnHandle_t>(handle);

    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1,
                             static_cast<int>(size));

    float* log_softmax;
    cudaMalloc(&log_softmax, size * sizeof(float));

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                        &alpha, desc, y_pred, &beta, desc, log_softmax);

    cublasHandle_t cublas;
    cublasCreate(&cublas);

    float loss = 0.0f;

    cublasSdot(cublas, static_cast<int>(size), y_true, 1, log_softmax, 1, &loss);

    cublasDestroy(cublas);
    cudaFree(log_softmax);
    cudnnDestroyTensorDescriptor(desc);

    return  -loss;
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
    assert(y_pred && y_true && grad_out && handle);

    const auto cudnn = static_cast<cudnnHandle_t>(handle);

    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             1, 1, 1, static_cast<int>(size));

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                        &alpha, desc, y_pred, &beta, desc, grad_out);

    cublasHandle_t cublas;
    cublasCreate(&cublas);

    const float neg = -1.0f;

    cublasSaxpy(cublas, static_cast<int>(size), &neg,
              y_true, 1, grad_out, 1);

    cublasDestroy(cublas);
    cudnnDestroyTensorDescriptor(desc);
}



