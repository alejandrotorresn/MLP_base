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

}



