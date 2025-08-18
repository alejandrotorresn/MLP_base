//
// Created by zephyr on 8/18/25.
//

#ifndef MLP_BASE_GPU_ERROR_CHECKING_HPP
#define MLP_BASE_GPU_ERROR_CHECKING_HPP

#pragma once
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdexcept>
#include <string>

/*
 * @brief Verifica el estado de una llamada CUDa y lanza excepcion si falla.
 * @param code Codigo de error devuelto por una funcion CUDA
 * @param file Archivo fuente donde se invoca la macro
 * @param line Linea de codifo donde se invoca la macro
 * @throws std::runtime_error con mensaje detallado si hay error.
 */
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(const cudaError_t code, const char* file, const int line) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string("CUDA Error: ")
            + cudaGetErrorString(code) + " at" + file + ":"
            + std::to_string(line));
}

/*
 * @brief Verifica el estado de una llamanada cuDNN y lanza una exepcion si falla.
 * @param status codigo de estado devuelto por una funcion cuDNN.
 * @param file Archivo fuente donde se invoco la macro.
 * @param line Linea de codigo donde se invoco la macro
 * @throws std::runtime_error con mensaje detallado si hay error.
 */
#define CUDNN_CHECK(status) { cudnnAssert((status), __FILE__, __LINE__); }
inline void cudnnAssert(const cudnnStatus_t status, const char* file, const int line) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuDNN Error: ")
            + cudnnGetErrorString(status) + " at " + file + ":"
            + std::to_string(line));
}

#endif //MLP_BASE_GPU_ERROR_CHECKING_HPP