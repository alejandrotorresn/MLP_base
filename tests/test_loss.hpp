//
// Created by zephyr on 8/21/25.
//

#ifndef MLP_BASE_TEST_LOSS_HPP
#define MLP_BASE_TEST_LOSS_HPP

#include <vector>
#include <cudnn.h>
#include <cublas_v2.h>

#include "activation.hpp"
#include "loss.hpp"

/*
 * @brief Ejecuta pruebas unitarias comparativas entre CPU y GPU para MSELoss
 * Valudad consistencia numerica entre rutas y correcta ejecucion en GPU
 */
void test_mse_loss_consistency(cublasHandle_t cublas);

/*
 * @brief ejecuta pruenas unitarias para CrossEntropyLoss en GPU
 * Valida calculo de perdida y gradiente usando cuDNN
 */
void test_cross_entropy_gpu(cudnnHandle_t cudnn, cublasHandle_t cublas);

#endif //MLP_BASE_TEST_LOSS_HPP