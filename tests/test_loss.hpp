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

bool compare_vectors_loss(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-4f);
void test_cross_entropy_loss(size_t size);
void test_mse_loss(size_t size);

#endif //MLP_BASE_TEST_LOSS_HPP