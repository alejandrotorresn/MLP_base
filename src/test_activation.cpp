//
// Created by zephyr on 8/18/25.
//
#include "activation.hpp"
#include "test_activation.hpp"
#include <iostream>
#include <cmath>
#include <chrono>

bool compare_vectors(const std::vector<float>& a, const std::vector<float>& b, const float tol = 1e-5) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tol)
            return false;
    }
    return true;
}

void test_activation_consistency(const ActType type, const int size) {
    Activation act("test", size, type);
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    std::vector<float> input(size);
    std::vector<float> grad(size);
    for (int i = 0; i < size; ++i) {
        input[i] = 0.1f * static_cast<float>(i) - 1.0f;
        grad[i] = 1.0f;
    }

    const auto out_cpu = act.forward_cpu(input);
    const auto out_gpu = act.forward_gpu(input, handle);

    const auto back_cpu = act.backward_cpu(grad);
    const auto back_gpu = act.backward_gpu(grad, handle);

    std::cout << "Forward match: " << (compare_vectors(out_cpu, out_gpu) ? "OK" : "FAIL") << "\n";
    std::cout << "Backward match: " << (compare_vectors(back_cpu, back_gpu) ? "OK" : "FAIL") << "\n";

    cudnnDestroy(handle);
}