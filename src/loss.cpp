//
// Created by zephyr on 8/19/25.
//
#include "loss.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>


float MSELoss::compute(const std::vector<float>& y_pred,
                       const std::vector<float>& y_true) const {
    assert(y_pred.size() == y_true.size() && "MSELoss::compute - y_pred y y_true deben tener el mismo tamaño");
    float sum = 0.0f;
    for (size_t i = 0; i < y_pred.size(); ++i)
        sum += static_cast<float>(std::pow(y_pred[i] - y_true[i], 2));
    return sum / static_cast<float>(y_pred.size());
}

std::vector<float> MSELoss::gradient(const std::vector<float>& y_pred,
                                     const std::vector<float>& y_true) const {
    assert(y_pred.size() == y_true.size() && "MSELoss::gradient - y_pred y y_true deben tener el mismo tamaño");
    std::vector<float> grad(y_pred.size());
    for (size_t i = 0; i < y_pred.size(); ++i)
        grad[i] = 2.0f * (y_pred[i] - y_true[i]) / static_cast<float>(y_pred.size());
    return grad;
}

float CrossEntropyLoss::compute(const std::vector<float>& y_pred,
                                const std::vector<float>& y_true) const {
    assert(y_pred.size() == y_true.size() && "CrossEntropyLoss::compute - y_pred y y_true deben tener el mismo tamaño");
    float loss = 0.0f;
    for (size_t i = 0; i < y_pred.size(); ++i)
        loss -= y_true[i] * std::log(std::max(y_pred[i], 1e-8f));
    return loss;
}

std::vector<float> CrossEntropyLoss::gradient(const std::vector<float>& y_pred,
                                              const std::vector<float>& y_true) const {
    assert(y_pred.size() == y_true.size() && "CrossEntropyLoss::gradient - y_pred y y_true deben tener el mismo tamaño");
    std::vector<float> grad(y_pred.size());
    for (size_t i = 0; i < y_pred.size(); ++i)
        grad[i] = (y_pred[i] - y_true[i]) / std::max(y_pred[i], 1e-8f);
    return grad;
}
