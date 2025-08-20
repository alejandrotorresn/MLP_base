//
// Created by zephyr on 8/19/25.
//

#include "optimizer.hpp"

SGD::SGD(const float lr) : learning_rate(lr) {}

void SGD::update(std::vector<float>& weights,
                 std::vector<float>& grad) {
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] -= learning_rate * grad[i];
}