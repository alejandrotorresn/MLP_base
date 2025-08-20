//
// Created by vermeer on 8/12/25.
//
#include "mlp.hpp"
#include <iostream>

MLP::MLP() = default;

void MLP::add_layer(Layer* layer) {
    layers.push_back(layer);
}

std::vector<float> MLP::forward(const std::vector<float>& input, void* handle, const bool use_gpu) const {
    std::vector<float> x = input;
    for (auto* layer : layers) {
        std::cout << "Forward en capa " << layer->name << " usando " << (use_gpu ? "GPU" : "CPU") << "\n";
        x = use_gpu ? layer->forward_gpu(x, handle) : layer->forward_cpu(x);
    }
    return x;
}

std::vector<float> MLP::backward(const std::vector<float>& grad, void* handle, const bool use_gpu) const {
    std::vector<float> g = grad;
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
        std::cout << "backward en capa " << layers[i]->name << " usando " << (use_gpu ? "GPU" : "CPU") << "\n";
        g = use_gpu ? layers[i]->backward_gpu(g, handle) : layers[i]->backward_cpu(g);
    }
    return g;
}

void MLP::update_weights(Optimizer& optimizer) {
    for (auto layer : layers) {
        layer->update_weights(optimizer);
    }
}
