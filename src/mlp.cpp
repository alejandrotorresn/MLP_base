//
// Created by vermeer on 8/12/25.
//
#include "mlp.hpp"
#include <iostream>
#include <unistd.h>

MLP::MLP() = default;

void MLP::add_layer(Layer* layer) {
    layers.push_back(layer);
}

std::vector<float> MLP::forward(const std::vector<float>& input, void* handle) {
    std::vector<float> x = input;
    for (auto layer : layers) {
        std::string device = profiler.getDevice(layer->name, "forward");
        std::cout << "Forward en capa " << layer->name << " usando " << device << std::endl;
        x = (device == "GPU") ? layer->forward_gpu(x, handle) : layer->forward_cpu(x);
    }
    return x;
}

std::vector<float> MLP::backward(const std::vector<float>& grad, void* handle) {
    std::vector<float> g = grad;
    for (int i = layers.size() - 1; i >= 0; i--) {
        std::string device = profiler.getDevice(layers[i]->name, "backward");
        std::cout << "backward en capa " << layers[i]->name << " usando " << device << std::endl;
        g = (device == "GPU") ? layers[i]->backward_gpu(g, handle) : layers[i]->backward_cpu(g);
    }
    return g;
}