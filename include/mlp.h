//
// Created by vermeer on 8/12/25.
//

#ifndef MLP_H
#define MLP_H

#pragma once
#include "layer.h"
#include "profiler.h"
#include <vector>

/*
 * @brief Clase que representa un perceptron multicapa (MLP).
 * Gestiona las capas y ejecuta forward/backward segun profiler.
 */
class MLP {
    public:
    std::vector<Layer*> layers;     // Lista de capas del modelo
    Profiler profiler;              // Perfilador para decidir CPU/GPU por fase

    MLP();

    /*
     * @brief Agrega una capa al modelo
     */
    void add_layer(Layer* layer);

    /*
     * @brief Ejecuta forward sobre todas las capas
     */
    std::vector<float> forward(const std::vector<float>& input, void* handle);

    /*
     * @brief Ejecuta backward sobre todas las capas
     */
    std::vector<float> backward(const std::vector<float>& grad, void* handle);

    ~MLP() = default;
};

#endif //MLP_H
