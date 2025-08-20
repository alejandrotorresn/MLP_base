//
// Created by vermeer on 8/12/25.
//

#ifndef MLP_H
#define MLP_H

#pragma once
#include "layer.hpp"
#include "profiler.hpp"
#include <vector>

#include "optimizer.hpp"

/*
 * @brief Clase que representa un perceptron multicapa (MLP).
 * Ejecuta forward/backward y actualiza pesos en todas las capas.
 */
class MLP {
public:
    std::vector<Layer*> layers;     // Lista de capas del modelo
    //Profiler profiler;              // Perfilador para decidir CPU/GPU por fase

    MLP();

    /*
     * @brief Agrega una capa al modelo
     */
    void add_layer(Layer* layer);

    /*
     * @brief Ejecuta forward sobre todas las capas
     * @param input Entrada del modelo.
     * @param handle Puntero a cudnnHandle_t (solo si se usa GPU).
     * @param use_gpu Si es true, ejecuta en GPU.
     */
    std::vector<float> forward(const std::vector<float>& input, void* handle, bool use_gpu) const;

    /*
     * @brief Ejecuta backward sobre todas las capas
     * @param grad Gradiente respecto a la salida.
     * @param handle Puntero a cudnnHandle_t (solo si se usa GPU).
     * @param use_gpu si es true, ejecuta todo en GPU.
     */
    std::vector<float> backward(const std::vector<float>& grad, void* handle, bool use_gpu) const;

    /*
     * @brief Actualiza los pesos de cada capa usando el optimizador
     */
    void update_weights(Optimizer& optimizer);

    ~MLP() = default;
};

#endif //MLP_H
