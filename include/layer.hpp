//
// Created by vermeer on 8/12/25.
//

#ifndef LAYER_H
#define LAYER_H

#pragma once
#include <string>
#include <utility>
#include <vector>
#include "optimizer.hpp"

/*
 * @brief Clase abstracta base para todas lsa capas del MLP.
 * Define la interfaz comun para forward y backward en CPU y GPU.
 */
class Layer {
public:
    std::string name;   // Nombre del identificador de la capa
    int input_size;     // Tamano del vector de entrada
    int output_size;    // Tamano del vector de salida

    Layer(std::string name, const int in, const int out)
        : name(std::move(name)), input_size(in), output_size(out) {}

    /*
     * @brief Propagacion hacia adelante en CPU
     */
    virtual std::vector<float> forward_cpu(const std::vector<float>& x) = 0;

    /*
     * @brief Propagacion hacia atras en CPU
     */
    virtual std::vector<float> backward_cpu(const std::vector<float>& grad) = 0;

    /*
     * @brief Propagacion hacia adelante en GPU
     */
    virtual std::vector<float> forward_gpu(const std::vector<float>& x, void* handle) = 0;

    /*
     * @brief Propagacion hacia atras en GPU
     */
    virtual std::vector<float> backward_gpu(const std::vector<float>& grad, void* handle) = 0;

    /*
     * @brief Aplica la actualizacion de pesos usando el optimizador.
     * Las capas sin parametros pueden dejar esta funcion vacia.
     * @param optimizer Referencia al optimizador.
     */
    virtual void update_weights(Optimizer& optimizer) {};

    virtual ~Layer() = default;
};

#endif //LAYER_H
