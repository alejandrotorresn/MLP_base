//
// Created by vermeer on 8/12/25.
//

#ifndef LINEAR_H
#define LINEAR_H

#pragma once
#include "layer.h"

enum class InitType {
    Xavier,
    He
};

/*
 * @brief Capa lineal (fully connected) que implementa la interfaz Layer.
 * Realiza la operacion: y = Wx + b
 */
class Linear : public Layer {
public:
    /*
     * @brief Constructor de la capa lineal
     * @param name Nombre identificador de la capa
     * @param in Numero de entradas
     * @param out Numero de salidas
     */
    Linear(const std::string& name, int in, int out, InitType init = InitType::Xavier);

    /*
     * @brief Propagacion hacia adelante en CPU
     */
    std::vector<float> forward_cpu(const std::vector<float> &x) override;

    /*
     * @brief Propagacion hacia atras en CPU
     */
    std::vector<float> backward_cpu(const std::vector<float> &grad) override;

    /*
     * @brief Propagacion hacia adelante en GPU (placeholder)
     */
    std::vector<float> forward_gpu(const std::vector<float> &x, void* handle) override;

    /*
     * @brief Propagacion hacia atras en GPU (placeholder)
     */
    std::vector<float> backward_gpu(const std::vector<float> &x,void* handle) override;

private:
    std::vector<float> weights;     // Matriz de pesos [out x in]
    std::vector<float> bias;        // Vector de sesgos [out]
    std::vector<float> last_input;  // Cache de entrada para backward

    void initialize_weights(InitType init);
};

#endif //LINEAR_H
