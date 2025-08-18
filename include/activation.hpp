//
// Created by zephyr on 8/14/25.
//

#ifndef MLP_BASE_ACTIVATION_H
#define MLP_BASE_ACTIVATION_H

#pragma once
#include "layer.hpp"
#include <vector>
#include <string>
#include <cudnn.h>

/*
 * @brief Tipos de activacion de funciones disponibles
 */
enum class ActType {
    ReLU,
    Sigmoid,
    Tanh
};

// CAMBIO:
// Se debe cambiar a template para que se pueda trabajar con tipos de datos
// diferentes a float.

/*
 * @brief Capa de activacion que aplica una funcion no linead elemento
 * a elemento. Hereda de Layer y opera vectores de tamano fijo.
 */
class Activation : public Layer {
public:
    /*
     * @brief Constructor de la capa de activacion
     * @param name Nombre identificador de la capa
     * @param size Tamano del vector de entrada/salida
     * @param type Tipo de activacion (ReLU, Sigmoid, Tanh)
     */
    Activation(const std::string& name, int size, ActType type);

    /*
     * @brief Propagacion hacia adelante en CPU
     */

    std::vector<float> forward_cpu(const std::vector<float>& x) override;

    /*
     * @brief Propagacion hacias atras en CPU
     */
    std::vector<float> backward_cpu(const std::vector<float>& grad) override;

    /*
     * @brief Propagacion hacia adelanta en GPU
     */
    std::vector<float> forward_gpu(const std::vector<float>& x, void* handle_void) override;

    /*
     * brief Propagacion hacia atras en GPU
     */
    std::vector<float> backward_gpu(const std::vector<float>& grad, void* handle_void) override;


private:
    ActType act_type;                       // Tipo de activacion
    std::vector<float> last_input;          // Cache de entrada para calculo de derivadas

    // cuDNN descriptors
    cudnnActivationDescriptor_t act_desc{};
    cudnnTensorDescriptor_t tensor_desc{};

    /*
     * @brief Inicializa los descriptores cuDNN segun el tipo de activacion
     */
    void setup_cudnn();
    ~Activation() override;
};

#endif //MLP_BASE_ACTIVATION_H