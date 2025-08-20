//
// Created by zephyr on 8/19/25.
//

#ifndef MLP_BASE_TRAINER_HPP
#define MLP_BASE_TRAINER_HPP

#include <vector>

#include "loss.hpp"
#include "optimizer.hpp"
#include "mlp.hpp"

/*
 * @brief Clase encargada de entrenar un modelo MLP usando perdidad y optimizador.
 */
class Trainer {
public:
    MLP* model;
    Loss* loss;
    Optimizer* optimizer;
    bool use_gpu;
    /*
     * @brief Constructor del entrenador.
     * @param model Referencia al modelo MLP.
     * @param loss Funcion de perdida a utilizar.
     * @param optimizer Estrategia de optimizacion.
     */
    Trainer(MLP* model, Loss* loss, Optimizer* optimizer, bool use_gpu);

    /*
     * @brief Ejecuta el entrenamiento supervisado por multiples epocas.
     * @param X Conjunto de entradas (features).
     * @param Y Conjunto de etiquetas (targets).
     * @param epochs Numero de epocas de entrenamiento.
     */
    void train_batch(const std::vector<float>& input,
               const std::vector<float>& target,
               void* handle) const;

    std::vector<float> evaluate(const std::vector<float>& input, void* handle) const;
};

#endif //MLP_BASE_TRAINER_HPP