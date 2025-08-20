//
// Created by zephyr on 8/19/25.
//

#ifndef MLP_BASE_OPTIMIZER_HPP
#define MLP_BASE_OPTIMIZER_HPP

#include <vector>

/*
 * @brief Interfaz abstracta para optimizadores de parametros.
 * Define el metodo para actualizar pesos a partir de gradientes.
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;

    /*
     * @brief Aplica el gradiente a los pesos del modelo.
     * @param weights Vector de pesos actuales.
     * @param grad Vector de gradientes calculados.
     */
    virtual void update(std::vector<float>& weights,
                        std::vector<float>& grad) = 0;
};

/*
 * @brief Implementacion del optimizador SGD (Stochastic Gradient Descent).
 */
class SGD : public Optimizer {
public:
    /*
     * @brief Constructor con tasa de aprendizaje.
     * @param lr Tasa de aprendizaje (learning rate).
     */
    explicit SGD(float lr);

    void update(std::vector<float>& weights,
                std::vector<float>& grad) override;
private:
    float learning_rate;
};

#endif //MLP_BASE_OPTIMIZER_HPP