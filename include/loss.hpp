//
// Created by zephyr on 8/19/25.
//

#ifndef MLP_BASE_LOSS_HPP
#define MLP_BASE_LOSS_HPP

#include <vector>

/*
 * @brief Interfaz abstracta para funciones de perdidad supervisada.
 * Define los metodos necesarios para calcular el valor de la perdida.
 * y su gradiente respecto a la salida del modelo.
 */
class Loss {
public:
    virtual ~Loss() = default;

    /*
     * @brief Calcula el valor escalar de la perdida entre prediccion y etiqueta.
     * @param y_pred Vector de salida del modelo.
     * @param y_true Vector de etiquetas verdaderas.
     * @return Valor escalar de la perdida.
     */
    virtual float compute(const std::vector<float>& y_pred,
                          const std::vector<float>& y_true) const = 0;

    /*
     * @brief Calcula el gradiente de la perdida respecto a las salida del modelo.
     * @param y_pred Vector de salida del modelo
     * @param y_true Vector de etqiuetas verdaderas
     * @return Vector de gradientes por componente.
     */
    virtual std::vector<float> gradient(const std::vector<float>& y_pred,
                                        const std::vector<float>& y_true) const = 0;
};

/*
 * @brief Implementacion de perdida MSE (Mean Square Error) para regresion.
 */
class MSELoss final : public Loss {
public:
    [[nodiscard]] float compute(const std::vector<float>& y_pred,
                  const std::vector<float>& y_true) const override;
    [[nodiscard]] std::vector<float> gradient(const std::vector<float>& y_pred,
                                const std::vector<float>& y_true) const override;
};

/*
 * @brief Implementacion de perdida Cross-Entropy para clasificacion multiclase.
 * Asume que las salidas estan normalizadas (softmax)
 */
class CrossEntropyLoss final : public Loss {
public:
    [[nodiscard]] float compute(const std::vector<float>& y_pred,
                  const std::vector<float>& y_true) const override;

    [[nodiscard]] std::vector<float> gradient(const std::vector<float>& y_pred,
                                const std::vector<float>& y_true) const override;
};

#endif //MLP_BASE_LOSS_HPP