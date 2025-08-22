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
    [[nodiscard]] virtual float compute(const std::vector<float>& y_pred,
                          const std::vector<float>& y_true) const = 0;

    /*
     * @brief Calcula el gradiente de la perdida respecto a las salida del modelo.
     * @param y_pred Vector de salida del modelo
     * @param y_true Vector de etqiuetas verdaderas
     * @return Vector de gradientes por componente.
     */
    [[nodiscard]] virtual std::vector<float> gradient(const std::vector<float>& y_pred,
                                        const std::vector<float>& y_true) const = 0;

    /*
     * @brief Calcula el valor de la perdidad entre prediccion y etiqueta en GPU.
     * @param y_pred Puntero a salida del modelo en memoria GPU.
     * @param y_true Puntero a etiquetas verdaderas en memoria GPU.
     * @param size Numero de elementos
     * @param handle Contexto o recurso auxiliar para ejecucion en GPU
     * @return Valor escalar de la perdida
     */
    virtual float compute_gpu(const float* y_pred,
                             const float* y_true,
                             size_t size,
                             void* handle) const = 0;

    /*
     * @brief Calcula el gradiente de la perdida en GPU.
     * @param y_pred Puntero a salida del modelo en memoria GPU.
     * @param y_tru Puntero a etiquetas verdaderas en memoria GPU.
     * @param grad_out Puntero de salida para almacenar los gradientes.
     * @param size Numero de elementos.
     * @param handle Contexto o recurso auxiliar para ejecucion en GPU.
     */
    virtual void gradient_gpu(const float* y_pred,
                              const float* y_true,
                              float* grad_out,
                              size_t size,
                              void* handle) const = 0;

    /*
     * @breif Indica si la implementacion soporta ejecucion en GPU.
     * @return true si tiene implementacion GPU, false en caso contrario.
     */
    [[nodiscard]] virtual bool supports_gpu() const { return false; }
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

    float compute_gpu(const float* y_pred,
                      const float* y_true,
                      size_t size,
                      void* handle) const override;

    void gradient_gpu(const float* y_pred,
                      const float* y_true,
                      float* grad_out,
                      size_t size,
                      void* handle) const override;

    [[nodiscard]] bool supports_gpu() const override { return true; }
};


/*
 * @brief Implementacion de perdida Cross-Entropy para clasificacion multiclase.
 * Asume que las salidas estan normalizadas (softmax)
 * Soporta ejecucion en GPU mediante cuDNN.
 */
class CrossEntropyLoss final : public Loss {
public:
    [[nodiscard]] float compute(const std::vector<float>& y_pred,
                  const std::vector<float>& y_true) const override;

    [[nodiscard]] std::vector<float> gradient(const std::vector<float>& y_pred,
                                const std::vector<float>& y_true) const override;

    /*
     * @brief Calcula la perdida Cross-Entropy en GPU usando cuDNN
     * Aplica softmax logaritmico y realiza la reduccion:
     *      CE = -∑(y_true * log_softmax(y_pred))
     */
    float compute_gpu(const float* y_pred,
                      const float* y_true,
                      size_t size,
                      void* handle) const override;

    /*
     * @brief Calcula el gradiente de Cross-Entropy en GPU usando cuDNN.
     * Realiza la operacion:
     *      grad = softmax(y_pred) - y_true
     */
    void gradient_gpu(const float* y_pred,
                      const float* y_true,
                      float* grad_out,
                      size_t size,
                      void* handle) const override;

    /*
     * @brief Indica que esta implementacion soporta ejecucion en GPU.
     */
    [[nodiscard]] bool supports_gpu() const override { return true; }
};

#endif //MLP_BASE_LOSS_HPP