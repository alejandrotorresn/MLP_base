//
// Created by zephyr on 8/19/25.
//

#include "test_mlp_linear.hpp"
#include "mlp.hpp"
#include "linear.hpp"
#include "optimizer.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

/*
 * @brief Compara dos vectores con tolerancia.
 */
bool compare_vectors_optimization(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-6f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

/*
 * @brief ejecuta un test unitario sobre una red MLP de dos capas lineales.
 */
void test_mlp_two_linear_layers() {
    std::cout << "Testing MLP two linear layers\n";

    // Crear modelo
    MLP model;
    model.add_layer(new Linear("fc1", 2, 3));
    model.add_layer(new Linear("fc2", 3, 1));

    std::cout << "Modelo creado con capas:\n";
    for (auto* layer : model.layers) {
        std::cout << " - " << layer->name << ": in=" << layer->input_size << ", out=" << layer->output_size << "\n";
    }

    // Crear optimizador
    SGD optimizer(0.1f);

    // Entrada y salida esperada
    std::vector<float> input = {1.0f, -1.0f};
    std::vector<float> target = {0.5f};

    // Paso 1: forward
    std::vector<float> output_before = model.forward(input, nullptr, true);
    std::cout << "Output inicial: " << output_before[0] << "\n";

    // Paso 2: calcular gradientes de perdida (MSE)
    std::vector<float> grad = {2.0f * (output_before[0] - target[0])};

    // Paso 3: backward
    model.backward(grad, nullptr, true);

    // Paso 4: guardar pesos antes de actualizar
    auto* fc2 = dynamic_cast<Linear*>(model.layers[1]);
    std::vector<float> weights_before = fc2->forward_cpu({1.0f, 1.0f, 1.0f}); //dummy input

    // Paso 5: actualizar pesos
    model.update_weights(optimizer);

    // Paso 6: verificar que los pesos cambiaron
    std::vector<float> weights_after = fc2->forward_cpu({1.0f, 1.0f, 1.0f});
    bool changed = !compare_vectors_optimization(weights_before, weights_after);

    std::cout << "Pesos modificados?: " << (changed ? "Si" : "No") << "\n";

    // Paso 7: nuevo forward
    std::vector<float> output_after = model.forward(input, nullptr, true);
    std::cout << "Output despues de update: " << output_after[0] << "\n";

    assert(changed && "Los pesos no se modificaron tras update_weights");
    std::cout << "Test completado correctamente.\n";
}