#include "mlp.hpp"
#include "linear.hpp"
#include "activation.hpp"
#include "tests/test_activation.hpp"
#include "tests/test_mlp_linear.hpp"
#include "tests/test_loss.hpp"
#include <iostream>


int main() {
    MLP model;
    void* handle = nullptr;         // Simulacion de contexto GPU

    // Crear capas dummy
    model.add_layer(new Linear("fc1", 2, 3));
    model.add_layer(new Linear("fc2", 3, 1));

    // Asignacion de dispositivos
    /*model.profiler.setDevice("layer1", "forward", "CPU");
    model.profiler.setDevice("layer1", "forward", "GPU");
    model.profiler.setDevice("layer2", "backward", "CPU");
    model.profiler.setDevice("layer2", "backward", "GPU");
*/
    // Datos de entrada simulados
    std::vector<float> input = {1.0f, -1.0f};
    const std::vector<float> output = model.forward(input, handle, false);
    const std::vector<float> grad = model.backward(output, handle, false);

    std::cout << "→ Forward output size: " << output.size() << std::endl;
    std::cout << "→ Backward output size: " << grad.size() << std::endl;

    // Segunda prueba
    Linear layer("linear_test", 4, 3, InitType::He);

    std::cout << "Pesos inicializados (He): \n";
    for (const float w : layer.forward_cpu({1.0f, 2.0f, 3.0f, 4.0f}))
        std::cout << w << " ";
    std::cout << std::endl;


    std::cout << "-------------------------------------------\n";
    std::cout << "Testing activation consistency...\n";
    test_activation_consistency(ActType::ReLU, 64);
    test_activation_consistency(ActType::Sigmoid, 64);
    test_activation_consistency(ActType::Tanh, 64);

    std::cout << "-------------------------------------------\n";
    std::cout << "Testing weights modification...\n";
    test_mlp_two_linear_layers();


    // Tercera prueba - Loss
    std::cout << "-------------------------------------------\n";
    std::cout << "=== Pruebas Unitarias de Losses ===\n";
    test_cross_entropy_loss(10);
    test_mse_loss(10);



    return 0;
}