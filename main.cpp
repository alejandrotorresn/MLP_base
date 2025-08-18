#include "mlp.hpp"
#include "test_activation.hpp"
#include "linear.hpp"
#include <iostream>

#include "activation.hpp"

/*
 * @brief Clase dummy para pruebas. No realiza operaciones reales
 */
class DummyLayer : public Layer {
public:
    DummyLayer(const std::string& name, const int in, const int out) : Layer(name, in, out) {}

    std::vector<float> forward_cpu(const std::vector<float>& x) override {
        std::cout << "-> Dummy forward CPU\n";
        return std::vector<float>(output_size, 1.0f);
    }

    std::vector<float> backward_cpu(const std::vector<float>& grad) override {
        std::cout << "-> Dummy backward CPU\n";
        return std::vector<float>(input_size, 0.5f);
    }

    std::vector<float> forward_gpu(const std::vector<float>& x, void* handle) override {
        std::cout << "-> Dummy forward GPU\n";
        return std::vector<float>(output_size, 2.0f);
    }

    std::vector<float> backward_gpu(const std::vector<float>& grad, void* handle) override {
        std::cout << "-> Dummy backward GPU\n";
        return std::vector<float>(input_size, 0.25f);
    }
};

int main() {
    MLP model;
    void* handle = nullptr;         // Simulacion de contexto GPU

    // Crear capas dummy
    model.add_layer(new DummyLayer("layer1", 10, 5));
    model.add_layer(new DummyLayer("layer2", 5, 2));

    // Asignacion de dispositivos
    model.profiler.setDevice("layer1", "forward", "CPU");
    model.profiler.setDevice("layer1", "forward", "GPU");
    model.profiler.setDevice("layer2", "backward", "CPU");
    model.profiler.setDevice("layer2", "backward", "GPU");

    // Datos de entrada simulados
    const std::vector<float> input(10, 1.0f);
    const std::vector<float> output = model.forward(input, handle);
    const std::vector<float> grad = model.backward(output, handle);

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

    return 0;
}