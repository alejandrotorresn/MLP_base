//
// Created by zephyr on 8/19/25.
//

#include "trainer.hpp"
#include <iostream>

Trainer::Trainer(MLP* model, Loss* loss, Optimizer* optimizer, bool use_gpu)
    : model(model), loss(loss), optimizer(optimizer), use_gpu(use_gpu) {}

void Trainer::train_batch(const std::vector<float>& input,
                          const std::vector<float>& target,
                          void* handle) const {
    // Paso 1: Forward
    const std::vector<float> output = model->forward(input, handle, use_gpu);
    std::cout << "[Forward] output = ";
    for (float val : output) std::cout << val << " ";
    std::cout << "\n";

    // Paso 2. Loss
    float loss_value = loss->compute(output, target);
    std::cout << "[Loss] value = " << loss_value << "\n";

    // Paso 3. Gradiente de perdida (MSE)
    const std::vector<float> grad = loss->gradient(output, target);
    std::cout << "[Loss] gradient = ";
    for (float val : grad) std::cout << val << " ";
    std::cout << "\n";

    // Paso 4. Backward
    std::vector<float> grad_input = model->backward(grad, handle, use_gpu);
    std::cout << "[Backward] grad_input = ";
    for (float val : grad_input) std::cout << val << " ";
    std::cout << "\n";

    // Paso 5. Update
    model->update_weights(*optimizer);
    std::cout << "[Update] pesos actualizados\n";
}

std::vector<float> Trainer::evaluate(const std::vector<float>& input, void* handle) const {
    return model->forward(input, handle, use_gpu);
}


