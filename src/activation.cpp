//
// Created by zephyr on 8/14/25.
//
#include "activation.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

/*
 * @brief Verifica el estado de una llamada CUDa y lanza excepcion si falla.
 * @param code Codigo de error devuelto por una funcion CUDA
 * @param file Archivo fuente donde se invoca la macro
 * @param line Linea de codifo donde se invoca la macro
 * @throws std::runtime_error con mensaje detallado si hay error.
 */
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string("CUDA Error")
            + cudaGetErrorString(code) + " at " + file + ":"
            + std::to_string(line));
}

/*
 * @brief Verifica el estado de una llamanada cuDNN y lanza una exepcion si falla.
 * @param status codigo de estado devuelto por una funcion cuDNN.
 * @param file Archivo fuente donde se invoco la macro.
 * @param line Linea de codigo donde se invoco la macro
 * @throws std::runtime_error con mensaje detallado si hay error.
 */
#define CUDNN_CHECK(status) { cudnnAssert((status), __FILE__, __LINE__); }
inline void cudnnAssert(cudnnStatus_t status, const char* file, int line) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuDNN Error")
            + cudnnGetErrorString(status) + " at " + file + ":"
            + std::to_string(line));
}

/*
 * @brief Constructor de la clase Activation
 * Inicializa tipo de activacion, deescriptores cuDNN y estructura base Layer.
 * @param name Nombre de la capa
 * @param size Tamano de entrada/salida
 * @param type Tipo de activacion (ReLU, Sigmoid, Tanh)
 */
Activation::Activation(const std::string& name, const int size, const ActType type)
    : Layer(name, size, size), act_type(type), last_input(size, 0.0f) {
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensor_desc));
    setup_cudnn();      // Inicializacion de los descriptores
}

/*
 * @brief Inicializa los descriptores cuDNN para activacion y tensor.
 * configura el modo de activacion y la forma del tensor como [1,1,1,N].
 * @throws std::runtime_error si hay error en llamadas cuDNN.
 */
void Activation::setup_cudnn() {
    cudnnActivationMode_t mode = {};
    switch (act_type) {
        case ActType::ReLU:
            mode = CUDNN_ACTIVATION_RELU;
            break;
        case ActType::Tanh:
            mode = CUDNN_ACTIVATION_TANH;
            break;
        case ActType::Sigmoid:
            mode = CUDNN_ACTIVATION_SIGMOID;
            break;
    }

    CUDNN_CHECK(cudnnSetActivationDescriptor(
        act_desc, mode, CUDNN_PROPAGATE_NAN, 0.0f));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        1, 1, 1, input_size));
}

/*
 * @brief Destructor de la clase Activation.
 * Libera los descriptores cuDNN asociados a la capa
 */
Activation::~Activation() {
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(tensor_desc));
}

// Estas versiones de CPU son seriales, se deben implementar een OneDNN

/*
 * @brief Aplica la funcion de activacion en CPU.
 * @param x Vector de entrada
 * @return Vector de salida activado.
 * @throws std::invalid_argument si la dimension de entrada es incorrecta
 */
std::vector<float> Activation::forward_cpu(const std::vector<float>& x) {
    if (x.size() != input_size)
        throw std::invalid_argument("Dimension de entrada incorrecto en forward_cpu");

    last_input = x;
    std::vector<float> output(input_size, 0.0f);

    for (int i = 0; i < output_size; ++i) {
        switch (act_type) {
            case ActType::ReLU:
                output[i] = std::max(0.0f, x[i]);
                break;
            case ActType::Tanh:
                output[i] = std::tanh(x[i]);
                break;
            case ActType::Sigmoid:
                output[i] = 1.0f / (1.0f + std::exp(-1.0f * x[i]));
                break;
        }
    }
    return output;
}

/*
 * @brief Calcula el gradiente de entrada en CPU.
 * @param grad Gradiente recibido desde la capa superior.
 * @return Gradiente respecto a la entrada
 * @throws std::invalid_argument si la dimension del gradiente es incorrecta
 */
std::vector<float> Activation::backward_cpu(const std::vector<float>& grad) {
    if (grad.size() != output_size)
        throw std::invalid_argument("Dimension de entrada incorrecto en backward_cpu");

    std::vector<float> grad_input(input_size, 0.0f);

    for (int i = 0; i < input_size; ++i) {
        float derivative = 0.0f;
        switch (act_type) {
            case ActType::ReLU:
                derivative = (last_input[i] > 0.0f) ? 1.0f : 0.0f;
                break;
            case ActType::Tanh: {
                const float t = std::tanh(last_input[i]);
                derivative = 1.0f - t * t;
                break;
            }
            case ActType::Sigmoid: {
                const float sig = 1.0f / (1.0f + std::exp(-1.0f * last_input[i]));
                derivative = sig * (1.0f - sig);
                break;
            }
        }
        grad_input[i] = grad[i] * derivative;
    }
    return grad_input;
}

/*
 * @brief Aplica la funcion de activacion en GPU usando cuDNN.
 * @param x Vector de entrada
 * @param handle_void Puntero a cundnnHandle_t
 * @return Vector de salida activado
 * @throws std::invalid_argument si la dimension de entrada es incorrecta.
 * @throws std::runtime_error si hay error CUDA/cuDNN.
 */
std::vector<float> Activation::forward_gpu(const std::vector<float>& x, void* handle_void) {
    if (x.size() != input_size)
        throw std::invalid_argument("Dimension de entrada incorrecto en forward_gpu");

    last_input = x;
    const auto handle = static_cast<cudnnHandle_t>(handle_void);

    // vectores de entrada y salida en GPU
    float* d_input;
    float* d_output;

    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, input_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input,x.data(), input_size * sizeof(float),
        cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUDNN_CHECK(
        cudnnActivationForward(handle, act_desc,
        &alpha, tensor_desc, d_input,
        &beta, tensor_desc, d_output)
        );

    std::vector<float> output(input_size);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, input_size * sizeof(float),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return output;
}

/*
 * @brief calcula el gradiente de entrada en GPU usando cuDNN.
 * Utiliza cudnnActivationBacward para aplicar la derivada de la funcion de activacion
 * sobre el gradiente recibido desde la capa superior.
 * @param grad Gradiente recibido desde la capa superior (dY)
 * @param handle_void Puntero generico a cudnnHandle_t.
 * @return Gradiente respecto a la entrada (dX).
 * @throws std::invalid_argument si la dimension del gradiente es incorrecta
 * @throws std::runtime_error si hay un error CUDA/cuDNN.
 */
std::vector<float> Activation::backward_gpu(const std::vector<float>& grad, void* handle_void) {
    if (grad.size() != output_size)
        throw std::invalid_argument("Dimension de entrada incorrecto en backward_gpu");

    const auto handle = static_cast<cudnnHandle_t>(handle_void);

    // Reservar memoria en GPU
    float* d_input;         // entrada original de la activacion
    float* d_output;        // salida de la activacion
    float* d_grad;          // gradiente con respecto a la salida (dy)
    float* d_input_grad;    // gradiente con respecto a las entrada (dx)

    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_grad, input_size * sizeof(float)));

    // Copiar datos desde CPU a GPU
    CUDA_CHECK(cudaMemcpy(d_input, last_input.data(), input_size * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad, grad.data(), input_size * sizeof(float),
        cudaMemcpyHostToDevice));

    // Parametros escalares para cuDNN
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Aplicar retropropagacion de activacion: dx = dY * f'(x)
    CUDNN_CHECK(
        cudnnActivationBackward(handle, act_desc,
            &alpha, tensor_desc, d_output,    // y: salida de la activacion
            tensor_desc, d_grad,            // dy: gradiente w.r.t salida
            tensor_desc, d_input,             // x: entrada original
            &beta, tensor_desc, d_input_grad)  // dx: gradiente w.r.t. entrada
    );

    // Copiar resultado a CPU
    std::vector<float> grad_input(input_size);
    CUDA_CHECK(cudaMemcpy(grad_input.data(), d_input_grad, input_size * sizeof(float),
        cudaMemcpyDeviceToHost));

    // Liberar memoria GPU
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_input_grad));

    return grad_input;
}