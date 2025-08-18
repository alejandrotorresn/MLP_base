//
// Created by zephyr on 8/18/25.
//

#ifndef MLP_BASE_GPU_ACTIVATION_HELPER_H
#define MLP_BASE_GPU_ACTIVATION_HELPER_H

#include <vector>
#include <cudnn.h>

/*
 * @brief Clase auxiliar para ejecutar funciones de activacion en GPU usando cuDNN.
 */
class GPUActivationHelper {
public:
    GPUActivationHelper(int size, cudnnActivationDescriptor_t act_desc, cudnnTensorDescriptor_t tensor_desc);
    ~GPUActivationHelper();

    std::vector<float> forward(const std::vector<float>& input, cudnnHandle_t handle) const;
    std::vector<float> backward(const std::vector<float>& input,const std::vector<float>& grad, cudnnHandle_t handle) const;

private:
    int input_size;
    cudnnActivationDescriptor_t act_desc;
    cudnnTensorDescriptor_t tensor_desc;

    float* d_input;
    float* d_output;
    float* d_grad;
    float* d_input_grad;

    void allocate();
    void release() const;
};

#endif //MLP_BASE_GPU_ACTIVATION_HELPER_H