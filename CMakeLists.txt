cmake_minimum_required(VERSION 3.31)
project(MLP_base LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MKL CONFIG REQUIRED)
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(include)

set(SOURCES
        src/mlp.cpp
        main.cpp
        src/profiler.cpp
        include/linear.hpp
        src/linear.cu
        include/activation.hpp
        src/activation.cpp
        src/activation.cpp
        include/gpu_activation_helper.hpp
        src/gpu_activation_helper.cpp
        include/gpu_error_checking.hpp
        tests/test_activation.cpp
        tests/test_activation.hpp
        include/timing_utils.hpp
        include/loss.hpp
        src/loss.cpp
        include/optimizer.hpp
        src/optimizer.cpp
        include/trainer.hpp
        src/trainer.cpp
        tests/test_mlp_linear.cpp
        tests/test_mlp_linear.hpp
        src/loss.cu
)
add_executable(mlp_exec ${SOURCES})

#target_link_libraries(mlp_exec ${MKL_IMPORTED_TARGETS} ${MPI_LIBRARIES} -lcublas -lcurand)
target_link_libraries(mlp_exec
        ${MKL_IMPORTED_TARGETS}
        ${MPI_LIBRARIES}
        ${CUDNN_LIBRARY}
        CUDA::cublas
        CUDA::cudart
        -lcurand
        -lcudnn
)
target_include_directories(mlp_exec PRIVATE ${MKL_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})

message(STATUS "cuBLAS: ${CUBLAS_LIB}")
message(STATUS "cuDNN: ${CUDNN_LIB}")