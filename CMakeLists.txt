cmake_minimum_required(VERSION 3.31)
project(MLP_base LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MKL CONFIG REQUIRED)
find_package(MPI REQUIRED)
#find_package(CUDA REQUIRED)

include_directories(include)

set(SOURCES
        src/mlp.cpp
        main.cpp
        src/profiler.cpp
        include/linear.h
        src/linear.cu
        include/activation.h
        src/activation.cpp
)
add_executable(mlp_exec ${SOURCES})
target_link_libraries(mlp_exec ${MKL_IMPORTED_TARGETS} ${MPI_LIBRARIES} -lcublas -lcurand)
target_include_directories(mlp_exec PRIVATE ${MKL_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
