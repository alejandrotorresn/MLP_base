//
// Created by zephyr on 8/18/25.
//

#ifndef MLP_BASE_TIMING_UTILS_HPP
#define MLP_BASE_TIMING_UTILS_HPP

#pragma once
#include <chrono>
#include <string>
#include <iostream>
#include <utility>

class Timer {
public:
    explicit Timer(std::string  label) : label(std::move(label)), start(std::chrono::high_resolution_clock::now()) {}
    ~Timer() {
        const auto end = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "[TIMER] " << label << ": " << ms << " ms\n";
    }
private:
    std::string label;
    std::chrono::high_resolution_clock::time_point start;
};

#endif //MLP_BASE_TIMING_UTILS_HPP