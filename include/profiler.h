//
// Created by vermeer on 8/12/25.
//

#ifndef PROFILER_H
#define PROFILER_H

#pragma once
#include <string>
#include <unordered_map>

/*
 * @brief Clase para decidir si una capa se ejecuta en CPU o GPU por fase.
 */
class Profiler
{
    std::unordered_map<std::string, std::string> device_map;    // Mapa capa+fase → dispositivo

public:
    /*
     * @brief Asigna dispositivo para una capa y fase
     */
    void setDevice(const std::string& layer_name, const std::string& phase, const std::string& device) {
        device_map[layer_name + "_" + phase] = device;
    }

    /*
     * @brief Obtiene el dispositivo asignado para una capa y fase
     */
    std::string getDevice(const std::string& layer_name, const std::string& phase) {
        return device_map[layer_name + "_" + phase];
    }
};

#endif //PROFILER_H
