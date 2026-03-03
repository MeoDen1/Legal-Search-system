#pragma once
#ifndef COMMON_H
#define COMMON_H

#include "types.h"
#include <iostream>
#include <random>
#include <sstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace common {
    std::string convert_json_str(json &data, const std::vector<std::string> keys);
    std::string convert_dict_str(std::unordered_map<std::string, std::string>& data, std::vector<std::string> keys);
    std::unordered_map<std::string, std::string> convert_jsonstr_dict(const std::string jsonstr);
}

#endif
