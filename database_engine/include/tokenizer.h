#pragma once
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "nlohmann/json.hpp"
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <algorithm>

class VNTokenizer {
private:
    std::string db_path;
    std::string vocab_path;
    long next_id;
    std::unordered_map<std::string, uint32_t> vocab;

public:
    VNTokenizer(const std::string db_path);

    // Convert string to IDs
    std::vector<uint32_t> tokenize(const std::string& text);
    void save();
};

#endif
