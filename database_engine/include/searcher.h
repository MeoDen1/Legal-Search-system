#pragma once
#include "types.h"
#ifndef SEARCHER_H
#define SEARCHER_H

#include "searcher.h"
#include "structure.h"
#include "bm25.h"
#include "tokenizer.h"
#include "database.h"
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <iostream>
#include <stdexcept>
#include <torch/torch.h>
#include <torch/script.h>

class Searcher {
public:
    Database* db;
    torch::DeviceType device;
    Searcher(Database& db);
    json search(const std::vector<float>& embedding, const std::string& query_text);

private:
    std::vector<std::string> find_relevant_trees(torch::Tensor& input_tensor);
    std::vector<std::vector<uint32_t>> traverse_tree(const std::string& tree_uid, torch::Tensor& input, const std::string& query);
    
};

#endif
