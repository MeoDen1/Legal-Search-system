#pragma once
#include "structure.h"
#include "database.h"
#include <vector>
#include <string>
#include <cmath>
#include <mutex>
#include <condition_variable>
#include <map>

// TODO
// - Extract from database.yaml config file instead of hard coding

struct BM25Config {
    float k1 = 1.5f;
    float b = 0.75f;
};

class BM25Ranker {
public:
    // We pass the tree pointer to access the ContentNodes and term frequencies
    static std::vector<std::string> rank(
        TreePtr* tree, 
        const std::vector<uint32_t>& query_tokens,
        const std::vector<std::vector<uint32_t>>& contents, // documents in BM25
        const BM25Config& config = BM25Config()
    );

private:
    static float calculate_rscore(const uint32_t& nqi, const size_t& doc_len, const float& avgdl, const BM25Config& config);
    static float calculate_idf(size_t N, size_t nqi);
};
