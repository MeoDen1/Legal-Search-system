#pragma once
#ifndef STRUCTURE_H
#define STRUCTURE_H

#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
#define MAX_EMBEDDING_DIM 2048
#define MAX_VALUE_SIZE 4096
#define MAX_TERM_SIZE 128

// TODO
// The ContentNode is large to fit L1 & L2 cache
// uid + name = 128 bytes
// value = 4096 bytes
// total_words + value_size + term_cnt = 12 bytes
// terms: 128 * 8 = 1024 bytes
// Total ~5kB per nodes 

// - Separate TermFreq and ContentNode

struct TreeNode {
    char uid[64];
    uint32_t idx;
    float threshold;
    uint32_t parent_idx;
    int children_start_idx;
    uint32_t children_cnt;
};

struct ClusterNode {
    char uid[64];
    uint32_t idx;
    uint32_t parent_idx;
    int children_start_idx;
    uint32_t children_cnt;
    // if chilren are Tree
    bool is_cluster;
};

struct TermFreq {
    uint32_t term_id;
    uint32_t count;
};

struct ContentNode {
    char uid[64];
    char name[64];
    // # of characters
    uint32_t value_size;
    // # of words (split by space)
    uint32_t total_words = 0;
    // # of unique words
    uint32_t term_cnt = 0;
    char value[MAX_VALUE_SIZE];

    // terms must be sorted by word_id for binary search
    TermFreq terms[MAX_TERM_SIZE];

    // Using Binary Search to optimzie performance when scoring
    uint32_t get_word_count(uint32_t target_id) const {
        // if the content_node is empty, term_cnt is 0
        // then r = term_cnt - 1 = MAX_INT
        if (term_cnt == 0) return 0.0f;
        int l = 0, r = term_cnt - 1;

        while (l <= r) {
            int m = (r + l) / 2;
            if (terms[m].term_id < target_id) {
                l = m + 1;
            } else if (terms[m].term_id > target_id) {
                r = m - 1;
            } else {
                return terms[m].count;
            }
        }

        return 0;
    }
};

struct VectorNode {
    char uid[64];
    float threshold;
    uint32_t dim;
    uint32_t offset;
};

struct Tree {
    int size;
    std::vector<TreeNode*> tree_nodes;
    std::vector<ContentNode*> content_nodes;
    std::vector<VectorNode*> vector_nodes;
    std::vector<float> raw_vectors;
};

struct Cluster {
    int size;
    std::vector<ClusterNode*> cluster_nodes;
    std::vector<VectorNode*> vector_nodes;
    std::vector<float> raw_vectors;
};

struct LoggingLevel {
    int val;
    std::string name;
    LoggingLevel(const int _level, std::string name) : val(_level), name(name) {}
};


#endif
