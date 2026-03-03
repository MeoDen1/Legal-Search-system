#pragma once
#ifndef DATABASE_H
#define DATABASE_H

#include "structure.h"
#include "types.h"
#include "tokenizer.h"
#include "common.h"
#include "logging.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <structure.h>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <torch/torch.h>
#include <torch/script.h>


struct Cache {
    torch::jit::script::Module model;
    long last_used_time;
};

class TreePtr {
public:
    int node_cnt;
    TreeNode* nodes_ptr = nullptr;
    ContentNode* contents_ptr = nullptr;
    VectorNode* vectors_ptr = nullptr;
    float* raw_vectors_ptr = nullptr;

    TreePtr() : node_cnt(0) {}
    TreePtr(int cnt, TreeNode* n, ContentNode* c, VectorNode* v, float* r) 
        : node_cnt(cnt), nodes_ptr(n), contents_ptr(c), vectors_ptr(v), raw_vectors_ptr(r) {}

    // 1. Disable Copying
    TreePtr(const TreePtr&) = delete;
    TreePtr& operator=(const TreePtr&) = delete;

    // 2. Enable Moving
    TreePtr(TreePtr&& other) noexcept {
        *this = std::move(other);
    }
    TreePtr& operator=(TreePtr&& other) noexcept {
        if (this != &other) {
            // Unmap existing if any
            this->cleanup();
            node_cnt = other.node_cnt;
            nodes_ptr = other.nodes_ptr;
            contents_ptr = other.contents_ptr;
            vectors_ptr = other.vectors_ptr;
            raw_vectors_ptr = other.raw_vectors_ptr;

            other.nodes_ptr = nullptr;
            other.contents_ptr = nullptr;
            other.vectors_ptr = nullptr;
            other.raw_vectors_ptr = nullptr;
            other.node_cnt = 0;
        }
        return *this;
    }

    ~TreePtr() { cleanup(); }
    TreeNode* get_node(uint32_t idx) { return nodes_ptr + idx; };
    ContentNode* get_content(uint32_t idx) { return contents_ptr + idx; };
    VectorNode* get_vector(uint32_t idx) { return vectors_ptr + idx; };
    float* get_raw_vector(uint32_t idx) { return raw_vectors_ptr + idx; };

private:
    void cleanup() {
        // Calculate size and munmap
        if (nodes_ptr) munmap(nodes_ptr, node_cnt * sizeof(TreeNode));
        if (contents_ptr) munmap(contents_ptr, node_cnt * sizeof(ContentNode));
        if (vectors_ptr) munmap(vectors_ptr, node_cnt * sizeof(VectorNode));
        if (raw_vectors_ptr) munmap(raw_vectors_ptr, node_cnt * sizeof(float));
    }
};


class ClusterPtr {
public:
    int node_cnt = 0;
    ClusterNode* nodes_ptr = nullptr;
    VectorNode* vectors_ptr = nullptr;
    float* raw_vectors_ptr = nullptr;

    // Default constructor for map compatibility
    ClusterPtr() = default;

    ClusterPtr(int cnt, ClusterNode* n, VectorNode* v, float* r)
        : node_cnt(cnt), nodes_ptr(n), vectors_ptr(v), raw_vectors_ptr(r) {}

    // 1. Disable Copying (No double-frees)
    ClusterPtr(const ClusterPtr&) = delete;
    ClusterPtr& operator=(const ClusterPtr&) = delete;

    // 2. Enable Moving (Transfer ownership)
    ClusterPtr(ClusterPtr&& other) noexcept {
        *this = std::move(other);
    }

    ClusterPtr& operator=(ClusterPtr&& other) noexcept {
        if (this != &other) {
            this->cleanup(); // Unmap current memory before taking new
            
            this->node_cnt = other.node_cnt;
            this->nodes_ptr = other.nodes_ptr;
            this->vectors_ptr = other.vectors_ptr;
            this->raw_vectors_ptr = other.raw_vectors_ptr;

            // Nullify 'other' so its destructor doesn't kill the memory we just took
            other.nodes_ptr = nullptr;
            other.vectors_ptr = nullptr;
            other.raw_vectors_ptr = nullptr;
            other.node_cnt = 0;
        }
        return *this;
    }

    // 3. Destructor
    ~ClusterPtr() {
        cleanup();
    }

    // Helper accessors
    ClusterNode* get_node(uint32_t idx) { return nodes_ptr + idx; }
    VectorNode* get_vector(uint32_t idx) { return vectors_ptr + idx; }
    float* get_raw_vector(uint32_t idx) { return raw_vectors_ptr + idx; }

private:
    void cleanup() {
        if (nodes_ptr) {
            munmap(nodes_ptr, node_cnt * sizeof(ClusterNode));
        }
        if (vectors_ptr) {
            munmap(vectors_ptr, node_cnt * sizeof(VectorNode));
        }

        if (raw_vectors_ptr) {
            munmap(raw_vectors_ptr, node_cnt * sizeof(float));
        }
        nodes_ptr = nullptr;
        vectors_ptr = nullptr;
    }
};

class Database {
public:
    std::string db_path;
    Database(const std::string& db_path);
    VNTokenizer tokenizer;

    ClusterPtr cluster_ptr;
    std::unordered_map<std::string, TreePtr> tree_ptrs;
    std::unordered_map<std::string, Cache> model_cache;
    torch::jit::script::Module get_model(const std::string& uid);

private:
    json metadata;

    TreePtr map_tree(const std::string& tree_dir);
    ClusterPtr map_cluster(const std::string& tree_dir);
};

#endif
