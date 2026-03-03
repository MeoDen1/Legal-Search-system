#include "ingestion_manager.h"
#include "logging.h"
#include "nlohmann/json.hpp"
#include "structure.h"
#include "types.h"
#include <condition_variable>
#include <cstring>
#include <stdexcept>
// #include <torch/torch.h>
// #include <torch/script.h>


// NOTE
// 1.
// For each document / tree, as the root does not have vector
// the first VectorNode in the vector_nodes will be an empty VectorNode
// However, the the first vector in the raw_vectors will be the vector of the first non-empty VectorNode
// which should be the first child of the root

// TODO
// Add incremental flow for database ingestion
// - Instead of overwriting existing database, only include the new one

namespace fs = std::filesystem;

struct Qvalue {
    json data;
    uint32_t check_id;
    uint32_t parent_idx;
    uint32_t depth;
};

std::string get_now_str() {
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char buf[80];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

IngestionManager::IngestionManager(
    const std::string& data_path,
    const std::string& db_path
): data_path(data_path), db_path(db_path) {
    if (!fs::exists(db_path)) {
        fs::create_directory(db_path);
    }
}

void IngestionManager::load_vectors(
    const std::string& file_path,
    json& vector_data
) {
    // file_path: data_path/vectors/<uid>.json
    std::ifstream vector_fp (file_path);

    if (!vector_fp.is_open()) {
        throw std::runtime_error("File: " + file_path + " is invalid");
    }

    vector_fp >> vector_data;
    vector_fp.close();
}


Tree IngestionManager::load_document(
    const std::string& file_path,
    const std::string& vector_dir
) {
    auto& logger = logging::Logger::get_instance();
    std::ifstream fp (file_path);
    if (!fp.is_open()) {
        throw std::runtime_error("File: " + file_path + " is invalid");
    }

    json data, vector_data;
    fp >> data;

    // Serial data using breadth first search
    std::vector<TreeNode*> tree_nodes;
    std::vector<ContentNode*> content_nodes;
    std::vector<VectorNode*> vector_nodes;
    std::vector<float> raw_vectors;
    std::queue<Qvalue> q;
    q.push({data, 0, 0, 0});

    logger.log("Start tokenizer: " + file_path, logging::DEBUG());
    // Tokenizer to convert word to ids
    VNTokenizer tokenizer(db_path);

    while (!q.empty()) {
        auto top = q.front();
        auto json_data = top.data;
        q.pop();

        std::string uid = json_data["uid"].get<std::string>();
        std::string prev_depth = std::to_string(top.depth-1);

        if (top.depth > 0 && vector_data.contains(prev_depth)) {
            vector_data.erase(prev_depth);
        }

        // # Create TreeNode
        TreeNode* node = (struct TreeNode*)malloc(sizeof(struct TreeNode));

        strcpy(node->uid, json_data["uid"].get<std::string>().c_str());
        node->idx = tree_nodes.size();
        node->parent_idx = top.parent_idx;
        // children_start_idx = -1 marks as leaf node
        node->children_start_idx = (json_data["subitems"].size() > 0) ? node->idx+q.size()+1 : -1;
        node->children_cnt = json_data["subitems"].size();

        if (node->idx != top.check_id) {
            throw std::runtime_error("Node id does not match expected id");
        }
        
        // # Create ContentNode
        ContentNode* content_node = (struct ContentNode*)malloc(sizeof(struct ContentNode));
        if (!content_node) throw std::runtime_error("Malloc failed");
        std::memset(content_node, 0, sizeof(struct ContentNode));

        // ## Safe copy
        std::string val_str = json_data["value"].get<std::string>();
        std::string uid_str = json_data["uid"].get<std::string>();
        std::string key_str = json_data["key"].get<std::string>();

        auto safe_copy = [](char* dest, const std::string& src, size_t limit) {
            size_t len = std::min(src.size(), limit - 1);
            std::memcpy(dest, src.c_str(), len);
            dest[len] = '\0'; 
        };

        safe_copy(content_node->uid, uid_str, 64);
        safe_copy(content_node->name, key_str, 64);
        safe_copy(content_node->value, val_str, MAX_VALUE_SIZE);
        content_node->value_size = json_data["value"].get<std::string>().size();

        // ## Calculate term and count
        std::vector<uint32_t> tokens = tokenizer.tokenize(json_data["value"].get<std::string>());
        content_node->total_words = tokens.size();

        std::map<uint32_t, uint32_t> term_count;

        for (auto& token : tokens) term_count[token]++;
        int i = 0;
        for (auto& p : term_count) {
            if (i >= MAX_TERM_SIZE) break;
            content_node->terms[i++] = {p.first, p.second};
        }
        content_node->term_cnt = i;

        // Create VectorNode
        std::string cur_depth = std::to_string(top.depth);
        VectorNode* vector_node = new VectorNode();

        strcpy(vector_node->uid, uid.c_str());

        if (vector_data.contains(cur_depth) && vector_data[cur_depth].contains(uid)) {
            vector_node->dim = vector_data[cur_depth][uid]["vector"].size();
            vector_node->threshold = float(vector_data[cur_depth][uid]["threshold"]);
            vector_node->offset = raw_vectors.size();
            std::vector<float> extracted_vector = vector_data[cur_depth][uid]["vector"].get<std::vector<float>>();
            raw_vectors.insert(raw_vectors.end(), extracted_vector.begin(), extracted_vector.end());
        }

        tree_nodes.push_back(node);
        content_nodes.push_back(content_node);
        vector_nodes.push_back(vector_node);

        // Load children vector node to `json_data`
        std::string safe_uid = uid;
        std::replace(safe_uid.begin(), safe_uid.end(), '/', '_');
        std::string vector_path = vector_dir + "/" + safe_uid + ".json";

        if (fs::exists(vector_path))
            load_vectors(vector_path, vector_data[std::to_string(top.depth + 1)]);

        for (auto child : json_data["subitems"]) {
            uint32_t child_id = node->idx + q.size() + 1;
            q.push({child, child_id, node->idx, top.depth+1});
        }
    }


    // Save tokenizer
    tokenizer.save();
    return {(int)tree_nodes.size(), tree_nodes, content_nodes, vector_nodes, raw_vectors};
}

Cluster IngestionManager::load_cluster(
    const std::string& file_path,
    const std::string& vector_dir
) {
    // file_path: data_path/cluster.json
    std::ifstream fp (file_path);
    if (!fp.is_open()) {
        throw std::runtime_error("File: " + file_path + " is invalid");
    }
    json data, vector_data;
    fp >> data;
    // Recursive traverse the root node of the tree
    // If cur_node does not have children -> TreeNode
    std::vector<ClusterNode*> cluster_nodes;
    std::vector<VectorNode*> vector_nodes;

    // NOTE
    // Store vector data into a continuous array
    // -> prevent cache miss when calculate score
    std::vector<float> raw_vectors;
    std::queue<Qvalue> q;
    q.push({data, 0, 0, 0});

    while (!q.empty()) {
        auto top = q.front();
        q.pop();

        std::string uid = top.data["uid"].get<std::string>();
        std::string prev_depth = std::to_string(top.depth-1);
        if (top.depth > 0 && vector_data.contains(prev_depth)) {
            vector_data.erase(prev_depth);
        }

        // Build ClusterNode
        ClusterNode* node = new ClusterNode();
        strcpy(node->uid, uid.c_str());
        node->idx = cluster_nodes.size();
        node->parent_idx = cluster_nodes.size();
        node->is_cluster = top.data["children"].size() > 0;
        node->children_start_idx = node->is_cluster ? node->idx+q.size()+1 : -1;
        node->children_cnt = top.data["children"].size();

        if (node->idx != top.check_id) {
            throw std::runtime_error("Node id does not match expected id");
        }

        // Build VectorNode
        std::string cur_depth = std::to_string(top.depth);
        VectorNode* vector_node = new VectorNode();

        if (top.depth > 0 && !vector_data[cur_depth].contains(uid)) {
            throw std::runtime_error("Vector for cluster: " + uid + " does not exist");
        }

        // normally around 256->768
        strcpy(vector_node->uid, uid.c_str());
        if (top.depth > 0) {
            vector_node->dim = vector_data[cur_depth][uid]["vector"].size();
            vector_node->threshold = float(vector_data[cur_depth][uid]["threshold"]);
            vector_node->offset = raw_vectors.size();
            std::vector<float> extracted_vector = vector_data[cur_depth][uid]["vector"].get<std::vector<float>>();
            raw_vectors.insert(raw_vectors.end(), extracted_vector.begin(), extracted_vector.end());
        }

        cluster_nodes.push_back(node);
        vector_nodes.push_back(vector_node);
        
        // Load children vector node to `json_data`
        std::string safe_uid = uid;
        std::replace(safe_uid.begin(), safe_uid.end(), '/', '_');
        std::string vector_path = vector_dir + "/" + safe_uid + ".json";

        load_vectors(vector_path, vector_data[std::to_string(top.depth + 1)]);

        if (top.data.contains("children")) {
            for (auto& child : top.data["children"]) {
                uint32_t child_id = node->idx + q.size() + 1;
                q.push({child, child_id, node->idx, top.depth + 1});
            }
        }
    }

    return {(int)cluster_nodes.size(), cluster_nodes, vector_nodes, raw_vectors};
}


void IngestionManager::ingest() {
    auto& logger = logging::Logger::get_instance();
    std::string documents_path = data_path + "/document_jsons";
    std::string vectors_path = data_path + "/vectors";
    std::string cluster_path = data_path + "/cluster.json";
    std::string models_path = data_path + "/models";

    json metadata;
    std::mutex metadata_mtx;
    std::mutex queue_mtx;
    std::condition_variable cv;
    std::queue<Tree> tree_queue;
    bool producer_finished = false;

    // Cluster
    try {
        Cluster cluster = load_cluster(cluster_path, vectors_path);
        metadata["cluster"] = cluster.cluster_nodes[0]->uid;
        if (cluster.size > 0)
            metadata["cluster_path"] = Serializer::save_cluster(db_path, cluster);
    } catch (const std::exception& e) {
        // If no cluster is found -> skip ingestion
        throw std::runtime_error("Cluster load failed: " + std::string(e.what()));
    }

    // Consumer
    std::thread consumer([&]() {
        while (true) {
            Tree tree;
            // Lock step: wait until a tree is available
            {
                std::unique_lock<std::mutex> lock(queue_mtx);
                cv.wait(lock, [&] { return !tree_queue.empty() || producer_finished; });

                if (tree_queue.empty() && producer_finished) break;
                if (tree_queue.empty()) continue;

                tree = std::move(tree_queue.front());
                tree_queue.pop();
            }

            std::string root_uid = tree.tree_nodes[0]->uid;
            std::string save_path = Serializer::save_tree(db_path, tree);

            // Update metadata safely
            {
                std::lock_guard<std::mutex> lock(metadata_mtx);
                metadata["trees"][root_uid] = {
                    {"path", save_path},
                    {"updated_at", get_now_str()}
                };
            }

            logger.log("Saved and cleared memory for: " + root_uid, logging::SUCCESS());
        }
    });

    // Loader (producer)
    for (const auto& entry : fs::directory_iterator(documents_path)) {
        try {
            Tree tree = load_document(entry.path().string(), vectors_path);
            {
                std::lock_guard<std::mutex> lock(queue_mtx);
                tree_queue.push(std::move(tree)); 
            }
            cv.notify_one();
        } catch (const std::exception& e) {
            logger.log("Error loading " + entry.path().filename().string() + ": " + e.what(), logging::ERROR());
        }
    }

    {
        std::lock_guard<std::mutex> lock(queue_mtx);
        producer_finished = true;
    }

    cv.notify_one();
    consumer.join();

    // Copy Models
    for (const auto& entry : fs::directory_iterator(models_path)) {
        std::string safe_uid = entry.path().stem().string();
        std::string dest_dir = db_path + "/" + safe_uid;

        fs::copy(entry.path(), dest_dir + "/model.jit", fs::copy_options::overwrite_existing);
    }

    metadata["updated_at"] = get_now_str();
    std::ofstream fp(db_path + "/metadata.json");
    fp << metadata.dump(4);
}
