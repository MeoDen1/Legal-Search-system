#include "database.h"
#include <fstream>

namespace fs = std::filesystem;

// Database
template<typename T>
int map_ptr(const std::string& file_db_path, T*& ptr) {
    ptr = nullptr; // Initialize to null
    int fd = open(file_db_path.c_str(), O_RDONLY);
    if (fd == -1) return 0; // Or throw error

    struct stat st;
    if (fstat(fd, &st) == -1 || st.st_size == 0) {
        close(fd);
        return 0;
    }

    size_t size = st.st_size;
    void* addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); // You can close FD immediately after mmap

    if (addr == MAP_FAILED) return 0;

    ptr = static_cast<T*>(addr);
    return static_cast<int>(size / sizeof(T));
}

TreePtr Database::map_tree(const std::string& tree_dir) {
    TreeNode* nodes_ptr;
    ContentNode* contents_ptr;
    VectorNode* vectors_ptr;
    float* raw_vectors;

    int node_cnt_1 = map_ptr(tree_dir + "/tree.db", nodes_ptr);
    int node_cnt_2 = map_ptr(tree_dir + "/content.db", contents_ptr);
    int node_cnt_3 = map_ptr(tree_dir + "/vector.db", vectors_ptr);

    if (node_cnt_1 != node_cnt_2 || node_cnt_1 != node_cnt_3)
        throw std::runtime_error("Number of nodes between db files does not match");

    map_ptr(tree_dir + "/vector.bin", raw_vectors);
    // std::cout << nodes_ptr->children_cnt << " " << nodes_ptr->uid << std::endl;
    // std::cout << (vectors_ptr + 1)->dim << " " << vectors_ptr->uid << std::endl;
    // std::cout << raw_vectors << std::endl;

    return {node_cnt_1, nodes_ptr, contents_ptr, vectors_ptr, raw_vectors};
}

ClusterPtr Database::map_cluster(const std::string& tree_dir) {
    ClusterNode* nodes_ptr;
    VectorNode* vectors_ptr;
    float* raw_vectors;

    int node_cnt_1 = map_ptr(tree_dir + "/cluster.db", nodes_ptr);
    int node_cnt_2 = map_ptr(tree_dir + "/vector.db", vectors_ptr);

    if (node_cnt_1 != node_cnt_2)
        throw std::runtime_error("Number of nodes between db files does not match");

    map_ptr(tree_dir + "/vector.bin", raw_vectors);
    // std::cout << nodes_ptr->children_cnt << " " << nodes_ptr->uid << std::endl;
    // std::cout << (vectors_ptr+1)->dim << " " << vectors_ptr->uid << std::endl;
    // std::cout << raw_vectors << std::endl;

    return {node_cnt_1, nodes_ptr, vectors_ptr, raw_vectors};
}

Database::Database(const std::string& db_path): db_path(db_path), tokenizer(db_path) {
    std::ifstream fp (db_path + "/metadata.json");

    if (!fp.is_open()) {
        throw std::runtime_error("Can not find metadata.json in " + db_path);
    }

    fp >> metadata;
    std::string cluster_path = metadata["cluster_path"].get<std::string>();
    cluster_ptr = map_cluster(db_path + "/" + cluster_path);

    for (auto& [uid, meta] : metadata["trees"].items()) {
        tree_ptrs[uid] = map_tree(db_path + "/" + meta["path"].get<std::string>());
    }
}

torch::jit::script::Module Database::get_model(const std::string& uid) {
    auto& logger = logging::Logger::get_instance();
    // TODO
    // Auto clear cache if memory limit reaches
    // (remove the least recently used model)

    // Logic
    // Check if model exists in database
    // If exists, return model
    // - If the model is not in the cache, load it
    // If not, return error -> searcher should handle this
    std::string safe_uid = uid;
    std::replace(safe_uid.begin(), safe_uid.end(), '/', '_');
    std::string model_path = db_path + "/" + safe_uid + "/model.jit";

    if (!fs::exists(model_path)) {
        throw std::runtime_error("Model " + uid + " not found in database");
    }

    if (model_cache.find(uid) == model_cache.end()) {
        std::string safe_uid = uid;
        std::replace(safe_uid.begin(), safe_uid.end(), '/', '_');

        model_cache[uid] = {
            torch::jit::load(model_path),
            // update the last used time (long)
            std::time(nullptr)
        };
    }

    return model_cache[uid].model;

}
