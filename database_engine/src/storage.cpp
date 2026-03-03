#include "storage.h"
#include "logging.h"
#include <algorithm>

namespace fs = std::filesystem;

namespace {
    template<typename T>
    void write_template(
        const std::string& file_path,
        const std::vector<T*>& data
    ) {
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("File not found: " + file_path);

        for (T* item : data) file.write(reinterpret_cast<char*>(item), sizeof(T));
        file.close();
    }

    template<typename T>
    void write_template(
        const std::string& file_path,
        const std::vector<T> data
    ) {
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("File not found: " + file_path);
        
        file.write(reinterpret_cast<const char*>(data.data()), sizeof(T) * data.size());
        file.close();
    }


    template<typename T>
    void load_template(
        const std::string& file_path,
        std::vector<T*>& data
    ) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("File not found: " + file_path);

        file.seekg(0, std::ifstream::end);
        auto size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        size_t cnt = size / sizeof(T);
        // Prevent re-allocation for each push back
        data.reserve(data.size()+cnt);
        
        for (int i = 0; i < cnt; i++) {
            T* newt = new T();
            file.read(reinterpret_cast<char*>(newt), sizeof(T));
            data.push_back(newt);
        }
        file.close();
    }
}

std::string Serializer::save_cluster(
    const std::string& db_path,
    Cluster& cluster
) {
    auto& logger = logging::Logger::get_instance();
    std::vector<ClusterNode*>& cluster_nodes = cluster.cluster_nodes;
    std::vector<VectorNode*>& vector_nodes = cluster.vector_nodes;
    std::vector<float>& raw_vectors = cluster.raw_vectors;

    std::string uid = cluster_nodes[0]->uid;
    std::replace(uid.begin(), uid.end(), '/', '_');
    std::string path = db_path + "/" + uid;
    if (fs::create_directories(path))
        logger.log(path + " is created", logging::INFO());
    else logger.log(path + " already exists", logging::INFO());

    write_template(path + "/cluster.db", cluster_nodes);
    write_template(path + "/vector.db", vector_nodes);
    write_template(path + "/vector.bin", raw_vectors);

    return uid;
}

std::string Serializer::save_tree(
    const std::string& db_path,
    Tree& tree
) {
    auto& logger = logging::Logger::get_instance();
    std::vector<TreeNode*>& tree_nodes = tree.tree_nodes;
    std::vector<ContentNode*>& content_nodes = tree.content_nodes;
    std::vector<VectorNode*>& vector_nodes = tree.vector_nodes;
    std::vector<float>& raw_vectors = tree.raw_vectors;

    std::string uid = tree_nodes[0]->uid;
    std::replace(uid.begin(), uid.end(), '/', '_');
    std::string path = db_path + "/" + uid;
    if (fs::create_directories(path))
        logger.log(path + " is created", logging::INFO());
    else logger.log(path + " already exists", logging::INFO());

    write_template(path + "/tree.db", tree_nodes);
    write_template(path + "/content.db", content_nodes);
    write_template(path + "/vector.db", vector_nodes);
    write_template(path + "/vector.bin", raw_vectors);

    return uid;
}
