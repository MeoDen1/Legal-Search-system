#include "searcher.h"
#include "logging.h"
#include "tokenizer.h"


namespace {
    // Collect all descendants of node `cur_idx` (including `cur_idx`)
    // NOTE: using DFS
    void collect_all_descendants(
        TreePtr* tree,
        uint32_t cur_idx,
        std::vector<uint32_t>& output
    ) {
        std::stack<uint32_t> stk;
        stk.push(cur_idx);

        while (!stk.empty()) {
            auto top = stk.top();
            stk.pop();
            output.push_back(top);
            auto node = tree->get_node(top);
            int children_start_idx = node->children_start_idx;
            
            if (children_start_idx == -1 || node->children_cnt == 0) continue;
            
            for (int i = node->children_cnt - 1; i >= 0; i--) {
                stk.push(children_start_idx + i);
            }
        }
    }
}

Searcher::Searcher(Database& db): db(&db) {
    device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
}

// The API entry point
json Searcher::search(const std::vector<float>& embedding, const std::string& query_text) {
    auto& logger = logging::Logger::get_instance();
    // 1. Convert input to Tensor (Zero-copy view)
    torch::Tensor input_tensor = torch::from_blob(
        (void*)embedding.data(), 
        {1, (long)embedding.size()}, 
        torch::kFloat32
    ).to(device);

    // 2. Start traversal from the Cluster (Root)
    std::vector<std::string> selected_tree_uids = find_relevant_trees(input_tensor);

    std::string log_msg = "Found " + std::to_string(selected_tree_uids.size()) + " relevant trees: ";
    for (auto result : selected_tree_uids) {
        log_msg += result + " ";
    }
    logger.log(log_msg, logging::INFO());

    // 3. Traverse the tree (multi-thread)
    // Get all articles (and their descendant), which is further used for BM25
    std::vector<std::future<std::vector<std::vector<uint32_t>>>> futures;
    for (std::string& tree_id : selected_tree_uids) {
        // Launch a thread for each tree
        futures.push_back(std::async(std::launch::async, [this, tree_id, &input_tensor, &query_text]() {
            return this->traverse_tree(tree_id, input_tensor, query_text);
        }));
    }

    // Collect all leaf nodes from all document trees and rank them
    std::vector<std::vector<std::vector<uint32_t>>> leaf_nodes;
    leaf_nodes.reserve(futures.size());
    for (auto& f : futures) {
        // When call .get(), the internal state of the future is destroyed
        // Only call .get() one time
        auto tree_results = f.get();
        leaf_nodes.push_back(std::move(tree_results));
    }

    int total_extracted_objects = 0;
    for (auto& selected_tree : leaf_nodes) total_extracted_objects += selected_tree.size();

    logger.log("Retrieving " + std::to_string(total_extracted_objects) + " articles/objects. Ranking...", logging::INFO());


    if (leaf_nodes.empty()) return {};

    // 5. BM25 Ranking
    std::vector<uint32_t> token_ids = db->tokenizer.tokenize(query_text);
    
    std::vector<std::future<std::vector<std::string>>> bm25_futures;
    for (int i = 0; i < selected_tree_uids.size(); i++) {
        std::string& tree_id = selected_tree_uids[i];
        auto& contents = leaf_nodes[i];
        bm25_futures.push_back(std::async(std::launch::async, [this, tree_id, &token_ids, &contents]() {
            return BM25Ranker::rank(&db->tree_ptrs[tree_id], token_ids, contents);
        }));
    }

    json output;
    for (int i = 0; i < selected_tree_uids.size(); i++) {
        json t0;
        std::string& tree_id = selected_tree_uids[i];
        std::string document_name = db->tree_ptrs[tree_id].get_content(0)->name;

        auto bm25_results = bm25_futures[i].get();
        t0["document_name"] = document_name;
        t0["articles"] = bm25_results;
        output[tree_id] = t0;
    }

    return output;
}

// Search
std::vector<std::string> Searcher::find_relevant_trees(torch::Tensor& input_tensor) {
    auto& logger = logging::Logger::get_instance();
    // Use -> since 'db' is likely a pointer/reference member in Searcher
    std::vector<std::string> selected_tree_uids;

    // 1. Initialize Queue with the Root Index (0)
    // Explicitly use std:: to avoid "too few template arguments" error
    std::queue<uint32_t> q;
    q.push(0); 

    while (!q.empty()) {
        int current_idx = q.front();
        q.pop();
        
        // 1. Fetch node and its specific model
        auto node = db->cluster_ptr.get_node(current_idx);
        std::string uid = node->uid;

        std::string prefix_log = "[" + uid + "] ";
        logger.log(prefix_log + " Searching cluster: " + uid, logging::DEBUG());

        // If the node is leaf (is_cluster = false) then add to output
        if (!node->is_cluster) {
            selected_tree_uids.push_back(node->uid);
            continue;
        }

        // 2. Inference
        torch::jit::script::Module decoder = db->get_model(uid);
        decoder.to(device);
        std::vector<torch::jit::IValue> inputs{input_tensor};
        torch::Tensor decoded_query = decoder.forward(inputs).toTensor();  // (1, dim)

        // 3. Calculate Score
        uint32_t children_start_idx = node->children_start_idx;
        long dim = db->cluster_ptr.get_vector(children_start_idx)->dim;
        // Get offset (index in vectors.bin) of the first child
        uint32_t children_start_offset = db->cluster_ptr.get_vector(children_start_idx)->offset;

        torch::Tensor children_matrix = torch::from_blob(
            db->cluster_ptr.raw_vectors_ptr + children_start_offset, 
            {node->children_cnt, dim},
            torch::kFloat32
        ); // (children_cnt, dim)

        torch::Tensor logits = torch::matmul(decoded_query, children_matrix.t());
        torch::Tensor probs = torch::sigmoid(logits).cpu();

        // 4. Vectorized Threshold Comparison
        auto probs_ptr = probs.data_ptr<float>();
        for (uint32_t i = 0; i < node->children_cnt; i++) {
            uint32_t cur_child = children_start_idx + i;
            float threshold = db->cluster_ptr.get_vector(cur_child)->threshold;

            if (probs_ptr[i] >= threshold) {
                q.push(cur_child);
            }
        }
    }
    return selected_tree_uids;
}

// Traverse tree in parallel
// For each tree, returning the article nodes (which is a vector of node indices related to the article)
std::vector<std::vector<uint32_t>> Searcher::traverse_tree(
    const std::string& tree_uid,
    torch::Tensor& input,
    const std::string& query
) {
    auto& logger = logging::Logger::get_instance();
    std::string prefix_log = "[" + tree_uid + "] ";
    std::queue<uint32_t> q;
    std::vector<std::vector<uint32_t>> results;
    std::uint32_t job_cnt = 0;
    TreePtr& tree = db->tree_ptrs[tree_uid];
    
    std::mutex mtx;
    std::mutex output_mtx;
    std::condition_variable cv;

    auto traverse = [&](uint32_t node_idx) {
        // 1. if the current node has a decoder, push the selected child node to the queue
        // 2. if the current node has children but no decoder, collect all the children
        // and use BM25 to get the score of the current node and push to output
        // 3. if the current node is a leaf, calculate score and push to output
        std::string node_uid = tree.get_content(node_idx)->uid;
        logger.log(prefix_log + " Searching node: " + node_uid, logging::DEBUG());

        torch::jit::script::Module decoder;
        bool has_decoder = false;

        // Case 1, if node supports decoder
        try {
            decoder = db->get_model(node_uid);
            decoder.to(device);
            has_decoder = true;
        }
        // Case 2: if not, using BM25 to ranking its children and push to output
        catch (std::exception& e) {
            // do nothing
        }

        if (has_decoder) {
            auto node = tree.get_node(node_idx);
            torch::Tensor decoded_query = decoder.forward({input}).toTensor();
            
            // Optimize into matrix instead
            uint32_t children_start_idx = node->children_start_idx;

            if (children_start_idx == -1) {
                throw std::runtime_error(prefix_log + "Node" + node_uid + " has decoder model but no children");
            }
            long dim = db->cluster_ptr.get_vector(children_start_idx)->dim;
            uint32_t children_start_offset = db->cluster_ptr.get_vector(children_start_idx)->offset;

            torch::Tensor children_matrix = torch::from_blob(
                db->cluster_ptr.raw_vectors_ptr + children_start_offset, 
                {node->children_cnt, dim},
                torch::kFloat32
            ); // (children_cnt, dim)
            
            torch::Tensor logits = torch::matmul(decoded_query, children_matrix.t()); // (1, children_cnt)
            torch::Tensor probs = torch::sigmoid(logits).cpu(); // (1, children_cnt)
            auto probs_ptr = probs.data_ptr<float>();

            std::vector<uint32_t> output;

            for (uint32_t i = 0; i < node->children_cnt; ++i) {
                uint32_t cur_child = node->children_start_idx + i;
                float threshold = db->cluster_ptr.get_vector(cur_child)->threshold;
                
                if (probs_ptr[i] >= threshold) output.push_back(cur_child);
            }
            
            {
                std::lock_guard<std::mutex> lock(mtx);
                for (auto child: output) q.push(child);
            }
        } else {
            // NOTE
            // For this case, there two approach
            // 1. If the current node is a leaf, push it to the output
            // 2. If the current node has children, push its children to the output
            
            // Because of this, this implement does not support to use decoder 
            // model to classify the node having children while also extracts that node
            // TODO
            // Further update should address this limitation for general and flexible use case

            auto node = tree.get_node(node_idx);
            std::vector<std::vector<uint32_t>> candidates;
            if (node->children_cnt == 0) candidates.push_back({node_idx});
            else {
                for (uint32_t i = 0; i < node->children_cnt; i++) {
                    uint32_t children_start_idx = node->children_start_idx;
                    std::vector<uint32_t> output;
                    collect_all_descendants(&tree, children_start_idx + i, output);
                    candidates.push_back(output);
                }
            }

            {
                std::lock_guard<std::mutex> lock(output_mtx);
                results.insert(
                    results.end(),
                    candidates.begin(),
                    candidates.end()
                );
            }
        }

        {
            std::lock_guard<std::mutex> lock(mtx);
            job_cnt--;
        }
        cv.notify_all();
    };


    // Start at Root of the Tree (index 0)
    q.push(0);

    while (true) {
        uint32_t curr_idx;
        {
            // the job_cnt must be locked as well, fix this
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return !q.empty() || job_cnt == 0; });

            if (q.empty() && job_cnt == 0)
                break;

            // From this point, q MUST BE NOT empty
            curr_idx = q.front();
            q.pop();
            job_cnt++;
        }
        std::thread(traverse, curr_idx).detach();
    }

    return results;
}
