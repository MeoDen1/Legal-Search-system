#include "bm25.h"
#include "structure.h"
#include <algorithm>
#include <future>

float BM25Ranker::calculate_rscore(
    const uint32_t& nqi,
    const size_t& doc_len,
    const float& avgdl,
    const BM25Config& config
) {
    // Standard Lucene/Okapi BM25 RS formula
    if (doc_len == 0) return 0.0f;
    float tf = (float)nqi / doc_len;
    return tf * (config.k1 + 1.0f) / (tf + config.k1 * (1.0f - config.b + config.b * (float)doc_len / avgdl));
}

float BM25Ranker::calculate_idf(size_t N, size_t nqi) {
    // Standard Lucene/Okapi BM25 IDF formula
    return std::log(1.0f + (N - nqi + 0.5f) / (nqi + 0.5f));
}

std::vector<std::string> BM25Ranker::rank(
    TreePtr* tree, 
    const std::vector<uint32_t>& query_tokens,
    const std::vector<std::vector<uint32_t>>& contents,
    const BM25Config& config
) {
    // Calculate N, avgdl
    size_t N = contents.size(); // Total documents in this tree
    std::vector<float> doc_lens (N, 0.0);
    std::vector<size_t> doc_size (N, 0);

    // TODO
    // + precalculate document metadata
    // - move avgdl calculate to ingestion (after consistent configuration)
    float avgdl = 0.0;

    for (size_t i = 0; i < N; i++) {
        // Candidate : root node of the content / document
        for (uint32_t node_idx: contents[i]) {
            ContentNode* content = tree->get_content(node_idx);
            doc_lens[i] += (float)content->total_words;
            // Use this to reserve full_text (+1 for each \n)
            doc_size[i] += content->value_size + 1;
        }
        avgdl += doc_lens[i];
    }

    avgdl /= N;

    std::vector<std::vector<float>> rscores (query_tokens.size(), std::vector<float> (N, 0.0));
    std::vector<float>  lscores (query_tokens.size(), 0.0);
    // Combine job_cnt_mtx with lscore (idf)

    // Calculate score for each token
    auto calculate_score = [&](int idx, uint32_t token_id) {
        // ndqi: number of documents containing the token
        uint32_t ndqi = 0; // number of contents having the token_id
        
        for (size_t i = 0; i < N; i++) {
            const std::vector<uint32_t>& candidates = contents[i];
            uint32_t nqi = 0; // number of token_id in this document
            for (uint32_t node_idx : candidates) {
                ContentNode* content = tree->get_content(node_idx);
                nqi += content->get_word_count(token_id);
            }

            if (nqi > 0) {
                // If the document contains the token,
                // increase number of documents having token by 1
                ndqi += 1;
                rscores[idx][i] = calculate_rscore(nqi, doc_lens[i], avgdl, config);
            }
        }
        lscores[idx] = calculate_idf(N, ndqi);
    };

    std::vector<std::future<void>> futures;
    futures.reserve(query_tokens.size());
    for (size_t i = 0; i < query_tokens.size(); i++) {
        uint32_t token_id = query_tokens[i];
        futures.push_back(std::async(std::launch::async, [&, i, token_id]() {
            calculate_score(i, token_id);
        }));
    }

    // Wait for all calculation complete
    for (auto& f : futures) f.get();

    // Calculate score
    std::vector<std::pair<float, std::string>> scored_results (N);

    for (int i = 0; i < N; i++) {
        float total_score = 0.0f;
        std::string full_text;
        full_text.reserve(doc_size[i]);

        for (int j = 0; j < query_tokens.size(); j++) {
            float rscore = rscores[j][i];
            float lscore = lscores[j];
            total_score += rscore * lscore;
        }
        scored_results[i].first = total_score;

        for (uint32_t node_idx : contents[i]) {
            ContentNode* content = tree->get_content(node_idx);
             full_text += std::string(content->name) + ", " + content->value;
             full_text += "\n";
        }
        scored_results[i].second = std::move(full_text);
    }


    // 3. Sort by score descending
    std::sort(scored_results.begin(), scored_results.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // 4. Return top UIDs
    std::vector<std::string> final_uids;
    final_uids.reserve(N);
    for (auto& p : scored_results) final_uids.push_back(p.second);
    
    return final_uids;
}
