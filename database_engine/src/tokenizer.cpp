#include "tokenizer.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

VNTokenizer::VNTokenizer(const std::string db_path): db_path(db_path) {
    std::string vocab_dir = db_path + "/tokenizers";
    vocab_path = vocab_dir + "/vn_vocab.json";

    // 1. Ensure directory exists
    if (!fs::exists(vocab_dir)) {
        fs::create_directories(vocab_dir);
    }

    // 2. Load existing vocab if it exists
    if (fs::exists(vocab_path)) {
        std::ifstream file(vocab_path);
        if (file.is_open()) {
            json j;
            file >> j;
            vocab = j.get<std::unordered_map<std::string, uint32_t>>();
            next_id = vocab.size(); 
        }
    } else {
        next_id = 0;
    }
    
}

// TODO
// Auto assign tokenizer, must be improve to better tokenizer
std::vector<uint32_t> VNTokenizer::tokenize(const std::string& text) {
    std::vector<uint32_t> ids;
    std::string current_word;

    auto process_word = [&](std::string& word) {
        if (word.empty()) return;
        
        // Normalization
        std::transform(word.begin(), word.end(), word.begin(), 
                       [](unsigned char wc){ return std::tolower(wc); });

        // If it's a new word, assign a new ID
        if (vocab.find(word) == vocab.end()) {
            vocab[word] = next_id++;
        }
        ids.push_back(vocab[word]);
        word.clear();
    };

    for (size_t i = 0; i < text.length(); ++i) {
        unsigned char c = text[i];
        if (std::isspace(c) || std::ispunct(c)) {
            process_word(current_word);
        } else {
            current_word += c;
        }
    }
    
    // Catch the last word if string doesn't end in punctuation/space
    process_word(current_word);

    return ids;
}

void VNTokenizer::save() {
    // Create json object from map
    json j = vocab;
    
    std::ofstream file(vocab_path);
    if (file.is_open()) {
        // Use dump(4) for pretty-printing, making it readable for debugging
        file << j.dump(4);
    }
}
