#include "nlohmann/json.hpp"
#include "database.h"
#include "searcher.h"
#include "ingestion_manager.h"
#include "logging.h"
#include <iostream>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <yaml-cpp/yaml.h>

void load_sample_input(std::string sample_input_path, json& sample_input) {
    // Load sample_input
    std::ifstream fp (sample_input_path);

    if (!fp.is_open()) {
        throw std::runtime_error("Can not find sample_input.json in " + sample_input_path);
    }
    
    fp >> sample_input;
    fp.close();
}

void test_sample_input(Searcher& searcher) {
    json sample_inputs;
    std::string output_path = "./output.json";
    std::ofstream fp (output_path);
    load_sample_input("./sample_input.json", sample_inputs);

    for (int i = 0; i < sample_inputs.size(); i++) {
        json sample_input = sample_inputs["input" + std::to_string(i+1)];
        std::vector<float> embedding = sample_input["embedding"].get<std::vector<float>>();
        std::string query_text = sample_input["text"].get<std::string>();
        
        json results = searcher.search(embedding, query_text);

        fp << results << std::endl;
    }
}


int main(int argc, char* argv[]) {
    auto& logger = logging::Logger::get_instance();
    logger.set_level(logging::INFO());

    YAML::Node config;
    std::string action;
    std::string data_path;
    std::string db_path;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <config_path> <action>[ingest|start]" << std::endl;
        return 1;
    }

    config = YAML::LoadFile(argv[1]);
    action = std::string(argv[2]);

    data_path = config["data_path"].as<std::string>();
    db_path = config["db_path"].as<std::string>();

    if (action == "ingest") {
        IngestionManager ingestion_manager (data_path, db_path);
        ingestion_manager.ingest();
        return 0;
    } else if (action != "start") {
        std::cerr << "Usage: " << argv[0] << " <config_path> <action>[ingest|start]" << std::endl;
        return 1;
    }

    Database db(db_path);
    Searcher searcher(db);

    test_sample_input(searcher);

    return 0;
}
