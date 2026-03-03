#pragma once
#ifndef INGESTION_MANAGER_H
#define INGESTION_MANAGER_H

#include "structure.h"
#include "types.h"
#include "storage.h"
#include "logging.h"
#include "common.h"
#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <queue>
#include <string>
#include <format>
#include <mutex>
#include <thread>
#include <condition_variable>

class IngestionManager {
private:
    std::string data_path;
    std::string db_path;
    float avgdl;
    void load_vectors(const std::string& file_path, json& vector_data);
    Tree load_document(const std::string& file_path, const std::string& vector_dir);
    Cluster load_cluster(const std::string& file_path, const std::string& vector_dir);

public:
    IngestionManager(const std::string& data_path, const std::string& db_path);
    std::vector<Tree> trees;
    void ingest();
};

#endif
