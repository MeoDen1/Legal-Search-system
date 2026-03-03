#pragma once
#ifndef STORAGE_H
#define STORAGE_H

#include "structure.h"
#include "logging.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstddef>
#include <filesystem>

class Serializer {
public:
    static std::string save_cluster(
        const std::string& db_path,
        Cluster& cluster
    );

    static std::string save_tree(
        const std::string& db_path,
        Tree& tree
    );
};

#endif
