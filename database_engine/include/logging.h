#pragma once
#ifndef LOGGING_H
#define LOGGING_H

#include "structure.h"
#include <iostream>
#include <string>
#include <cstdlib>

namespace logging {
    // Meyer singleton syntax
    LoggingLevel& DEBUG();
    LoggingLevel& INFO();
    LoggingLevel& SUCCESS();
    LoggingLevel& WARNING();
    LoggingLevel& ERROR();

    class Logger {
    public:
        static Logger& get_instance() {
            static Logger instance;
            return instance;
        }
        Logger(Logger const&) = delete;
        void operator=(Logger const&) = delete;
        void set_level(LoggingLevel& level);
        void log(const std::string msg, const LoggingLevel& level);
    private:
        Logger();
        std::string varname = "DB_LOGGING_LEVEL";
        LoggingLevel* level;
        char* get_level();
    };

}
#endif
