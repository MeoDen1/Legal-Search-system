#include "logging.h"
#include "structure.h"

namespace logging {
    LoggingLevel& DEBUG()   { static LoggingLevel val(0, "DEBUG");   return val; }
    LoggingLevel& INFO()    { static LoggingLevel val(1, "INFO");    return val; }
    LoggingLevel& SUCCESS() { static LoggingLevel val(2, "SUCCESS"); return val; }
    LoggingLevel& WARNING() { static LoggingLevel val(3, "WARNING"); return val; }
    LoggingLevel& ERROR()   { static LoggingLevel val(4, "ERROR");   return val; }

    Logger::Logger() {
        this->level = &INFO();
    }

    void Logger::set_level(LoggingLevel& level) {
        // 0: the value remain unchanged if the variable already exists
        // !0: overwrite the value
        this->level = &level;
    }

    void Logger::log(const std::string msg, const LoggingLevel& level) {
        // If cur_level > level -> skip log
        if (this->level->val > level.val) return;
        else std::cout << level.name << " | " << msg << std::endl;
    }
}
