#include "common.h"
#include "logging.h"

namespace common {
    // Custom json string conversion
    std::string convert_json_str(json& data, const std::vector<std::string> keys) {
        json filtered_json = json::object();

        for (const auto& key: keys) {
            if (data.contains(key)) {
                filtered_json[key] = data[key];
            }
        }

        return filtered_json.dump();
    }

    std::string convert_dict_str(
        std::unordered_map<std::string, std::string>& data,
        std::vector<std::string> keys
    ) {
        json j;
        if (!keys.size()) j = data;
        else {
            j = json::object();
            for (const auto& key : keys) {
                auto it = data.find(key);
                if (it != data.end()) {
                    j[key] = it->second;
                }
            }
        }

        return j.dump();
    }

    std::unordered_map<std::string, std::string> convert_jsonstr_dict(const std::string jsonstr) {
        // Regex to find single quotes at JSON boundaries
        // 1. ([:{,])\s*'  Matches ' preceded by { , or :
        // 2. '\s*([:},])   Matches ' followed by } , or :
        std::regex json_open_quote("([\\{:,])\\s*'");
        std::regex json_close_quote("'\\s*([\\}:,])");
        auto& logger = logging::Logger::get_instance();

        auto check = [&](const std::string& str) {
            size_t first_bracket = str.find('{');
            if (first_bracket == std::string::npos) return false;

            for (size_t i = first_bracket + 1; i < str.length(); i++) {
                if (std::isspace(str[i])) continue;
                if (str[i] == '\'') return false;
                if (str[i] == '"') return true;
            }

            return false;
        };

        bool is_valid = check(jsonstr);
        std::unordered_map<std::string, std::string> umap;
        std::string preprocessed_jsonstr = jsonstr;

        if (!is_valid) {
            preprocessed_jsonstr = std::regex_replace(preprocessed_jsonstr, json_open_quote, "$1 \"");
            preprocessed_jsonstr = std::regex_replace(preprocessed_jsonstr, json_close_quote, "\"$1");
        }

        json j = json::parse(preprocessed_jsonstr);
        for (auto& [key, value] : j.items()) {
            umap[key] = value.get<std::string>();
        }

        return umap;
    }
}
