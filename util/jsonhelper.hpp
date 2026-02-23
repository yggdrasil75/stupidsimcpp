#ifndef JSONHELPER_HPP
#define JSONHELPER_HPP

#include <string>
#include <vector>
#include <cctype>

namespace JsonHelper {

    // Helper to get string value between quotes
    inline std::string parseString(const std::string& json, const std::string& key) {
        size_t pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = json.find(":", pos);
        if (pos == std::string::npos) return "";
        size_t start = json.find("\"", pos);
        if (start == std::string::npos) return "";
        size_t end = json.find("\"", start + 1);
        if (end == std::string::npos) return "";
        return json.substr(start + 1, end - start - 1);
    }

    // Helper to get raw non-string value (int, float, bool)
    inline std::string parseRaw(const std::string& json, const std::string& key) {
        size_t pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = json.find(":", pos);
        if (pos == std::string::npos) return "";
        pos++; // skip ':'
        
        while (pos < json.length() && std::isspace(json[pos])) pos++;
        size_t end = pos;
        while (end < json.length() && json[end] != ',' && json[end] != '}' && json[end] != ']' && !std::isspace(json[end])) end++;
        
        return json.substr(pos, end - pos);
    }

    inline int parseInt(const std::string& json, const std::string& key, int defaultVal = 0) {
        std::string raw = parseRaw(json, key);
        if (raw.empty()) return defaultVal;
        try { return std::stoi(raw); } catch(...) { return defaultVal; }
    }

    inline float parseFloat(const std::string& json, const std::string& key, float defaultVal = 0.0f) {
        std::string raw = parseRaw(json, key);
        if (raw.empty()) return defaultVal;
        try { return std::stof(raw); } catch(...) { return defaultVal; }
    }

    inline bool parseBool(const std::string& json, const std::string& key, bool defaultVal = false) {
        std::string raw = parseRaw(json, key);
        if (raw.empty()) return defaultVal;
        return raw == "true" || raw == "1";
    }

    // Helper to extract JSON objects out of a JSON array
    inline std::vector<std::string> parseArray(const std::string& json, const std::string& key) {
        std::vector<std::string> items;
        size_t pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return items;
        pos = json.find(":", pos);
        if (pos == std::string::npos) return items;
        pos = json.find("[", pos);
        if (pos == std::string::npos) return items;
        
        int depth = 0;
        size_t start = 0;
        bool inString = false;
        
        for (size_t i = pos + 1; i < json.length(); ++i) {
            if (json[i] == '"' && (i == 0 || json[i-1] != '\\')) {
                inString = !inString;
            }
            if (!inString) {
                if (json[i] == '{') {
                    if (depth == 0) start = i;
                    depth++;
                } else if (json[i] == '}') {
                    depth--;
                    if (depth == 0) {
                        items.push_back(json.substr(start, i - start + 1));
                    }
                } else if (json[i] == ']') {
                    if (depth == 0) break;
                }
            }
        }
        return items;
    }

}

#endif