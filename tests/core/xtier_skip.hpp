// When the AMD tier tests skip (exit 77) and when they fail.
#pragma once

#include <string>

// Only the loader's own "not found" text or a missing card is a skip, any other failure (a dlerror text, say) is a real one.
inline bool xtier_is_skip(const std::string& e) {
    const std::string lib = "AMD tier library ";
    return (e.compare(0, lib.size(), lib) == 0 && e.find(" not found (build it") != std::string::npos) ||
           e.find("no AMD GPU") != std::string::npos;
}
