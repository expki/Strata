// --expert-gib: a cap on the expert memory of each card, in GiB per CUDA device and per AMD tier card (header only).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>

namespace strata::program {

/// "cuda0=1,hip0=6.5" holds CUDA device 0 to 1 GiB of experts and AMD tier card 0 to 6.5 GiB, a card not listed has no cap.
struct ExpertGib {
    std::map<int, double> cuda;
    std::map<int, double> hip;
};

namespace expert_gib_detail {

constexpr double kGiB = 1073741824.0;

/// One "cudaN=GiB" or "hipN=GiB" item into `out`.
inline bool parse_item(const std::string& item, ExpertGib& out, std::string& err) {
    const size_t eq = item.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 == item.size()) {
        err = "\"" + item + "\" is not NAME=GiB";
        return false;
    }
    const std::string key = item.substr(0, eq), val = item.substr(eq + 1);
    const bool hip = key.compare(0, 3, "hip") == 0;
    if (!hip && key.compare(0, 4, "cuda") != 0) {
        err = "\"" + key + "\" is not cudaN or hipN";
        return false;
    }
    const std::string num = key.substr(hip ? 3 : 4);
    if (num.empty() || num.size() > 3 || num.find_first_not_of("0123456789") != std::string::npos) {
        err = "\"" + key + "\" needs a device number after " + (hip ? "hip" : "cuda");
        return false;
    }
    int dots = 0;
    for (const char c : val) dots += c == '.';
    if (val.find_first_not_of("0123456789.") != std::string::npos || dots > 1 || val == ".") {
        err = "\"" + val + "\" is not a number of GiB";
        return false;
    }
    const double gib = std::strtod(val.c_str(), nullptr);
    if (!std::isfinite(gib) || !(gib * kGiB >= 1.0)) {
        err = "\"" + val + "\" must be a positive number of GiB";
        return false;
    }
    auto& into = hip ? out.hip : out.cuda;
    if (!into.emplace(std::atoi(num.c_str()), gib).second) {
        err = "\"" + key + "\" is given twice";
        return false;
    }
    return true;
}

}  // namespace expert_gib_detail

/// Reads "cuda0=1,hip0=6.5" into `out`.  Returns false with a message in `err` and leaves `out` as it was.
inline bool parse_expert_gib(const std::string& text, ExpertGib& out, std::string& err) {
    if (text.empty()) {
        err = "the list is empty";
        return false;
    }
    ExpertGib got;
    for (size_t at = 0;;) {
        const size_t comma = text.find(',', at);
        const std::string item = text.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        if (!expert_gib_detail::parse_item(item, got, err)) return false;
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    out = std::move(got);
    return true;
}

/// The cap in bytes for a device, or UINT64_MAX when none is set.
inline uint64_t expert_cap_bytes(const ExpertGib& g, bool hip, int device) {
    const auto& m = hip ? g.hip : g.cuda;
    const auto it = m.find(device);
    if (it == m.end()) return UINT64_MAX;
    const double bytes = it->second * expert_gib_detail::kGiB;
    return bytes >= 18446744073709551615.0 ? UINT64_MAX : (uint64_t) bytes;
}

}  // namespace strata::program
