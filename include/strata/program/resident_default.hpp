// The pinned RAM copy by default beside the AMD tier: when a start keeps it without being asked (header only).
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::program {

/// Whether a start keeps every expert its NVIDIA card does not hold in pinned RAM without being asked (a flag decides as given).
inline bool resident_by_default(bool linux_build, int nvidia_cards, bool amd_tier, bool resident_flag_given,
                                bool budget_given, bool off_flag, uint64_t copy_bytes, uint64_t mem_available,
                                uint64_t keep_free) {
    // no memlock test: cudaHostRegister is not charged against RLIMIT_MEMLOCK (98.17 GiB under a 23.6 GiB limit)
    if (!linux_build || off_flag || resident_flag_given || budget_given) return false;
    if (nvidia_cards != 1 || !amd_tier || copy_bytes == 0) return false;
    return mem_available >= copy_bytes && mem_available - copy_bytes >= keep_free;
}

/// The share of the bytes the NVIDIA card streams that sit in the page-locked copy, from what the copy holds (or has room for) of them.
inline double copy_pinned_share(uint64_t pinned, uint64_t streamed) {
    return streamed == 0 || pinned >= streamed ? 1.0 : (double) pinned / (double) streamed;
}

/// The decode's PCIe share beside the copy and the AMD tier (frac < 0: the link probe's rule), and whether the env holds it.
struct CopyShare {
    double frac;
    bool held;
};

/// --pcie-frac as given, else STRATA_COPY_PCIE: a share it holds, "probe" for the probe's rule, unset for 0.3 (the box's best).
inline CopyShare copy_pcie_share(double given, const char* env) {
    if (given >= 0.0) return {given, false};
    if (env == nullptr || env[0] == '\0') return {0.3, false};
    if (std::strcmp(env, "probe") == 0) return {-1.0, false};
    char* end = nullptr;
    const double v = std::strtod(env, &end);
    if (end == env || *end != '\0') return {0.3, false};
    return {v < 0.0 ? 0.0 : v > 1.0 ? 1.0 : v, true};
}

}  // namespace strata::program
