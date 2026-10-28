// The pinned RAM copy by default: a lone NVIDIA card with the AMD tier and the RAM for it, nothing else. CPU only.
#include "strata/program/resident_default.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using strata::program::copy_pcie_share;
using strata::program::copy_pinned_share;
using strata::program::CopyShare;
using strata::program::resident_by_default;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

constexpr uint64_t GiB = 1ull << 30;
constexpr uint64_t kCopy = 98 * GiB, kAvail = 180 * GiB, kKeep = 18 * GiB;

}  // namespace

int main() {
    check(resident_by_default(true, 1, true, false, false, false, kCopy, kAvail, kKeep),
          "a lone card with the tier and no flags keeps the copy");
    check(!resident_by_default(true, 1, true, true, false, false, kCopy, kAvail, kKeep),
          "--resident-experts decides as given, not the default");
    check(!resident_by_default(true, 1, true, false, true, false, kCopy, kAvail, kKeep),
          "--resident-budget-gib decides as given, not the default");
    check(!resident_by_default(true, 1, true, false, false, true, kCopy, kAvail, kKeep),
          "--no-resident-experts turns it off");
    check(!resident_by_default(true, 2, true, false, false, false, kCopy, kAvail, kKeep),
          "a layer split keeps no copy by default");
    check(!resident_by_default(true, 1, false, false, false, false, kCopy, kAvail, kKeep),
          "without the AMD tier no copy by default");
    check(!resident_by_default(false, 1, true, false, false, false, kCopy, kAvail, kKeep),
          "not on another platform");
    check(!resident_by_default(true, 1, true, false, false, false, kCopy, kCopy + kKeep - 1, kKeep),
          "a copy that leaves under the RAM to keep free is not made");
    check(resident_by_default(true, 1, true, false, false, false, kCopy, kCopy + kKeep, kKeep),
          "a copy that leaves exactly the RAM to keep free is made");
    check(!resident_by_default(true, 1, true, false, false, false, kCopy, 64 * GiB, kKeep),
          "a copy larger than the RAM is not made");
    check(!resident_by_default(true, 1, true, false, false, false, 0, kAvail, kKeep), "nothing to copy, no copy");
    check(copy_pinned_share(88 * GiB, 88 * GiB) == 1.0, "a copy that holds every streamed expert prices them all pinned");
    check(copy_pinned_share(160 * GiB, 88 * GiB) == 1.0, "room past the streamed bytes still prices them all pinned");
    check(copy_pinned_share(44 * GiB, 88 * GiB) == 0.5, "a copy with room for half prices half pinned");
    check(copy_pinned_share(0, 88 * GiB) == 0.0, "no copy prices nothing pinned");
    check(copy_pinned_share(0, 0) == 1.0, "a card that streams nothing has nothing unpinned");
    // the decode's PCIe share beside the copy: 0.3 measured best on the box, the given one, or the env's
    const CopyShare d = copy_pcie_share(-1.0, nullptr);
    check(d.frac == 0.3 && !d.held, "no flag and no env: 0.3, and the opt-in balance may still refine it");
    const CopyShare g = copy_pcie_share(0.45, "0");
    check(g.frac == 0.45 && !g.held, "--pcie-frac decides as given, over the env");
    const CopyShare z = copy_pcie_share(-1.0, "0");
    check(z.frac == 0.0 && z.held, "STRATA_COPY_PCIE=0 holds the share at 0, also under STRATA_PCIE_BALANCE");
    const CopyShare h = copy_pcie_share(-1.0, "0.4");
    check(h.frac == 0.4 && h.held, "STRATA_COPY_PCIE=0.4 holds 0.4");
    const CopyShare pr = copy_pcie_share(-1.0, "probe");
    check(pr.frac < 0.0 && !pr.held, "STRATA_COPY_PCIE=probe leaves the share to the link probe's rule");
    const CopyShare e = copy_pcie_share(-1.0, "");
    check(e.frac == 0.3 && !e.held, "an empty value is unset");
    const CopyShare bad = copy_pcie_share(-1.0, "x");
    check(bad.frac == 0.3 && !bad.held, "a value that is no number is ignored");
    const CopyShare big = copy_pcie_share(-1.0, "2");
    check(big.frac == 1.0 && big.held, "a share above 1 is 1");
    const CopyShare neg = copy_pcie_share(-1.0, "-0.5");
    check(neg.frac == 0.0 && neg.held, "a share below 0 is 0");
    std::printf("resident_default_test: %d checks PASS\n", checks);
    return 0;
}
