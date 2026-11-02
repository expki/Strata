// include/strata/program/serve_drafter.hpp - the serve loop's model drafter, MTP or DFlash2, behind one interface.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {
class Dflash2Drafter;
class MtpDrafter;
class Verifier;
}

namespace strata::program {

// One round's drafts and, when the drafter has them, their probabilities.
struct DraftRound { int n = 0; int32_t tok[8] = {}; float prob[8] = {}; bool has_prob = false; };

// The verify window just committed: p is its start before the advance, a its accepted drafts.
struct RoundInput {
    const core::Verifier& ver; int T; const int32_t* outv; int64_t p; int a;
    float min_p = 0.0f;   // the request's --spec-min-p, where the drafter can stop early
};

class ServeDrafter {
public:
    virtual ~ServeDrafter() = default;
    virtual const char* name() const = 0;                                   // "mtp" | "dflash2"
    virtual core::MtpDrafter* mtp() { return nullptr; }                      // MTP-only features
    virtual std::vector<int> tap_layers() const { return {}; }               // verifier and prompt taps
    virtual int max_window() const = 0;                                      // widest T this drafter fills
    virtual bool begin_request(int64_t resume, int64_t prompt_len, std::string& err) = 0;
    virtual bool prompt_rows(const float* /*taps_dev*/, int64_t /*n*/, int64_t /*pos0*/, std::string& /*err*/) { return true; }
    virtual bool window_rows(const core::Verifier& /*ver*/, int /*T*/, int64_t /*pos0*/, std::string& /*err*/) { return true; }
    virtual bool draft(const RoundInput& in, DraftRound& out, std::string& err) = 0;
    virtual bool save_window(int64_t /*upto*/, std::vector<uint8_t>& out, std::string& /*err*/) { out.clear(); return true; }
    virtual bool restore_window(const std::vector<uint8_t>& /*in*/, int64_t /*upto*/, const char* /*kind*/, std::string& /*err*/) {
        return true;
    }
    virtual void clear_window(int64_t /*upto*/) {}                           // drop the context, rows from upto on come next
    virtual size_t window_max_bytes() const { return 0; }                    // a full saved window, 0 when it keeps none
    virtual int64_t window_end() const { return -1; }                        // where its context rows end, -1 without a window
    virtual void report(FILE* f) const = 0;                                  // rounds, ms a round, VRAM
};

std::unique_ptr<ServeDrafter> make_mtp_serve_drafter(core::MtpDrafter& mtp);
#if defined(STRATA_HAVE_DFLASH2)
// max_window() is min(cap, 1 + kDrafts), with cap already at most --spec
std::unique_ptr<ServeDrafter> make_dflash2_serve_drafter(core::Dflash2Drafter& d, const core::Verifier& ver, int cap);
#endif

}  // namespace strata::program
