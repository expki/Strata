// src/program/serve_drafter.cpp - the serve loop's drafters: MTP here, DFlash2 beside it.
#include "strata/program/serve_drafter.hpp"

#include "strata/core/mtp.hpp"
#if defined(STRATA_HAVE_DFLASH2)
#include "strata/core/dflash2.hpp"
#include "strata/core/verify.hpp"
#endif

#include <algorithm>

namespace strata::program {

namespace {

class MtpServeDrafter final : public ServeDrafter {
public:
    explicit MtpServeDrafter(core::MtpDrafter& m) : m_(m) {}
    const char* name() const override { return "mtp"; }
    core::MtpDrafter* mtp() override { return &m_; }
    int max_window() const override { return m_.max_t(); }

    bool begin_request(int64_t resume, int64_t prompt_len, std::string& /*err*/) override {
        // the ring may hold cells past resume from a longer turn
        if (resume > 0) m_.kv_restore(resume);
        m_.set_prompt_len(prompt_len);
        return true;
    }

    bool draft(const RoundInput& in, DraftRound& out, std::string& err) override {
        // draft writes max_t - 1 slots each round, zeros past its last draft
        int n = 0;
        if (!m_.draft(in.T, in.outv, in.p, in.a, out.tok, err, out.prob, in.min_p, &n)) return false;
        out.n = n;
        out.has_prob = true;
        return true;
    }

    void report(FILE* f) const override {
        if (m_.rounds > 0)
            std::fprintf(f, "%-24s %.3f ms/round drafting (%lld rounds), MTP prompt %.1f ms, %.0f MiB of VRAM\n", "mtp",
                         m_.ms_draft / (double) m_.rounds, (long long) m_.rounds, m_.ms_prefill,
                         (double) m_.vram_bytes() / 1048576.0);
    }

private:
    core::MtpDrafter& m_;
};

#if defined(STRATA_HAVE_DFLASH2)
class Dflash2ServeDrafter final : public ServeDrafter {
public:
    Dflash2ServeDrafter(core::Dflash2Drafter& d, const core::Verifier& ver, int cap)
        : d_(d), ver_(ver), max_t_(std::clamp(cap, 1, 1 + core::Dflash2Drafter::kDrafts)) {}
    const char* name() const override { return "dflash2"; }
    std::vector<int> tap_layers() const override { return d_.tap_layers(); }
    int max_window() const override { return max_t_; }

    bool begin_request(int64_t resume, int64_t /*prompt_len*/, std::string& /*err*/) override {
        // the ring holds no saved window yet, so a resumed request drafts from the rows it reads
        d_.clear_ring(std::max<int64_t>(resume, 0));
        return true;
    }

    bool prompt_rows(const float* taps_dev, int64_t n, int64_t pos0, std::string& err) override {
        return d_.context(taps_dev, n, pos0, err);
    }

    bool window_rows(const core::Verifier& ver, int T, int64_t pos0, std::string& err) override {
        return d_.context(ver.tap_dev(), T, pos0, err);
    }

    bool draft(const RoundInput& in, DraftRound& out, std::string& err) override {
        const int rows = in.a + 1;
        if (rows > core::Dflash2Drafter::kRows) {
            err = "dflash2: a window committed more than 8 rows";
            return false;
        }
        if (!d_.draft(ver_.tap_dev(), rows, in.p, in.outv[in.a], out.tok, out.prob, err)) return false;
        // the drafts the window takes: up to max_t - 1, cut where the chance that all of them hold falls below min_p
        int n = 0;
        double chain = 1.0;
        while (n < max_t_ - 1) {
            chain *= (double) out.prob[n];
            if (in.min_p > 0.0f && chain < (double) in.min_p) break;
            ++n;
        }
        out.n = n;
        out.has_prob = false;
        return true;
    }

    bool save_window(int64_t upto, std::vector<uint8_t>& out, std::string& err) override { return d_.save_window(upto, out, err); }
    bool restore_window(const std::vector<uint8_t>& in, int64_t upto, const char* /*kind*/, std::string& err) override {
        return d_.restore_window(in, upto, err);
    }
    void clear_window(int64_t upto) override { d_.clear_ring(upto); }

    void report(FILE* f) const override {
        if (d_.rounds > 0)
            std::fprintf(f, "%-24s %.3f ms/round drafting (%lld rounds), %.0f MiB of VRAM\n", "dflash2",
                         d_.ms_draft / (double) d_.rounds, (long long) d_.rounds, (double) d_.vram_bytes() / 1048576.0);
    }

private:
    core::Dflash2Drafter& d_;
    const core::Verifier& ver_;
    const int max_t_;
};
#endif

}  // namespace

std::unique_ptr<ServeDrafter> make_mtp_serve_drafter(core::MtpDrafter& mtp) {
    return std::make_unique<MtpServeDrafter>(mtp);
}

#if defined(STRATA_HAVE_DFLASH2)
std::unique_ptr<ServeDrafter> make_dflash2_serve_drafter(core::Dflash2Drafter& d, const core::Verifier& ver, int cap) {
    return std::make_unique<Dflash2ServeDrafter>(d, ver, cap);
}
#endif

}  // namespace strata::program
