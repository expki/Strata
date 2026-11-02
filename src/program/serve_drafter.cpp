// src/program/serve_drafter.cpp - the serve loop's drafters: MTP here, DFlash2 beside it.
#include "strata/program/serve_drafter.hpp"

#include "strata/core/mtp.hpp"

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

}  // namespace

std::unique_ptr<ServeDrafter> make_mtp_serve_drafter(core::MtpDrafter& mtp) {
    return std::make_unique<MtpServeDrafter>(mtp);
}

}  // namespace strata::program
