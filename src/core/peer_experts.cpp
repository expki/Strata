// src/core/peer_experts.cpp - see include/strata/core/peer_experts.hpp.
#include "strata/core/peer_experts.hpp"

#include "strata/core/device.hpp"   // primary_device() (STRATA_PRIMARY_DEVICE)
#include "strata/core/peer_adapt.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <vector>

namespace strata::core {

static bool g_peer_portable = false;
void set_peer_portable(bool on) { g_peer_portable = on; }
bool peer_portable() { return g_peer_portable; }

namespace {

constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t CAP = strata::kernels::cpu::MAXT * 10;   // entries per layer: MAXT tokens x top-10

using Meta = strata_xtier_meta;         // one block, uploaded with one copy per layer (the AMD tier library takes the same)
static_assert(STRATA_XTIER_CAP == CAP, "the AMD tier library's plan holds a layer's entries");

/// Switches the calling thread to `dev` and back.  The engine's own thread lives on device 0.
struct On {
    int prev = 0;
    explicit On(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
    ~On() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
};

/// #2666: `cudaDeviceCanAccessPeer` said yes and `cudaDeviceEnablePeerAccess` succeeded, and on a pair of GPUs behind a host bridge
/// (2x RTX A4000, "PHB" in nvidia-smi topo) `cudaMemcpyPeerAsync` then returned success having moved nothing: the peer's input rows
/// were zeros and the rows it sent back never arrived, so the first prompt's MoE output was garbage and the logits NaN.  Without
/// peer access enabled the same copies are right (the driver stages them).  So the pair is asked once, with a small copy each way,
/// whether the direct path really carries data.  true = both ways exact (or nothing could be tested: the old trust is kept).
bool p2p_copies_data(int device) {
    constexpr size_t kN = 1 << 14;   // floats
    std::vector<float> h(kN), back(kN);
    for (size_t i = 0; i < kN; ++i) h[i] = 1.0f + (float) ((i * 2654435761u) % 4093u);
    float* d[2] = {nullptr, nullptr};
    const int dev[2] = {0, device};
    bool ok = true, tested = true;
    for (int k = 0; k < 2 && tested; ++k) {
        On on(dev[k]);
        if (cudaMalloc((void**) &d[k], kN * 4) != cudaSuccess) { cudaGetLastError(); tested = false; }
    }
    for (int dir = 0; dir < 2 && tested && ok; ++dir) {   // 0: device 0 -> peer, 1: peer -> device 0
        const int from = dir == 0 ? 0 : 1, to = 1 - from;
        {
            On on(dev[from]);
            if (cudaMemcpy(d[from], h.data(), kN * 4, cudaMemcpyHostToDevice) != cudaSuccess) { tested = false; break; }
        }
        {
            On on(dev[to]);
            if (cudaMemset(d[to], 0, kN * 4) != cudaSuccess) { tested = false; break; }
        }
        if (cudaMemcpyPeer(d[to], dev[to], d[from], dev[from], kN * 4) != cudaSuccess) { tested = false; break; }
        {
            On on(dev[to]);
            if (cudaMemcpy(back.data(), d[to], kN * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { tested = false; break; }
        }
        ok = std::memcmp(back.data(), h.data(), kN * 4) == 0;
    }
    for (int k = 0; k < 2; ++k)
        if (d[k] != nullptr) { On on(dev[k]); cudaFree(d[k]); }
    if (!tested) { cudaGetLastError(); return true; }
    return ok;
}

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("peer experts: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

}  // namespace

PeerExperts::~PeerExperts() { close(); }

void PeerExperts::close() {
    xdev_.close();   // the library waits for the AMD card's streams and frees it, unless a wait for the card timed out
    xslot_.clear();
    xbytes_ = 0;
    free_.clear();
    if (device_ < 0) return;
    {
        On on(device_);
        if (stream_) cudaStreamSynchronize(stream_);
        if (refill_) cudaStreamSynchronize(refill_);
        cache_.close();
        if (d_x_) cudaFree(d_x_);
        if (d_out_) cudaFree(d_out_);
        if (d_meta_) cudaFree(d_meta_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (h_x_) cudaFreeHost(h_x_);
        if (h_out_) cudaFreeHost(h_out_);
        if (h_meta_) cudaFreeHost(h_meta_);
        if (refill_ev_) cudaEventDestroy(refill_ev_);
        if (stream_) cudaStreamDestroy(stream_);
        if (refill_) cudaStreamDestroy(refill_);
    }
    d_x_ = d_out_ = h_x_ = h_out_ = nullptr;
    d_meta_ = h_meta_ = d_scratch_ = nullptr;
    d_q8_ = nullptr;
    d_sx_ = d_sh_ = nullptr;
    stream_ = refill_ = nullptr;
    refill_ev_ = nullptr;
    device_ = -1;
}

bool PeerExperts::open(int device, const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
                       ExpertSource& src, int64_t n_layers, int64_t n_expert, int reserve_mib, int64_t max_slots,
                       std::string& err) {
    close();
    int count = 0;
    if (!ck(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err)) return false;
    if (device < 1 || device >= count) {
        err = "peer experts: CUDA device " + std::to_string(device) + " is not visible (" + std::to_string(count) +
              " devices; set CUDA_VISIBLE_DEVICES)";
        return false;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    src_ = &src;
    res_.assign((size_t) (n_layers * n_expert), kNotResident);
    {   // direct access both ways (NVLink or another P2P path): the prompt path copies activations and results over it
        int a = 0, b = 0;
        cudaDeviceCanAccessPeer(&a, primary_device(), device);
        cudaDeviceCanAccessPeer(&b, device, primary_device());
        if (const char* np = std::getenv("STRATA_PEER_NO_P2P"); np != nullptr && std::atoi(np) != 0) a = 0;   // test knob: pretend the pair has no P2P
        p2p_ = a && b;
        if (p2p_) {
            On on0(primary_device());
            cudaError_t e0 = cudaDeviceEnablePeerAccess(device, 0);
            if (e0 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e0 = cudaSuccess; }
            On on1(device);
            cudaError_t e1 = cudaDeviceEnablePeerAccess(primary_device(), 0);
            if (e1 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e1 = cudaSuccess; }
            p2p_ = e0 == cudaSuccess && e1 == cudaSuccess;
            // recommend, never force: the direct path is used only if a test copy each way arrives exact (#2666)
            if (p2p_ && !p2p_copies_data(device)) {
                std::fprintf(stderr, "strata: peer GPU %d: peer access is enabled but a test copy between the two GPUs arrived wrong "
                                     "(a host bridge between them, no real P2P): the prompt path takes the host route instead "
                                     "(STRATA_PEER_NO_P2P=1 does that by hand)\n", device);
                { On q0(0); cudaDeviceDisablePeerAccess(device); cudaGetLastError(); }
                { On q1(device); cudaDeviceDisablePeerAccess(0); cudaGetLastError(); }
                p2p_ = false;
            }
        }
    }
    On on(device);
    device_ = device;
    // scratch first, so the slots take what is really left
    int64_t ff = strata::kernels::cpu::FF;
    if (lay.native)
        for (const auto& f : lay.fmt) ff = std::max<int64_t>(ff, f.n_ff);
    const size_t scratch = std::max<size_t>((size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, ff),
                                            strata::kernels::native_expert_scratch_bytes(CAP, ff));
    const bool alloc_ok =
        ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err) &&
        ck(cudaStreamCreateWithFlags(&refill_, cudaStreamNonBlocking), "refill stream", err) &&
        ck(cudaEventCreateWithFlags(&refill_ev_, cudaEventDisableTiming), "event", err) &&
        ck(cudaHostAlloc((void**) &h_x_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err) &&
        ck(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err) &&
        ck(cudaHostAlloc(&h_meta_, sizeof(Meta), cudaHostAllocPortable | cudaHostAllocMapped), "plan staging", err) &&
        ck(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err) &&
        ck(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err) &&
        ck(cudaMalloc(&d_meta_, sizeof(Meta)), "plan", err) &&
        ck(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activations", err) &&
        ck(cudaMalloc(&d_scratch_, scratch), "scratch", err);
    if (!alloc_ok) { close(); return false; }
    if (strata::kernels::cpu::hadamard_spec().any()) {   // APR: this device's copies of the sign vectors
        d_sx_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) H), (int) H);
        d_sh_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) ff), (int) ff);
    }

    // the pairs the primary does not hold, in rank order, as many as fit
    size_t free_b = 0, total_b = 0;
    if (!ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err)) { close(); return false; }
    const uint64_t reserve = (uint64_t) std::max(reserve_mib, 128) << 20;
    const uint64_t budget = free_b > reserve ? free_b - reserve : 0;
    std::vector<std::pair<int32_t, int32_t>> pick;
    std::vector<int64_t> sizes;
    uint64_t used = 0;
    for (const auto& pr : ranked) {
        if (primary.slot_of(pr.first, pr.second) >= 0) continue;
        const uint64_t b = lay.blob_bytes(pr.first);
        const uint64_t b256 = lay.native ? (b + 255) / 256 * 256 : lay.max_blob;
        if (used + b256 > budget) break;
        if (max_slots > 0 && (int64_t) pick.size() >= max_slots) break;
        used += b256;
        pick.push_back(pr);
        sizes.push_back((int64_t) b);
    }
    if (pick.empty()) { err = "peer experts: no room or no expert left for the peer"; close(); return false; }
    const bool opened = lay.native ? cache_.open_sized(sizes, n_layers, n_expert, err)
                                   : cache_.open((int64_t) pick.size(), n_layers, n_expert, (int64_t) lay.max_blob, err);
    if (!opened) { err = "peer experts: " + err; close(); return false; }
    // open() zeroes the arena with cudaMemset on the legacy stream, which is not ordered against the non-blocking
    // refill stream: wait for it, or the zeroing can land on top of the fills
    if (!ck(cudaDeviceSynchronize(), "arena zeroing", err)) { close(); return false; }
    // mapped reads: advise the next kAhead pairs so their reads overlap
    constexpr size_t kAhead = 256;
    const bool advise = src.advise_pairs(pick.data(), (int64_t) std::min(kAhead, pick.size()));
    for (size_t i = 0; i < pick.size(); ++i) {
        const auto& pr = pick[i];
        if (advise && i + kAhead < pick.size()) (void) src.advise_pairs(&pick[i + kAhead], 1);
        const int32_t slot = cache_.admit(pr.first, pr.second);
        const uint8_t* b = src.blob(pr.first, pr.second);
        if (slot < 0 || b == nullptr || !cache_.fill_slot(slot, b, refill_, err, (int64_t) lay.blob_bytes(pr.first))) {
            err = "peer experts: fill failed: " + err;
            close();
            return false;
        }
        src.note_async_read(b, refill_);   // #1237
        res_[(size_t) (pr.first * n_expert + pr.second)] = slot;
    }
    if (!ck(cudaStreamSynchronize(refill_), "fill", err)) { close(); return false; }
    const auto& f0 = pick.front();
    if (!cache_.verify_slot(res_[(size_t) (f0.first * n_expert + f0.second)], src.blob(f0.first, f0.second), err,
                            (int64_t) lay.blob_bytes(f0.first))) {
        err = "peer experts: " + err;
        close();
        return false;
    }
    resident_ = (int64_t) pick.size();
    return true;
}

bool PeerExperts::open_foreign(const xtier::HipCard& card, const std::vector<std::pair<int32_t, int32_t>>& ranked,
                               const std::function<bool(int32_t, int32_t)>& held, ExpertSource& src, int64_t n_layers,
                               int64_t n_expert, int reserve_mib, int64_t max_slots, std::string& err) {
    close();
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native) { err = "peer experts: only native packs are supported"; return false; }
    for (const auto& f : lay.fmt) {   // what the library's decode cannot express would give wrong rows, not an error
        if (f.tail_off) { err = "peer experts: the AMD tier is not for NVFP4 experts yet"; return false; }
        if (f.had_gu || f.had_d) { err = "peer experts: the AMD tier is not for a Hadamard-folded (APR) pack yet"; return false; }
        if (f.n_embd > STRATA_XTIER_MAX_EMBD || f.n_ff > STRATA_XTIER_MAX_FF) {
            err = "peer experts: experts " + std::to_string(f.n_embd) + " wide with " + std::to_string(f.n_ff) +
                  " hidden rows are more than the AMD tier library takes (" + std::to_string(STRATA_XTIER_MAX_EMBD) + " and " +
                  std::to_string(STRATA_XTIER_MAX_FF) + ")";
            return false;
        }
    }
    // main set HIP_VISIBLE_DEVICES already when it was unset (xtier::claim_hip_card), and a set value, even an empty one, is honoured
    if (xtier::load(err) == nullptr) { err = "peer experts: " + err; return false; }
    const char* tv = std::getenv("STRATA_XTIER_TIMING");
    xtiming_ = tv != nullptr && std::atoi(tv) != 0;
    if (!xdev_.open(card.open, err)) {
        const char* hv = std::getenv("HIP_VISIBLE_DEVICES");
        const std::string hv_text = card.engine_set ? std::to_string(card.n) + " set by the engine"
                                    : hv == nullptr ? std::string("unset") : hv[0] == '\0' ? std::string("\"\"") : std::string(hv);
        const char* cv = std::getenv("CUDA_VISIBLE_DEVICES");
        err = "peer experts: AMD card hip:" + std::to_string(card.n) + ": " + err + " (HIP_VISIBLE_DEVICES=" + hv_text +
              ", CUDA_VISIBLE_DEVICES=" + (cv != nullptr ? cv : "unset") + ")";
        return false;
    }
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    src_ = &src;
    res_.assign((size_t) (n_layers * n_expert), kNotResident);
    // the pairs no CUDA card holds, in rank order, as many as fit (the library's buffers are already taken)
    uint64_t free_b = 0, total_b = 0;
    if (!xdev_.mem_info(free_b, total_b, err)) { err = "peer experts: " + err; close(); return false; }
    const uint64_t reserve = (uint64_t) std::max(reserve_mib, 128) << 20;
    const uint64_t budget = free_b > reserve ? free_b - reserve : 0;
    std::vector<std::pair<int32_t, int32_t>> pick;
    std::vector<uint64_t> sizes;
    uint64_t used = 0;
    for (const auto& pr : ranked) {
        if (held(pr.first, pr.second)) continue;
        const uint64_t b = lay.blob_bytes(pr.first);
        const uint64_t b256 = (b + 255) / 256 * 256;
        if (used + b256 > budget) break;
        if (max_slots > 0 && (int64_t) pick.size() >= max_slots) break;
        used += b256;
        pick.push_back(pr);
        sizes.push_back(b);
    }
    if (pick.empty()) { err = "peer experts: no room or no expert left for the peer"; close(); return false; }
    if (!xdev_.arena(sizes, xslot_, err)) { err = "peer experts: " + err; close(); return false; }
    xbytes_ = used;
    constexpr size_t kAhead = 256;   // mapped reads: advise the next kAhead pairs so their reads overlap
    const bool advise = src.advise_pairs(pick.data(), (int64_t) std::min(kAhead, pick.size()));
    for (size_t i = 0; i < pick.size(); ++i) {
        const auto& pr = pick[i];
        if (advise && i + kAhead < pick.size()) (void) src.advise_pairs(&pick[i + kAhead], 1);
        const uint8_t* b = src.blob(pr.first, pr.second);
        if (b == nullptr || !xdev_.fill((int64_t) i, b, sizes[i], err)) {
            err = "peer experts: fill failed: " + err;
            close();
            return false;
        }
        res_[(size_t) (pr.first * n_expert + pr.second)] = (int32_t) i;
    }
    if (!xdev_.refill_record(err) || xdev_.refill_done(true, &err) != 1) { err = "peer experts: fill: " + err; close(); return false; }
    std::vector<uint8_t> back((size_t) sizes[0]);   // slot 0 read back, as the CUDA tier checks its first slot
    if (!xdev_.read(0, back.data(), sizes[0], err)) { err = "peer experts: " + err; close(); return false; }
    const uint8_t* b0 = src.blob(pick[0].first, pick[0].second);
    if (b0 == nullptr || std::memcmp(back.data(), b0, (size_t) sizes[0]) != 0) {
        err = "peer experts: slot 0 does not hold its expert after the fill";
        close();
        return false;
    }
    resident_ = (int64_t) pick.size();
    return true;
}

bool PeerExperts::launch(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                         const int32_t* kind, std::string& err, float* out) {
    static const bool direct_env = [] { const char* v = std::getenv("STRATA_PEER_DIRECT"); return v == nullptr || std::atoi(v) != 0; }();
    const bool direct = direct_env && out != nullptr && !foreign();   // an AMD card does not write into the host rows
    if (foreign() && launched_rows_ > 0) {   // a failed window left its rows untaken, and the library holds one decode at a time
        const float* rows = nullptr;
        std::string e;
        if (!xdev_.decode_wait(&rows, e)) {   // it cannot be drained: no expert is planned for the tier from here on
            std::fprintf(stderr, "strata generate: the AMD tier stopped answering and is turned off\n");
            res_.assign(res_.size(), kNotResident);
            pending_.clear();
            free_.clear();
            launched_rows_ = 0;
            err = "peer experts: " + e;
            return false;
        }
    }
    launched_direct_ = false;
    launched_rows_ = 0;
    row_of_.clear();
    const auto t0 = xtiming_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (xtiming_) ++xt_calls_;
    const int64_t n = n_tok * k;
    if (n > CAP) { err = "peer experts: window too large"; return false; }
    Meta& m = foreign() ? xmeta_ : *(Meta*) h_meta_;
    int groups = 0, rows = 0;
    // distinct experts in routing order; each one's entries become compact rows
    for (int64_t i = 0; i < n; ++i) {
        if (kind[i] != 2) continue;
        bool seen = false;
        for (int64_t j = 0; j < i; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) { seen = true; break; }
        if (seen) continue;
        const int32_t slot = res_[(size_t) (layer * n_expert_ + ids[i])];
        if (slot < 0) { err = "peer experts: an entry was planned for the peer but is not resident"; return false; }
        m.ptr[groups] = foreign() ? xslot_[(size_t) slot] : (unsigned long long) cache_.device_slot(slot);
        m.start[groups] = rows;
        for (int64_t j = i; j < n; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) {
                m.dst[rows] = rows;
                m.tok[rows] = (int32_t) (j / k);
                m.out_row[rows] = (int32_t) j;
                row_of_.push_back((int32_t) j);
                ++rows;
            }
        ++groups;
        ++experts_;
    }
    if (groups == 0) return true;
    m.start[groups] = rows;
    m.count[0] = groups;
    m.count[1] = rows;
    entries_ += rows;
    launched_rows_ = rows;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (foreign()) {   // the library copies x and the plan before it returns
        const auto& f = lay.fmt[(size_t) layer];
        const strata_xtier_fmt xf = {f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_off, f.down_off, f.bytes};
        const auto t1 = xtiming_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (xdev_.decode(xf, x, n_tok, m, err)) {
            if (xtiming_) {
                xt_end_ = std::chrono::steady_clock::now();
                xt_[0] += std::chrono::duration<double, std::milli>(xt_end_ - t0).count();
                xt_[1] += std::chrono::duration<double, std::milli>(xt_end_ - t1).count();
                xt_rows_ += rows;
                xt_groups_ += groups;
            }
            return true;
        }
        err = "peer experts: " + err;
        launched_rows_ = 0;
        return false;
    }
    // direct: x is the verifier's pinned (portable) doorbell row block - copied from where it is
    if (!direct) std::memcpy(h_x_, x, (size_t) (n_tok * H) * sizeof(float));
    On on(device_);
    const cudaStream_t s = stream_;
    if (!ck(cudaMemcpyAsync(d_x_, direct ? x : h_x_, (size_t) (n_tok * H) * sizeof(float), cudaMemcpyHostToDevice, s), "input", err) ||
        !ck(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(Meta), cudaMemcpyHostToDevice, s), "plan", err))
        return false;
    Meta* dm = (Meta*) d_meta_;
    if (lay.native) {
        const auto& f = lay.fmt[(size_t) layer];
        // APR: Hadamard-folded gate/up take H (s * x); d_x_ is this device's private copy, rotated in place
        if (f.had_gu) strata::kernels::hadamard_rows(d_x_, d_x_, n_tok, H, f.had_block, d_sx_, s);
        strata::kernels::quantize_q8_1_rows(d_x_, n_tok, H, d_q8_, s);
        auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        if (f.had_d) { L.had_block = f.had_block; L.d_signs = d_sh_; }
        strata::kernels::native_expert_grouped(L, dm->ptr, dm->start, dm->count, dm->dst, dm->tok, groups, rows, d_q8_,
                                               d_scratch_, d_out_, s);
    } else {
        err = "peer experts: only native packs are supported";
        return false;
    }
    if (direct) {   // the rows straight into the host `out` (zero-copy, coalesced float4 writes)
        try {
            strata::kernels::scatter_rows_f32(d_out_, out, dm->out_row, rows, H, s);
        } catch (const std::exception& e) { err = std::string("peer experts: scatter: ") + e.what(); return false; }
        launched_direct_ = true;
        return true;
    }
    return ck(cudaMemcpyAsync(h_out_, d_out_, (size_t) rows * H * sizeof(float), cudaMemcpyDeviceToHost, s),
              "results", err);
}

bool PeerExperts::finish(float* out, std::string& err) {
    if (launched_rows_ == 0) return true;
    const auto t0 = std::chrono::steady_clock::now();
    const float* rows = h_out_;
    if (foreign()) {   // the library spins on its own decode stream
        if (!xdev_.decode_wait(&rows, err)) { err = "peer experts: " + err; return false; }
    } else {
        On on(device_);
        // spin on the stream (a blocking sync would sleep the pool thread and wake it late)
        cudaError_t e;
        while ((e = cudaStreamQuery(stream_)) == cudaErrorNotReady) {
        }
        if (!ck(e, "finish", err)) return false;
    }
    const auto t1 = std::chrono::steady_clock::now();
    ms_wait += std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (launched_direct_) { launched_rows_ = 0; return true; }
    for (size_t r = 0; r < row_of_.size(); ++r)
        std::memcpy(out + (size_t) row_of_[r] * H, rows + r * H, (size_t) H * sizeof(float));
    launched_rows_ = 0;
    if (xtiming_) {
        const auto t2 = std::chrono::steady_clock::now();
        xt_[2] += std::chrono::duration<double, std::milli>(t0 - xt_end_).count();
        xt_[3] += std::chrono::duration<double, std::milli>(t1 - t0).count();
        xt_[4] += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    return true;
}

void PeerExperts::timing_start() {
    if (!xtiming_) return;
    std::fill(std::begin(xt_), std::end(xt_), 0.0);
    xt_calls_ = xt_rows_ = xt_groups_ = 0;
    double card[3] = {};
    (void) xdev_.timing(card);   // the card's sums start over too
}

void PeerExperts::timing_print() {
    if (!xtiming_ || xt_calls_ == 0) return;
    double card[3] = {};
    const int n = xdev_.timing(card);
    const double c = (double) xt_calls_;
    std::fprintf(stderr, "strata decode AMD tier timing: %lld layer-windows, %.2f entries in %.2f experts each; ms per layer-window: "
                         "launch %.3f (of it the library's decode call %.3f), the pool between launch and finish %.3f, finish "
                         "wait %.3f, copy out %.3f; the card (%d decodes timed) copies in %.3f, kernels %.3f, copy out %.3f\n",
                 (long long) xt_calls_, (double) xt_rows_ / c, (double) xt_groups_ / c, xt_[0] / c, xt_[1] / c, xt_[2] / c,
                 xt_[3] / c, xt_[4] / c, n, card[0] / c, card[1] / c, card[2] / c);
    timing_start();
}

bool PeerExperts::adapt(const float* usage, const int32_t* res0, int max_swaps, std::string& err) {
    if (!pending_.empty() || max_swaps <= 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const std::vector<PeerSwap> swaps = peer_adapt_choose(usage, res0, res_.data(), free_, n_layers_, n_expert_, max_swaps);
    if (swaps.empty()) return true;
    std::optional<On> on;
    if (!foreign()) on.emplace(device_);
    for (const PeerSwap& s : swaps) {
        const size_t in = (size_t) (s.layer * n_expert_ + s.in);
        std::vector<int32_t>* fr = s.out < 0 ? &free_[(size_t) s.layer] : nullptr;
        int32_t* victim = fr != nullptr ? nullptr : &res_[(size_t) (s.layer * n_expert_ + s.out)];
        const int32_t slot = fr != nullptr ? fr->back() : *victim;
        const uint8_t* b = src_->blob(s.layer, s.in);
        if (slot < 0 || b == nullptr) continue;
        if (foreign()) {
            if (!xdev_.fill(slot, b, lay.blob_bytes(s.layer), err)) { err = "peer experts: refill: " + err; return false; }
        } else if (!ck(cudaMemcpyAsync(cache_.device_slot(slot), b, (size_t) lay.blob_bytes(s.layer), cudaMemcpyHostToDevice,
                                       refill_), "refill", err))
            return false;
        if (fr != nullptr) fr->pop_back();
        else *victim = kNotResident;           // evicted now: the CPU computes it meanwhile
        pending_.emplace_back((int32_t) in, slot);
        ++swaps_;
    }
    if (foreign()) {
        if (xdev_.refill_record(err)) return true;
        err = "peer experts: refill event: " + err;
        return false;
    }
    return ck(cudaEventRecord(refill_ev_, refill_), "refill event", err);
}

void PeerExperts::apply_pending(bool wait) {
    if (pending_.empty()) return;
    if (foreign()) {   // a decode reads a slot only once the library says its fill is done
        std::string e;
        const int done = xdev_.refill_done(wait, &e);
        if (done < 0) {   // the fills failed: their slots stay free, and the next adapt fills them again
            std::fprintf(stderr, "strata generate: the AMD tier's refill failed: %s\n", e.c_str());
            if (free_.empty()) free_.resize((size_t) n_layers_);
            for (const auto& [i, slot] : pending_) free_[(size_t) (i / n_expert_)].push_back(slot);
            pending_.clear();
            return;
        }
        if (done != 1) return;
    } else {
        On on(device_);
        if (wait) cudaEventSynchronize(refill_ev_);
        else if (cudaEventQuery(refill_ev_) != cudaSuccess) return;
    }
    for (const auto& [i, slot] : pending_) res_[(size_t) i] = slot;
    pending_.clear();
}

// Only adapt calls it, between windows, so no decode reads the slot while it changes hands.
void PeerExperts::release(int64_t layer, int64_t expert) {
    const int32_t i = (int32_t) (layer * n_expert_ + expert);
    int32_t slot = res_[(size_t) i];
    res_[(size_t) i] = kNotResident;
    for (size_t k = 0; slot < 0 && k < pending_.size(); ++k)   // still filling: its slot is freed the same way
        if (pending_[k].first == i) {
            slot = pending_[k].second;
            pending_.erase(pending_.begin() + (ptrdiff_t) k);
        }
    if (slot < 0) return;
    if (free_.empty()) free_.resize((size_t) n_layers_);
    free_[(size_t) layer].push_back(slot);
}

}  // namespace strata::core
