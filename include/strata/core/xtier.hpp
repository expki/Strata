// The CUDA engine's side of the AMD tier: the --peer-device hip:N value and the library that drives the card.
#pragma once

#include "strata/core/xtier_abi.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core::xtier {

// The --peer-device value: a second CUDA card (N >= 1) or an AMD card (hip:N, N >= 0), never both.
struct PeerSpec {
    int cuda = -1;
    int hip = -1;
};

// Accepts "N" or "hip:N", and "0" as no peer, and says why when it refuses. The AMD and Mac engines take a card number only.
bool parse_peer_device(const std::string& v, PeerSpec& out, std::string& err);

// Where the card of hip:N is for HIP, decided once in main before the engine reads its environment from other threads.
struct HipCard {
    int n = -1;                 // N of hip:N
    int open = -1;              // the HIP ordinal the library opens
    bool engine_set = false;    // the engine set HIP_VISIBLE_DEVICES=N itself
};

// HIP hides the AMD cards CUDA_VISIBLE_DEVICES leaves out, so an unset HIP_VISIBLE_DEVICES becomes N and the card is 0 then. A set value, even an empty one, is honoured.
HipCard claim_hip_card(int n);

// STRATA_XTIER_LIB if it is set, else the library beside the engine.
std::string library_path();

// STRATA_XTIER_WAIT_MS (default 30000): how long the engine waits for the AMD tier before it fails the request.
uint64_t wait_ms();

// One function pointer per strata_xtier_* export.
struct Api {
    int (*abi)() = nullptr;
    void* (*open)(int, char*, size_t) = nullptr;
    void (*close)(void*) = nullptr;
    int (*mem_info)(void*, uint64_t*, uint64_t*) = nullptr;
    int (*reg)(void*, void*, uint64_t, void**) = nullptr;
    int (*unreg)(void*, void*) = nullptr;
    const char* (*last_error)(void*) = nullptr;
    int (*flag_echo)(void*, const uint32_t*, uint32_t*, uint32_t*, uint32_t, uint64_t) = nullptr;
    int (*sync)(void*) = nullptr;
    void* (*host_alloc)(void*, uint64_t, int) = nullptr;
    void (*host_free)(void*, void*) = nullptr;
    int (*probe)(void*, int, void*, uint64_t, uint32_t, uint64_t*, float*) = nullptr;
    int (*arena)(void*, const uint64_t*, int64_t, uint64_t*) = nullptr;
    int (*fill)(void*, int64_t, const void*, uint64_t) = nullptr;
    int (*read)(void*, int64_t, void*, uint64_t) = nullptr;
    int (*refill_record)(void*) = nullptr;
    int (*refill_done)(void*, int) = nullptr;
    int (*decode)(void*, const strata_xtier_fmt*, const float*, int64_t, const strata_xtier_meta*) = nullptr;
    int (*decode_out)(void*, const strata_xtier_fmt*, const float*, int64_t, const strata_xtier_meta*, void*) = nullptr;
    int (*decode_counts)(void*, uint64_t*) = nullptr;
    int (*decode_wait)(void*, const float**, uint64_t) = nullptr;
    int (*timing)(void*, double*) = nullptr;
    int (*prefill_open)(void*, int64_t, int64_t, uint64_t, uint64_t, int64_t, uint64_t) = nullptr;
    int (*prefill_layer)(void*, const strata_xtier_prefill_args*) = nullptr;
    int (*prefill_stats)(void*, strata_xtier_prefill_record*, int64_t, int64_t*) = nullptr;
};

// Opens the library at path, or returns null with the reason in err. Not cached and never unloaded.
const Api* load_from(const std::string& path, std::string& err);

// load_from(library_path()) once for the process.
const Api* load(std::string& err);

// One AMD card, opened through the library.
class Device {
public:
    Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device();

    bool open(int hip_device, std::string& err);
    void close();
    bool valid() const { return h_ != nullptr; }
    bool mem_info(uint64_t& free_b, uint64_t& total_b, std::string& err);
    // Maps engine host memory into the card's address space.
    bool register_host(void* host, uint64_t bytes, void** hip_dev, std::string& err);
    void unregister_host(void* host);
    // Queues a wait for *in_hip >= seq (or the timeout, then *err_hip = 1) and then *out_hip = seq on the card, and returns at once.
    bool flag_echo(const uint32_t* in_hip, uint32_t* out_hip, uint32_t* err_hip, uint32_t seq, uint64_t timeout_ns, std::string& err);
    // Waits for the work queued on the card.
    bool sync(std::string& err);
    // Pinned host memory the card's runtime owns, mapped and portable, and coherent when asked, or null with the reason in err.
    void* host_alloc(uint64_t bytes, std::string& err, bool coherent = false);
    void host_free(void* host);
    // A probe, and it waits: op 0 writes the seed's pattern into host, op 1 counts the wrong words, op 2 times a copy into host, op 3 a copy out of host.
    bool probe(int op, void* host, uint64_t bytes, uint32_t seed, uint64_t& result, float& ms, std::string& err);

    // One device allocation cut into slots of these sizes, each 256 byte aligned, and each slot's address in slot_addr. Once per card.
    bool arena(const std::vector<uint64_t>& slot_bytes, std::vector<uint64_t>& slot_addr, std::string& err);
    // Queues a copy into a slot on the refill stream and returns at once. Until refill_done says 1, host stays valid and no decode reads the slot.
    bool fill(int64_t slot, const void* host, uint64_t bytes, std::string& err);
    // Copies a slot out after the queued fills and waits, for checking a fill.
    bool read(int64_t slot, void* host, uint64_t bytes, std::string& err);
    // Marks the fills queued so far.
    bool refill_record(std::string& err);
    // 1 when the fills before the last record are done, 0 when not yet, -1 on error (the reason in err when it is given). With wait it blocks.
    int refill_done(bool wait, std::string* err = nullptr);
    // Queues one layer's expert rows and returns at once, x and m are copied before it returns.
    bool decode(const strata_xtier_fmt& f, const float* x, int64_t n_tok, const strata_xtier_meta& m, std::string& err);
    // As decode, but row r lands at row m.out_row[r] of out_host (registered with register_host), so decode_wait's rows are not needed.
    bool decode_out(const strata_xtier_fmt& f, const float* x, int64_t n_tok, const strata_xtier_meta& m, void* out_host, std::string& err);
    // decode_out's launches since open: step by step, captured, replayed, graphs dropped, and the graphs held now.
    bool decode_counts(uint64_t c[5]);
    // Spins for the queued decode and gives its pinned rows (m.count[1] rows of n_embd floats, valid until the next decode), or fails after STRATA_XTIER_WAIT_MS (default 30000) and the decode stays pending until a later wait or close.
    bool decode_wait(const float** rows, std::string& err);
    // STRATA_XTIER_TIMING=1: the card's ms in copies in, kernels and copy out summed since the last call, and the count of decodes, or -1 (also for a library without the export).
    int timing(double ms[3]);
    // Sizes the prompt buffers on the card for chunks of up to T_max tokens, and the ring for stream_slots streamed experts of blob_max bytes, or grows them.
    bool prefill_open(int64_t T_max, int64_t N, uint64_t gu_max, uint64_t d_max, int64_t stream_slots, uint64_t blob_max, std::string& err);
    // The tokens of the largest chunk the prompt buffers were opened for, 0 when none.
    int64_t prompt_tokens() const { return prompt_tokens_; }
    // The slots of the stream ring the card has (0: none), and their bytes.
    int64_t stream_slots() const { return stream_slots_; }
    uint64_t stream_blob() const { return stream_blob_; }
    // A stage that may stream to the card joins at start with its weight (its RAM experts) and gets its place among them, which picks its slots of the stream ring.
    int stream_join(int64_t weight) {
        const std::lock_guard<std::mutex> lk(stream_mu_);
        stream_weights_.push_back(weight);
        return (int) stream_weights_.size() - 1;
    }
    std::vector<int64_t> stream_weights() const {
        const std::lock_guard<std::mutex> lk(stream_mu_);
        return stream_weights_;
    }
    // Queues one layer's prompt rows behind a.x_ready and returns at once. Any thread may call it.
    bool prefill_layer(const strata_xtier_prefill_args& a, std::string& err);
    // The prompt layers finished since the last call by any thread, oldest first, one record each. Any thread may call it.
    bool prefill_stats(std::vector<strata_xtier_prefill_record>& out, std::string& err);
    // A wait for the card timed out, so it may never finish its queued work: close and SharedHost then leave the card and its memory alone.
    void mark_hung() { hung_ = true; }
    bool hung() const { return hung_; }

private:
    std::string text() const;

    const Api* api_ = nullptr;
    void* h_ = nullptr;
    std::atomic<bool> hung_{false};
    int64_t prompt_tokens_ = 0;
    int64_t stream_slots_ = 0;
    uint64_t stream_blob_ = 0;
    mutable std::mutex stream_mu_;
    std::vector<int64_t> stream_weights_;   // per place, the weight of the stage that joined there
};

// Pinned host memory both makes map, always its own, so no expert blob is registered (their fills rely on HIP copying unpinned memory at once).
class SharedHost {
public:
    SharedHost() = default;
    SharedHost(const SharedHost&) = delete;
    SharedHost& operator=(const SharedHost&) = delete;
    ~SharedHost();

    // Tries hipHostMalloc then cudaHostRegister (C), cudaHostAlloc then register_host (A), mmap then both registers (B). Coherent asks C for coherent memory.
    bool alloc(Device& d, size_t bytes, std::string& err, bool coherent = false);
    // mmap, touched, then registered with the AMD card (and with cuda, CUDA) a piece at a time (Q), so no card's other work waits long, or false with the reason. A copy from it stays inside a piece.
    bool alloc_quiet(Device& d, size_t bytes, size_t piece, bool cuda, std::string& err);
    void* host() const { return host_; }
    void* cuda_dev() const { return cdev_; }
    void* hip_dev() const { return hdev_; }
    // The variant that worked: 'C', 'A', 'B' or 'Q', or 0 before alloc.
    char variant() const { return kind_; }
    // Gives it back, unless the card hung: then it stays mapped and allocated.
    void release();

private:

    Device* d_ = nullptr;
    char kind_ = 0;
    void* host_ = nullptr;
    void* cdev_ = nullptr;
    void* hdev_ = nullptr;
    size_t bytes_ = 0;
    bool cuda_reg_ = false, hip_reg_ = false;
    size_t piece_ = 0, cuda_pieces_ = 0, hip_pieces_ = 0;   // Q: the piece size and the pieces each make registered
};

}  // namespace strata::core::xtier
