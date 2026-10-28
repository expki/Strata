// The C interface between the CUDA engine and the HIP library that holds an AMD card as an expert tier.
#pragma once

#include <stddef.h>
#include <stdint.h>

// Bump this when a function changes meaning.
#define STRATA_XTIER_ABI_VERSION 9

#if defined(_WIN32)
#define STRATA_XTIER_LIB_NAME "strata_xtier_hip.dll"
#else
#define STRATA_XTIER_LIB_NAME "libstrata_xtier_hip.so"
#endif

// Word i of the probe's buffer for a seed, computed the same way by both makes.
#define STRATA_XTIER_PATTERN(seed, i) ((uint32_t) (seed) ^ ((uint32_t) (i) * 2654435761u))

// Decode entries per layer, MAXT 8 times top 10 (src/core/xtier.cpp checks it).
#define STRATA_XTIER_CAP 80

// The widest embedding and expert width a decode takes, so the library sizes its buffers once at open.
#define STRATA_XTIER_MAX_EMBD 4096
#define STRATA_XTIER_MAX_FF 2048

// What strata_xtier_decode_wait returns when the card did not finish in time. That decode stays pending until a later wait or close.
#define STRATA_XTIER_TIMEOUT (-2)

// The finished prompt layers the library keeps for strata_xtier_prefill_stats. Past this many the oldest are dropped.
#define STRATA_XTIER_PF_RECORDS 1024

// One layer's expert format, the numbers of the engine's native layout.
typedef struct strata_xtier_fmt {
    int32_t gu_type, d_type;
    int64_t n_embd, n_ff;
    uint64_t up_off, down_off, blob_bytes;
} strata_xtier_fmt;

// One decode call's plan: groups of entries, a group is one resident expert and an entry is one token routed to it.
typedef struct strata_xtier_meta {
    unsigned long long ptr[STRATA_XTIER_CAP];
    int32_t start[STRATA_XTIER_CAP + 1];
    int32_t dst[STRATA_XTIER_CAP];
    int32_t tok[STRATA_XTIER_CAP];
    int32_t count[4];   // [0] groups, [1] entries
    int32_t out_row[STRATA_XTIER_CAP];
} strata_xtier_meta;

// One layer's prompt rows: the device half of the compact SUMS host route of src/prefill/prefill.cpp, its experts in order_peer order.
typedef struct strata_xtier_prefill_args {
    uint32_t seq;                        /* this call's flag value */
    int32_t caller;                      /* who queued it, given back in its record */
    const uint32_t* x_ready;             /* HIP addresses of shared host words: the engine raises x_ready when x and w landed */
    uint32_t* done;                      /* the library raises done when the sums landed */
    uint32_t* wait_err;                  /* 1 = the wait for x_ready timed out */
    uint64_t timeout_ns;
    int64_t T, N, K; int32_t f16;   /* x and the sums in FP16 (0: FP32) */
    const void* x_hip; const float* w_hip;   /* shared host buffers, HIP addresses */
    void* sum_host;                          /* shared host buffer, host address (the D2H target) */
    const int32_t* pair; const int32_t* src; /* plain host arrays, rows_peer each, copied at the call */
    const int32_t* bounds; int64_t n_bounds;
    const int32_t* adds; int64_t n_adds;
    int64_t rows_peer;
    int32_t mmq_gu_type, mmq_d_type; uint64_t gu_bytes, d_bytes;
    strata_xtier_fmt fmt;
    int64_t n_experts; const uint64_t* expert_addr; const int32_t* expert_rows;   /* order_peer order */
    const uint64_t* expert_host;   /* per expert in order_peer order: 0 = in the tier's arena (expert_addr), else the HIP address of its staged or pinned host copy */
    const int64_t* expert_slot;    /* per expert: the stream ring slot its card copy goes to (read where expert_host is not 0) */
    const uint64_t* expert_tag;    /* per expert: which copy that is, never 0; a slot that holds the tag already is read without a copy */
    const uint32_t* expert_ready;  /* HIP address of the flag word the engine raises when the staged copies of this layer landed, or null */
    uint32_t ready_seq;
    int64_t n_groups; const int64_t* group_first; const int64_t* group_last;
    const int64_t* group_adds_at; const int32_t* group_ntok;
} strata_xtier_prefill_args;

// One prompt layer the card finished: its call's caller and seq, rows, experts, bytes in (x, w, tables) and streamed (copied onto the card, not those its ring held), ms in kernels and copies, and ms from queued to x seen (-1: not measured).
typedef struct strata_xtier_prefill_record {
    int32_t caller;
    uint32_t seq;
    int64_t rows, experts;
    uint64_t bytes_in, bytes_streamed;
    double kernel_ms, copy_ms, start_ms;
} strata_xtier_prefill_record;

// The sizeof of every struct in this header added together, modulo 100000.
static inline int strata_xtier_structs_size(void) {
    return (int) ((sizeof(strata_xtier_fmt) + sizeof(strata_xtier_meta) + sizeof(strata_xtier_prefill_args) +
                   sizeof(strata_xtier_prefill_record)) % 100000);
}

#ifdef __cplusplus
extern "C" {
#endif

// Every int returns 0 for ok or an error with its text in strata_xtier_last_error, except refill_done, decode_wait and timing, and every entry point that touches the card selects it first.

// STRATA_XTIER_ABI_VERSION * 100000 + strata_xtier_structs_size() of the header the library was built from.
int strata_xtier_abi(void);
// Starts the tier on one HIP card, or returns null with the reason in err.
void* strata_xtier_open(int hip_device, char* err, size_t err_len);
void strata_xtier_close(void* t);
// Free and total VRAM of the card in bytes.
int strata_xtier_mem_info(void* t, uint64_t* free_b, uint64_t* total_b);
// Maps engine host memory into the card's address space and gives the card's pointer in hip_dev.
int strata_xtier_register(void* t, void* host, uint64_t bytes, void** hip_dev);
int strata_xtier_unregister(void* t, void* host);
// The text of the last error on this tier, valid until this thread asks again.
const char* strata_xtier_last_error(void* t);
// Queues "wait until *in_hip >= seq or the timeout (then *err_hip = 1), then *out_hip = seq" on the decode stream and returns at once.
int strata_xtier_flag_echo(void* t, const uint32_t* in_hip, uint32_t* out_hip, uint32_t* err_hip, uint32_t seq, uint64_t timeout_ns);
// Waits for everything queued on the decode and prompt streams.
int strata_xtier_sync(void* t);
// Pinned host memory owned by the HIP runtime, mapped and portable, and coherent when asked (a flag the card spins on), or null on failure.
void* strata_xtier_host_alloc(void* t, uint64_t bytes, int coherent);
void strata_xtier_host_free(void* t, void* host);
// A probe, and it waits: op 0 writes the seed's pattern into host, op 1 counts the wrong words, op 2 times a copy into host, op 3 a copy out of host.
int strata_xtier_probe(void* t, int op, void* host, uint64_t bytes, uint32_t seed, uint64_t* result, float* ms);
// One device allocation cut into n slots of slot_bytes[i], each 256 byte aligned, and each slot's address in slot_addr. Once per tier.
int strata_xtier_arena(void* t, const uint64_t* slot_bytes, int64_t n, uint64_t* slot_addr);
// Queues a copy of host into a slot on the refill stream and returns at once. Until refill_done says 1, host stays valid and no decode reads the slot.
int strata_xtier_fill(void* t, int64_t slot, const void* host, uint64_t bytes);
// Copies a slot into host after the fills queued before it and waits, for checking a fill.
int strata_xtier_read(void* t, int64_t slot, void* host, uint64_t bytes);
// Marks the fills queued so far.
int strata_xtier_refill_record(void* t);
// 1 when the fills before the last record are done, 0 when not yet, -1 on error. With wait it blocks until they are.
int strata_xtier_refill_done(void* t, int wait);
// Queues one layer's expert rows on the decode stream and returns at once. x (n_tok rows of n_embd floats) and m are copied before it returns.
int strata_xtier_decode(void* t, const strata_xtier_fmt* f, const float* x, int64_t n_tok, const strata_xtier_meta* m);
// As decode, but row r lands at row m->out_row[r] of out_host (host memory registered with strata_xtier_register), launched as one graph per window shape.
int strata_xtier_decode_out(void* t, const strata_xtier_fmt* f, const float* x, int64_t n_tok, const strata_xtier_meta* m, void* out_host);
// decode_out's launches since open in counts[0..4]: step by step, captured, replayed, graphs dropped, and the graphs held now.
int strata_xtier_decode_counts(void* t, uint64_t* counts);
// Spins until the queued decode is done or timeout_ns has passed (0 is no bound) and gives its pinned rows, m->count[1] rows of n_embd floats valid until the next decode.
int strata_xtier_decode_wait(void* t, const float** rows, uint64_t timeout_ns);
// STRATA_XTIER_TIMING=1 only: the card's ms in copies in, kernels and copy out (ms[0..2]) over the decodes waited for since the last call, and their count, or -1.
int strata_xtier_timing(void* t, double* ms);
// Sizes the prompt buffers for chunks of T_max tokens and matrices of gu_max and d_max bytes, and a ring of stream_slots streamed blobs of blob_max bytes. Grows only.
int strata_xtier_prefill_open(void* t, int64_t T_max, int64_t N, uint64_t gu_max, uint64_t d_max, int64_t stream_slots, uint64_t blob_max);
// Queues one layer's prompt rows behind x_ready, its streamed experts' copies behind expert_ready, and returns, unless over 4 calls are in flight: then it waits for the oldest one's x_ready.
int strata_xtier_prefill_layer(void* t, const strata_xtier_prefill_args* a);
// The prompt layers finished since the last call, oldest first: at most cap records into out and their count into n, the rest wait for the next call. Always on.
int strata_xtier_prefill_stats(void* t, strata_xtier_prefill_record* out, int64_t cap, int64_t* n);

#ifdef __cplusplus
}
#endif
