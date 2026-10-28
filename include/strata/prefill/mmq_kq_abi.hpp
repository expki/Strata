// include/strata/prefill/mmq_kq_abi.hpp - the interface between the engine (src/prefill/moe_mmq.cu) and strata_kq, the
// shared library with the K-quant MMQ prompt kernels (src/prefill/mmq_kq_lib.cu).  Bump the version when a function's
// meaning changes; the argument block (ggml's mmq_args) is checked by its size.
#pragma once

#include <cstdlib>

#define STRATA_KQ_ABI_VERSION 1

#if defined(_WIN32)
#define STRATA_KQ_LIB_NAME "strata_kq.dll"
#else
#define STRATA_KQ_LIB_NAME "libstrata_kq.so"
#endif

extern "C" {
typedef int (*strata_kq_abi_fn)();
typedef int (*strata_kq_init_fn)(int device);
typedef int (*strata_kq_run_fn)(int device, int ggml_type, const void* mmq_args, void* stream);
}

// the engine's side (src/prefill/mmq_kq_loader.cpp)
namespace strata::prefill::mmq::kq {
/// ggml's id of Q6_K, the one K-quant type MMQ takes by itself (moe_mmq.cu checks it against ggml.h).
constexpr int kQ6K = 14;
/// Whether this K-quant type's prompt products go through MMQ: STRATA_MMQ_KQ=0 never, =1 all four types, unset Q6_K only.
inline bool wanted(int ggml_type) {
    static const char mode = [] {
        const char* v = std::getenv("STRATA_MMQ_KQ");
        return v != nullptr && (v[0] == '0' || v[0] == '1') ? v[0] : '-';
    }();
    return mode == '1' || (mode == '-' && ggml_type == kQ6K);
}
/// wanted(ggml_type) and the library is open and started on every GPU (the first call opens it and says what happened).
bool available(int ggml_type);
/// One product on the current GPU: 0 = launched, else an error (-1: no library or not this type).
int run(int ggml_type, const void* mmq_args, void* stream);
/// sizeof(mmq_args) on the engine's side (moe_mmq.cu), checked against the library's.
int abi_args_size();
}  // namespace strata::prefill::mmq::kq
