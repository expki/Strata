// include/strata/core/dflash2_file.hpp - the DFlash2 drafter's file on the host: the engine's DF2P pack or the incoai GGUF (Q8_0).
//
// Both read as the same thing: the pack's cfg keys, and tensors under the pack's names, dtypes and offsets (a "pack-layout blob").  The
// GGUF's two codebooks are widened from Q8_0 to bf16 on read, as the pack stores them.  Host only: no CUDA.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

struct Dflash2Tensor {
    std::string name, dtype;   ///< the pack's name, and "q8_0" (ggml blocks), "f32" or "bf16"
    int64_t rows = 0, cols = 0;
    uint64_t off = 0, bytes = 0;
};

class Dflash2Source {
public:
    /// Opens a DF2P pack or a GGUF, told apart by the first four bytes.  Null with `err` set when it cannot be read.
    static std::unique_ptr<Dflash2Source> open(const std::string& path, std::string& err);
    virtual ~Dflash2Source() = default;
    /// The pack's cfg keys (hidden, layers, heads, kv_heads, head_dim, inter, window, rope_theta, eps, conv_group, rank, mask_token, vocab, taps ...).
    virtual const std::map<std::string, std::string>& cfg() const = 0;
    /// Pack names, pack dtypes and offsets (256 byte aligned) in a pack-layout blob.
    virtual const std::vector<Dflash2Tensor>& tensors() const = 0;
    /// The tensor's `t.bytes` bytes in pack layout.
    virtual bool read(const Dflash2Tensor& t, uint8_t* dst, std::string& err) const = 0;
};

/// The pack name of a GGUF tensor of the incoai drafter, or "" when the name is not one.
std::string dflash2_pack_name(const std::string& gguf_name);

/// `n_values` (a multiple of 32) values of ggml Q8_0 blocks as bf16, rounded to nearest even.
void dflash2_q8_0_to_bf16(const uint8_t* blocks, int64_t n_values, uint16_t* out);

}  // namespace strata::core
