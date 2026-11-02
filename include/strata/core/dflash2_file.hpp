// include/strata/core/dflash2_file.hpp - the DFlash2 drafter's file on the host, a DF2P pack or the incoai GGUF, read as one pack layout (no CUDA).
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

/// True when `taps` is not empty and every tap is a layer in [0, n_layers - 1), the layers the verifier taps.  `n_layers` 0 checks the sign only.
bool dflash2_check_taps(const std::vector<int>& taps, int n_layers, std::string& err);

/// `n_values` (a multiple of 32) values of ggml Q8_0 blocks as bf16, rounded to nearest even.
void dflash2_q8_0_to_bf16(const uint8_t* blocks, int64_t n_values, uint16_t* out);

}  // namespace strata::core
