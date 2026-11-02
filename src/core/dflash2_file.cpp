// src/core/dflash2_file.cpp - see include/strata/core/dflash2_file.hpp.
#include "strata/core/dflash2_file.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace strata::core {
namespace {

// ---- the DF2P pack (tools/dflash2_pack.py)

class PackSource final : public Dflash2Source {
public:
    ~PackSource() override { if (f_) std::fclose(f_); }

    bool init(const std::string& path, std::string& err) {
        f_ = std::fopen(path.c_str(), "rb");
        if (f_ == nullptr) { err = "dflash2: cannot open " + path; return false; }
        uint32_t hdr[3];
        if (std::fread(hdr, 4, 3, f_) != 3 || std::memcmp(hdr, "DF2P", 4) != 0 || hdr[1] != 1) { err = "dflash2: " + path + " is not a DF2P v1 pack"; return false; }
        std::string index(hdr[2], ' ');
        if (std::fread(index.data(), 1, hdr[2], f_) != hdr[2]) { err = "dflash2: the pack's index is cut short"; return false; }
        data_off_ = 12 + (long) hdr[2];
        for (size_t i = 0; i < index.size();) {
            const size_t j = index.find('\n', i);
            if (j == std::string::npos) break;
            const std::string line = index.substr(i, j - i);
            i = j + 1;
            char a[256], b[64], c[256];
            long long v1, v2, v3, v4;
            if (std::sscanf(line.c_str(), "cfg %63s %255s", b, c) == 2) {
                cfg_[b] = c;
            } else if (std::sscanf(line.c_str(), "t %255s %63s %lld %lld %lld %lld", a, b, &v1, &v2, &v3, &v4) == 6) {
                Dflash2Tensor t;
                t.name = a; t.dtype = b; t.rows = v1; t.cols = v2; t.off = (uint64_t) v3; t.bytes = (uint64_t) v4;
                tensors_.push_back(t);
            }
        }
        return true;
    }
    const std::map<std::string, std::string>& cfg() const override { return cfg_; }
    const std::vector<Dflash2Tensor>& tensors() const override { return tensors_; }
    bool read(const Dflash2Tensor& t, uint8_t* dst, std::string& err) const override {
#if defined(_WIN32)
        _fseeki64(f_, data_off_ + (long long) t.off, SEEK_SET);
#else
        fseeko(f_, data_off_ + (off_t) t.off, SEEK_SET);
#endif
        if (std::fread(dst, 1, t.bytes, f_) != t.bytes) { err = "dflash2: the pack is shorter than its index"; return false; }
        return true;
    }

private:
    std::FILE* f_ = nullptr;
    long data_off_ = 0;
    std::map<std::string, std::string> cfg_;
    std::vector<Dflash2Tensor> tensors_;
};

// ---- the incoai GGUF

constexpr uint32_t kF32 = 0, kQ8_0 = 8;

std::string num_str(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.9g", v);
    return buf;
}

class GgufSource final : public Dflash2Source {
public:
    bool init(const std::string& path, std::string& err) {
        try {
            gguf_ = std::make_unique<GgufFile>(path);
            build();
        } catch (const std::exception& e) {
            err = std::string("dflash2: ") + path + ": " + e.what();
            return false;
        }
        return true;
    }
    const std::map<std::string, std::string>& cfg() const override { return cfg_; }
    const std::vector<Dflash2Tensor>& tensors() const override { return tensors_; }
    bool read(const Dflash2Tensor& t, uint8_t* dst, std::string& err) const override {
        const auto it = src_.find(t.name);
        if (it == src_.end()) { err = "dflash2: no tensor " + t.name; return false; }
        const uint8_t* p = gguf_->tensor_data(*it->second);
        if (t.dtype == "bf16") dflash2_q8_0_to_bf16(p, t.rows * t.cols, reinterpret_cast<uint16_t*>(dst));
        else std::memcpy(dst, p, t.bytes);
        return true;
    }

private:
    const MetaValue& meta(const std::string& key) const {
        const MetaValue* v = gguf_->get(key);
        if (v == nullptr) throw std::runtime_error("the metadata key " + key + " is missing");
        return *v;
    }
    std::string meta_int(const std::string& key) const { return std::to_string((long long) meta(key).num()); }

    void build() {
        const char* const ints[][2] = {
            {"hidden", "dflash.embedding_length"}, {"layers", "dflash.block_count"}, {"heads", "dflash.attention.head_count"},
            {"kv_heads", "dflash.attention.head_count_kv"}, {"head_dim", "dflash.attention.key_length"}, {"inter", "dflash.feed_forward_length"},
            {"window", "dflash.attention.sliding_window"}, {"block", "dflash.block_size"}, {"conv_group", "dflash.conv_group_size"},
            {"conv_kernel", "dflash.conv_kernel_size"}, {"rank", "dflash.selector_rank"}, {"top_k", "dflash.selector_top_k"},
            {"mask_token", "tokenizer.ggml.mask_token_id"}};
        for (const auto& kv : ints) cfg_[kv[0]] = meta_int(kv[1]);
        cfg_["rope_theta"] = num_str((float) meta("dflash.rope.freq_base").num());
        cfg_["eps"] = num_str((float) meta("dflash.attention.layer_norm_rms_epsilon").num());
        const MetaValue& taps = meta("dflash.target_layers");
        if (taps.type != MetaType::ARRAY || taps.items.empty()) throw std::runtime_error("dflash.target_layers is not an array");
        std::string list;
        for (const MetaValue& v : taps.items) {
            if (v.num() < 1) throw std::runtime_error("dflash.target_layers holds " + num_str(v.num()) + ", and the embedding (0) has no tap");
            list += (list.empty() ? "" : ",") + std::to_string((long long) v.num() - 1);   // the GGUF counts the embedding as 0
        }
        cfg_["taps"] = list;

        uint64_t off = 0;
        for (const TensorInfo& g : gguf_->tensors()) {
            const std::string name = dflash2_pack_name(g.name);
            if (name.empty()) throw std::runtime_error("the tensor " + g.name + " is not one of the drafter's");
            if (g.shape.empty()) throw std::runtime_error("the tensor " + g.name + " has no dimensions");
            Dflash2Tensor t;
            t.name = name;
            t.cols = (int64_t) g.shape[0];
            t.rows = 1;
            for (size_t i = 1; i < g.shape.size(); ++i) t.rows *= (int64_t) g.shape[i];
            if (g.shape.size() == 1) t.cols = 1, t.rows = (int64_t) g.shape[0];   // the pack folds a vector into rows
            const bool codebook = name.find("_codebook") != std::string::npos;
            if (g.type != kF32 && g.type != kQ8_0) throw std::runtime_error("the tensor " + g.name + " is " + g.type_name() + ", not F32 or Q8_0");
            if (g.type == kF32) {
                if (codebook) throw std::runtime_error("the codebook " + g.name + " is not Q8_0");
                t.dtype = "f32";
                t.bytes = g.elements() * 4;
            } else {
                if (g.shape[0] % 32 != 0) throw std::runtime_error("the tensor " + g.name + " has a row that is not whole Q8_0 blocks");
                t.dtype = codebook ? "bf16" : "q8_0";
                t.bytes = codebook ? g.elements() * 2 : g.elements() / 32 * 34;
            }
            const uint64_t file_bytes = tensor_payload_bytes(g);
            const uint64_t payload = gguf_->file_size() - gguf_->data_start();
            if (file_bytes == 0 || g.offset > payload || file_bytes > payload - g.offset) throw std::runtime_error("the tensor " + g.name + " lies outside the file");
            off = (off + 255) & ~(uint64_t) 255;
            t.off = off;
            off += t.bytes;
            if (codebook) cfg_["vocab"] = std::to_string(t.rows);
            src_[name] = &g;
            tensors_.push_back(t);
        }
        if (cfg_.find("vocab") == cfg_.end()) throw std::runtime_error("the selector codebooks are missing");
    }

    std::unique_ptr<GgufFile> gguf_;
    std::map<std::string, std::string> cfg_;
    std::vector<Dflash2Tensor> tensors_;
    std::map<std::string, const TensorInfo*> src_;
};

float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else {   // a subnormal half is normal in f32
            exp = 127 - 14;
            while (!(man & 0x400)) { man <<= 1; --exp; }
            bits = sign | (exp << 23) | ((man & 0x3FF) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint16_t float_to_bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    if ((b & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t) ((b >> 16) | 0x40);   // NaN stays a NaN
    b += 0x7FFF + ((b >> 16) & 1);
    return (uint16_t) (b >> 16);
}

}  // namespace

std::unique_ptr<Dflash2Source> Dflash2Source::open(const std::string& path, std::string& err) {
    uint8_t magic[4] = {};
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { err = "dflash2: cannot open " + path; return nullptr; }
    const bool got = std::fread(magic, 1, 4, f) == 4;
    std::fclose(f);
    if (got && std::memcmp(magic, "DF2P", 4) == 0) {
        auto s = std::make_unique<PackSource>();
        return s->init(path, err) ? std::move(s) : nullptr;
    }
    if (got && std::memcmp(magic, "GGUF", 4) == 0) {
        auto s = std::make_unique<GgufSource>();
        return s->init(path, err) ? std::move(s) : nullptr;
    }
    err = "dflash2: " + path + " is neither a DF2P pack nor a GGUF";
    return nullptr;
}

std::string dflash2_pack_name(const std::string& n) {
    static const std::map<std::string, std::string> top = {
        {"fc.weight", "fc.weight"},
        {"enc.output_norm.weight", "hidden_norm.weight"},
        {"output_norm.weight", "norm.weight"},
        {"selector_hidden.weight", "candidate_selector.hidden_projection.weight"},
        {"selector_predecessor.weight", "candidate_selector.predecessor_codebook"},
        {"selector_successor.weight", "candidate_selector.successor_codebook"}};
    if (const auto it = top.find(n); it != top.end()) return it->second;
    // blk.N.<part>: the parts that are matrices or norms end in .weight, the two conv base kernels do not
    static const std::map<std::string, std::string> part = {
        {"attn_q.weight", "self_attn.q_proj.weight"}, {"attn_k.weight", "self_attn.k_proj.weight"},
        {"attn_v.weight", "self_attn.v_proj.weight"}, {"attn_output.weight", "self_attn.o_proj.weight"},
        {"ffn_gate.weight", "mlp.gate_proj.weight"}, {"ffn_up.weight", "mlp.up_proj.weight"}, {"ffn_down.weight", "mlp.down_proj.weight"},
        {"attn_conv_proj.weight", "attention_conv.kernel_projection.weight"}, {"ffn_conv_proj.weight", "mlp_conv.kernel_projection.weight"},
        {"attn_norm.weight", "input_layernorm.weight"}, {"ffn_norm.weight", "post_attention_layernorm.weight"},
        {"attn_q_norm.weight", "self_attn.q_norm.weight"}, {"attn_k_norm.weight", "self_attn.k_norm.weight"},
        {"attn_conv_base", "attention_conv.base_kernel"}, {"ffn_conv_base", "mlp_conv.base_kernel"}};
    if (n.compare(0, 4, "blk.") != 0) return "";
    const size_t dot = n.find('.', 4);
    if (dot == std::string::npos || dot == 4 || dot - 4 > 4) return "";
    for (size_t i = 4; i < dot; ++i)
        if (n[i] < '0' || n[i] > '9') return "";
    const auto it = part.find(n.substr(dot + 1));
    return it == part.end() ? "" : "layers." + n.substr(4, dot - 4) + "." + it->second;
}

bool dflash2_check_taps(const std::vector<int>& taps, int n_layers, std::string& err) {
    if (taps.empty()) { err = "dflash2: the drafter names no target layers to tap"; return false; }
    for (int t : taps)
        if (t < 0 || (n_layers > 0 && t >= n_layers - 1)) {
            err = "dflash2: the drafter taps layer " + std::to_string(t) +
                  (n_layers > 0 ? ", which this model of " + std::to_string(n_layers) + " layers cannot give" : std::string(", which is not a layer"));
            return false;
        }
    return true;
}

void dflash2_q8_0_to_bf16(const uint8_t* blocks, int64_t n_values, uint16_t* out) {
    for (int64_t b = 0; b < n_values / 32; ++b) {
        const uint8_t* p = blocks + b * 34;
        const float d = half_to_float((uint16_t) (p[0] | (p[1] << 8)));
        for (int i = 0; i < 32; ++i) out[b * 32 + i] = float_to_bf16(d * (float) (int8_t) p[2 + i]);
    }
}

}  // namespace strata::core
