// src/core/dflash2_file_test.cpp - the DFlash2 drafter file: the GGUF name map, the Q8_0 to bf16 step and a synthetic GGUF (CPU only).
#include "strata/core/dflash2_file.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace strata::core;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

template <class T> static void put(std::vector<uint8_t>& b, T v) {
    const size_t n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
static void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
static void kv_u32(std::vector<uint8_t>& b, const std::string& k, uint32_t v) { put_str(b, k); put<uint32_t>(b, 4); put<uint32_t>(b, v); }
static void kv_f32(std::vector<uint8_t>& b, const std::string& k, float v) { put_str(b, k); put<uint32_t>(b, 6); put<float>(b, v); }

struct FakeTensor { std::string name; std::vector<uint64_t> shape; uint32_t type; std::vector<uint8_t> data; };

// A GGUF v3 with the dflash.* keys and the given tensors, data 32-byte aligned.
static std::filesystem::path write_gguf(const std::vector<FakeTensor>& ts) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);
    put<uint32_t>(b, 3);
    put<uint64_t>(b, ts.size());
    put<uint64_t>(b, 17);
    kv_u32(b, "dflash.block_count", 1);
    kv_u32(b, "dflash.embedding_length", 64);
    kv_u32(b, "dflash.feed_forward_length", 96);
    kv_u32(b, "dflash.attention.head_count", 32);
    kv_u32(b, "dflash.attention.head_count_kv", 8);
    kv_u32(b, "dflash.attention.key_length", 128);
    kv_f32(b, "dflash.rope.freq_base", 10000.0f);
    kv_f32(b, "dflash.attention.layer_norm_rms_epsilon", 1e-6f);
    kv_u32(b, "dflash.block_size", 8);
    kv_u32(b, "dflash.conv_kernel_size", 2);
    kv_u32(b, "dflash.conv_group_size", 16);
    kv_u32(b, "dflash.selector_rank", 256);
    kv_u32(b, "dflash.selector_top_k", 16);
    kv_u32(b, "dflash.attention.sliding_window", 2048);
    kv_u32(b, "tokenizer.ggml.mask_token_id", 248077);
    put_str(b, "dflash.target_layers");   // an array of 2 u32
    put<uint32_t>(b, 9); put<uint32_t>(b, 4); put<uint64_t>(b, 2); put<uint32_t>(b, 6); put<uint32_t>(b, 16);
    put_str(b, "general.alignment"); put<uint32_t>(b, 4); put<uint32_t>(b, 32);
    uint64_t off = 0;
    for (const FakeTensor& t : ts) {
        put_str(b, t.name);
        put<uint32_t>(b, (uint32_t) t.shape.size());
        for (uint64_t d : t.shape) put<uint64_t>(b, d);
        put<uint32_t>(b, t.type);
        put<uint64_t>(b, off);
        off += (t.data.size() + 31) / 32 * 32;
    }
    b.resize((b.size() + 31) / 32 * 32);
    for (const FakeTensor& t : ts) {
        b.insert(b.end(), t.data.begin(), t.data.end());
        b.resize((b.size() + 31) / 32 * 32);
    }
    const auto path = std::filesystem::temp_directory_path() / "strata_dflash2_file_test.gguf";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize) b.size());
    return path;
}

int main() {
    CHECK(dflash2_pack_name("blk.3.attn_q.weight") == "layers.3.self_attn.q_proj.weight");
    CHECK(dflash2_pack_name("blk.0.attn_output.weight") == "layers.0.self_attn.o_proj.weight");
    CHECK(dflash2_pack_name("blk.4.ffn_conv_base") == "layers.4.mlp_conv.base_kernel");
    CHECK(dflash2_pack_name("blk.1.attn_conv_proj.weight") == "layers.1.attention_conv.kernel_projection.weight");
    CHECK(dflash2_pack_name("blk.12.ffn_down.weight") == "layers.12.mlp.down_proj.weight");
    CHECK(dflash2_pack_name("blk.0.attn_q_norm.weight") == "layers.0.self_attn.q_norm.weight");
    CHECK(dflash2_pack_name("blk.0.ffn_norm.weight") == "layers.0.post_attention_layernorm.weight");
    CHECK(dflash2_pack_name("fc.weight") == "fc.weight");
    CHECK(dflash2_pack_name("enc.output_norm.weight") == "hidden_norm.weight");
    CHECK(dflash2_pack_name("output_norm.weight") == "norm.weight");
    CHECK(dflash2_pack_name("selector_hidden.weight") == "candidate_selector.hidden_projection.weight");
    CHECK(dflash2_pack_name("selector_predecessor.weight") == "candidate_selector.predecessor_codebook");
    CHECK(dflash2_pack_name("selector_successor.weight") == "candidate_selector.successor_codebook");
    CHECK(dflash2_pack_name("token_embd.weight") == "");
    CHECK(dflash2_pack_name("blk.0.attn_q") == "");           // a matrix needs its .weight
    CHECK(dflash2_pack_name("blk.0.ffn_conv_base.weight") == "");   // a base kernel has none
    CHECK(dflash2_pack_name("blk.x.attn_q.weight") == "");
    CHECK(dflash2_pack_name("blk.0.nope.weight") == "");

    // one Q8_0 block, d = 0.5 (fp16 0x3800), q = -2, 0, 3, rest 0: values -1.0, 0.0, 1.5
    uint8_t blk[34] = {0x00, 0x38, (uint8_t) -2, 0, 3};
    uint16_t o[32];
    dflash2_q8_0_to_bf16(blk, 32, o);
    CHECK(o[0] == 0xBF80 && o[1] == 0x0000 && o[2] == 0x3FC0);

    // d = 1/3 in fp16 (0x3555) times q = 127 is 42.323 in f32, which rounds to the bf16 42.25 (0x4229)
    uint8_t blk2[34] = {0x55, 0x35, 127};
    dflash2_q8_0_to_bf16(blk2, 32, o);
    CHECK(o[0] == 0x4229);

    // a synthetic GGUF: fc (Q8_0 [64 in, 3 out]), a norm (F32 [64]), the successor codebook (Q8_0 [256, 2])
    FakeTensor fc{"fc.weight", {64, 3}, 8, std::vector<uint8_t>(6 * 34, 0)};
    for (size_t i = 0; i < fc.data.size(); ++i) fc.data[i] = (uint8_t) (i * 7 + 1);
    FakeTensor norm{"enc.output_norm.weight", {64}, 0, std::vector<uint8_t>(64 * 4, 0)};
    for (int i = 0; i < 64; ++i) { const float v = 0.5f * (float) i; std::memcpy(norm.data.data() + 4 * i, &v, 4); }
    FakeTensor conv{"blk.0.attn_conv_base", {64, 2, 2}, 0, std::vector<uint8_t>(64 * 4 * 4, 0)};
    FakeTensor cb{"selector_successor.weight", {256, 2}, 8, std::vector<uint8_t>(2 * 8 * 34, 0)};
    for (int r = 0; r < 16; ++r) { cb.data[r * 34] = 0x00; cb.data[r * 34 + 1] = 0x38; cb.data[r * 34 + 2] = (uint8_t) (r + 1); }
    const auto path = write_gguf({fc, norm, conv, cb});
    {
        std::string err;
        auto src = Dflash2Source::open(path.string(), err);
        CHECK(src != nullptr);
        if (src) {
            const auto& c = src->cfg();
            CHECK(c.at("hidden") == "64" && c.at("layers") == "1" && c.at("inter") == "96");
            CHECK(c.at("heads") == "32" && c.at("kv_heads") == "8" && c.at("head_dim") == "128");
            CHECK(c.at("window") == "2048" && c.at("block") == "8" && c.at("conv_group") == "16" && c.at("conv_kernel") == "2");
            CHECK(c.at("rank") == "256" && c.at("top_k") == "16" && c.at("mask_token") == "248077" && c.at("vocab") == "2");
            CHECK(c.at("taps") == "6,16");
            CHECK(std::atof(c.at("rope_theta").c_str()) == 10000.0 && (float) std::atof(c.at("eps").c_str()) == 1e-6f);
            const auto& ts = src->tensors();
            CHECK(ts.size() == 4);
            if (ts.size() == 4) {
                CHECK(ts[0].name == "fc.weight" && ts[0].dtype == "q8_0" && ts[0].rows == 3 && ts[0].cols == 64 && ts[0].bytes == 6 * 34 && ts[0].off == 0);
                CHECK(ts[1].name == "hidden_norm.weight" && ts[1].dtype == "f32" && ts[1].rows == 64 && ts[1].cols == 1 && ts[1].off == 256);
                CHECK(ts[2].name == "layers.0.attention_conv.base_kernel" && ts[2].dtype == "f32" && ts[2].rows == 4 && ts[2].cols == 64 && ts[2].off == 512);
                CHECK(ts[3].name == "candidate_selector.successor_codebook" && ts[3].dtype == "bf16" && ts[3].rows == 2 && ts[3].cols == 256 && ts[3].bytes == 2 * 256 * 2);
                for (const auto& t : ts) CHECK(t.off % 256 == 0);
                std::vector<uint8_t> buf(ts[0].bytes);
                CHECK(src->read(ts[0], buf.data(), err) && buf == fc.data);
                buf.assign(ts[1].bytes, 0);
                CHECK(src->read(ts[1], buf.data(), err) && buf == norm.data);
                buf.assign(ts[3].bytes, 0);
                CHECK(src->read(ts[3], buf.data(), err));
                // block r holds 0.5 * (r + 1) first, then zeros, and a row is 8 blocks
                uint16_t v[3] = {};
                std::memcpy(&v[0], buf.data(), 2);
                std::memcpy(&v[1], buf.data() + 2, 2);
                std::memcpy(&v[2], buf.data() + 2 * 256, 2);
                CHECK(v[0] == 0x3F00 && v[1] == 0 && v[2] == 0x4090);
            }
        }
    }
    {
        std::string err;
        CHECK(Dflash2Source::open((std::filesystem::temp_directory_path() / "strata_dflash2_missing.bin").string(), err) == nullptr && !err.empty());
        const auto bad = write_gguf({FakeTensor{"token_embd.weight", {64}, 0, std::vector<uint8_t>(256, 0)}});
        err.clear();
        CHECK(Dflash2Source::open(bad.string(), err) == nullptr && err.find("token_embd.weight") != std::string::npos);
    }
    std::filesystem::remove(path);
    if (fails == 0) std::printf("dflash2_file_test: ok\n");
    return fails == 0 ? 0 : 1;
}
