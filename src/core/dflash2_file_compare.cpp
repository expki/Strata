// src/core/dflash2_file_compare.cpp - the DF2P pack against the incoai GGUF of the same drafter (host only, no GPU).
//
//     dflash2_file_compare <pack> <gguf>
//
// Checks the same tensor set (name, dtype, shape), every cfg key, the bytes of each q8_0 and f32 tensor, and each bf16 codebook value within
// max|row| / 254 + one bf16 ulp (the GGUF holds the codebooks as Q8_0).  Prints "OK <n> tensors" or the first mismatch.
#include "strata/core/dflash2_file.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace strata::core;

static int fail(const std::string& what) {
    std::printf("MISMATCH %s\n", what.c_str());
    return 1;
}

static float bf16_to_f32(uint16_t h) {
    const uint32_t b = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

static bool is_number(const std::string& s, double& v) {
    char* end = nullptr;
    v = std::strtod(s.c_str(), &end);
    return !s.empty() && end != nullptr && *end == 0;
}

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: dflash2_file_compare <pack> <gguf>\n"); return 2; }
    std::string err;
    auto a = Dflash2Source::open(argv[1], err);
    if (!a) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    auto b = Dflash2Source::open(argv[2], err);
    if (!b) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }

    std::map<std::string, std::string> ca = a->cfg(), cb = b->cfg();
    ca.erase("source");
    cb.erase("source");
    for (const auto& [k, va] : ca) {
        const auto it = cb.find(k);
        if (it == cb.end()) return fail("cfg " + k + " is only in the pack (" + va + ")");
        double x, y;
        const bool same = (is_number(va, x) && is_number(it->second, y)) ? std::fabs(x - y) <= 1e-6 * std::fabs(x) : va == it->second;
        if (!same) return fail("cfg " + k + ": pack " + va + ", gguf " + it->second);
    }
    for (const auto& [k, vb] : cb)
        if (ca.find(k) == ca.end()) return fail("cfg " + k + " is only in the gguf (" + vb + ")");

    std::map<std::string, const Dflash2Tensor*> ta;
    for (const Dflash2Tensor& t : a->tensors()) ta[t.name] = &t;
    if (ta.size() != b->tensors().size()) return fail("the pack has " + std::to_string(ta.size()) + " tensors, the gguf " + std::to_string(b->tensors().size()));
    int n = 0;
    for (const Dflash2Tensor& g : b->tensors()) {
        const auto it = ta.find(g.name);
        if (it == ta.end()) return fail("tensor " + g.name + " is only in the gguf");
        const Dflash2Tensor& p = *it->second;
        if (p.dtype != g.dtype || p.rows != g.rows || p.cols != g.cols || p.bytes != g.bytes)
            return fail("tensor " + g.name + ": pack " + p.dtype + " " + std::to_string(p.rows) + "x" + std::to_string(p.cols) + ", gguf " + g.dtype + " " +
                        std::to_string(g.rows) + "x" + std::to_string(g.cols));
        std::vector<uint8_t> da(p.bytes), db(g.bytes);
        if (!a->read(p, da.data(), err) || !b->read(g, db.data(), err)) return fail(err);
        if (g.dtype != "bf16") {
            if (std::memcmp(da.data(), db.data(), da.size()) != 0) {
                size_t i = 0;
                while (da[i] == db[i]) ++i;
                std::string what = "tensor " + g.name + " differs at byte " + std::to_string(i);
                if (g.dtype == "q8_0") {
                    int64_t bad_d = 0, bad_q = 0, bad_blocks = 0, maxdq = 0;
                    for (size_t k = 0; k < da.size() / 34; ++k) {
                        const uint8_t* x = da.data() + k * 34;
                        const uint8_t* y = db.data() + k * 34;
                        if (std::memcmp(x, y, 34) == 0) continue;
                        ++bad_blocks;
                        if (x[0] != y[0] || x[1] != y[1]) ++bad_d;
                        for (int j = 2; j < 34; ++j)
                            if (x[j] != y[j]) { ++bad_q; maxdq = std::max<int64_t>(maxdq, std::llabs((int64_t) (int8_t) x[j] - (int64_t) (int8_t) y[j])); }
                    }
                    what += " (" + std::to_string(bad_blocks) + " of " + std::to_string(da.size() / 34) + " blocks differ, " + std::to_string(bad_d) + " scales, " +
                            std::to_string(bad_q) + " quants, largest step " + std::to_string(maxdq) + ")";
                }
                return fail(what);
            }
        } else {
            const uint16_t* x = reinterpret_cast<const uint16_t*>(da.data());
            const uint16_t* y = reinterpret_cast<const uint16_t*>(db.data());
            double worst = 0;
            for (int64_t r = 0; r < g.rows; ++r) {
                float amax = 0;
                for (int64_t c = 0; c < g.cols; ++c) amax = std::fmax(amax, std::fabs(bf16_to_f32(x[r * g.cols + c])));
                for (int64_t c = 0; c < g.cols; ++c) {
                    const float va = bf16_to_f32(x[r * g.cols + c]), vb = bf16_to_f32(y[r * g.cols + c]);
                    const float ulp = std::ldexp(1.0f, std::ilogb(std::fabs(vb) > 0 ? vb : 1e-30f) - 7);
                    const float tol = amax / 254.0f + ulp;
                    worst = std::fmax(worst, std::fabs(va - vb) / tol);
                    if (!(std::fabs(va - vb) <= tol)) return fail("tensor " + g.name + " row " + std::to_string(r) + " col " + std::to_string(c) + ": pack " +
                                                                 std::to_string(va) + ", gguf " + std::to_string(vb) + ", tolerance " + std::to_string(tol));
                }
            }
            std::printf("  %s: worst error %.3f of the tolerance\n", g.name.c_str(), worst);
        }
        ++n;
    }
    std::printf("OK %d tensors\n", n);
    return 0;
}
