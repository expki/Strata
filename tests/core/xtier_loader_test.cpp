// The AMD tier loader on the CPU: the --peer-device value and the ways the library can be refused.
#include "strata/core/xtier.hpp"
#include "xtier_skip.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef STUB_PATH
#error "STUB_PATH is the library tests/core/xtier_stub.cpp builds"
#endif
#if !defined(FULL_STUB_PATH) || !defined(NOSTATS_STUB_PATH)
#error "FULL_STUB_PATH and NOSTATS_STUB_PATH are the libraries tests/core/xtier_stub_full.cpp builds"
#endif

using namespace strata::core::xtier;

static int g_fail = 0;
#define CHECK(c)                                                                                  \
    do {                                                                                          \
        const bool ok_ = (c);                                                                     \
        std::printf("  %-96s %s\n", #c, ok_ ? "ok" : "FAIL");                                     \
        if (!ok_) ++g_fail;                                                                       \
    } while (0)

static void set_env(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static bool ends_with(const std::string& s, const std::string& tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

int main() {
    PeerSpec s;
    std::string e;

    std::printf("parse_peer_device\n");
    CHECK(parse_peer_device("1", s, e) && s.cuda == 1 && s.hip == -1);
    CHECK(parse_peer_device("3", s, e) && s.cuda == 3 && s.hip == -1);
    CHECK(parse_peer_device("hip:0", s, e) && s.hip == 0 && s.cuda == -1);
    CHECK(parse_peer_device("hip:2", s, e) && s.hip == 2 && s.cuda == -1);
    CHECK(!parse_peer_device("hip:-1", s, e));
    CHECK(!parse_peer_device("hip:x", s, e));
    CHECK(!parse_peer_device("hip:", s, e));
    CHECK(!parse_peer_device("hip:1x", s, e));
    CHECK(parse_peer_device("0", s, e) && s.cuda == -1 && s.hip == -1);   // "0" is no peer, as atoi made it before
    CHECK(!parse_peer_device("-1", s, e));
    CHECK(!parse_peer_device("1x", s, e));
    CHECK(!parse_peer_device("", s, e));
    CHECK(!parse_peer_device("rocm:0", s, e));
    CHECK(!e.empty());

    std::printf("library_path\n");
    set_env("STRATA_XTIER_LIB", "/somewhere/else/lib.so");
    CHECK(library_path() == "/somewhere/else/lib.so");
    set_env("STRATA_XTIER_LIB", "");
    CHECK(ends_with(library_path(), STRATA_XTIER_LIB_NAME));

    std::printf("load_from\n");
    const std::string missing = "/nonexistent/libstrata_xtier_hip.so";
    CHECK(load_from(missing, e) == nullptr && e.find("not found") != std::string::npos &&
          e.find(missing) != std::string::npos);
    const std::string missing_text = e;
    CHECK(e == "AMD tier library " + missing +
               " not found (build it with -DSTRATA_XTIER=ON in the HIP tree, or set STRATA_XTIER_LIB)");

    const std::string stub = STUB_PATH;
    CHECK(load_from(stub, e) == nullptr && e.find("another engine version") != std::string::npos);
    CHECK(e == "AMD tier library " + stub + " is from another engine version");

    std::printf("version 9: its prompt records and decode_out\n");
    CHECK(STRATA_XTIER_ABI_VERSION == 9);
    const std::string full = FULL_STUB_PATH, nostats = NOSTATS_STUB_PATH;
    const Api* fa = load_from(full, e);
    CHECK(fa != nullptr && e.empty() && fa->prefill_stats != nullptr && fa->decode_out != nullptr && fa->decode_counts != nullptr);
    strata_xtier_prefill_record rec[4] = {};
    int64_t n_rec = -1;
    CHECK(fa != nullptr && fa->prefill_stats(nullptr, rec, 4, &n_rec) == 0 && n_rec == 2);
    CHECK(rec[0].caller == 2 && rec[0].seq == 7 && rec[0].rows == 1234 && rec[0].experts == 30 && rec[0].bytes_in == 5678 &&
          rec[0].bytes_streamed == 9000 && rec[0].kernel_ms == 1.5 && rec[0].copy_ms == 0.25 && rec[0].start_ms == 3.0);
    CHECK(rec[1].caller == 0 && rec[1].seq == 8 && rec[1].rows == 10 && rec[1].experts == 1 && rec[1].bytes_in == 20 &&
          rec[1].bytes_streamed == 0 && rec[1].kernel_ms == 0.5 && rec[1].copy_ms == 0.125 && rec[1].start_ms == 0.0);
    CHECK(load_from(nostats, e) == nullptr && e == "AMD tier library " + nostats + " is from another engine version");

    const std::filesystem::path junk = std::filesystem::temp_directory_path() / "xtier_loader_test_not_a_library.so";
    { std::ofstream(junk) << "this is not a shared library\n"; }
#if defined(_WIN32)
    CHECK(load_from(junk.string(), e) == nullptr && e.find(" did not load (Windows error ") != std::string::npos &&
          e.find("not found") == std::string::npos);   // LoadLibrary's text has no LD_LIBRARY_PATH hint
#else
    CHECK(load_from(junk.string(), e) == nullptr && ends_with(e, " (is /opt/rocm/lib on LD_LIBRARY_PATH?)") &&
          e.find("not found") == std::string::npos);
#endif
    std::filesystem::remove(junk);

    std::printf("load and Device\n");
    set_env("STRATA_XTIER_LIB", missing.c_str());
    CHECK(load(e) == nullptr && e.find(missing) != std::string::npos);
    set_env("STRATA_XTIER_LIB", stub.c_str());
    std::string e2;
    CHECK(load(e2) == nullptr && e2 == e);   // the first answer is kept for the process
    Device d;
    uint64_t f = 0, t = 0;
    CHECK(!d.valid() && !d.open(0, e) && !e.empty() && !d.valid());
    CHECK(!d.mem_info(f, t, e) && !e.empty());
    uint32_t w = 0;
    uint64_t r = 0;
    float ms = 0.f;
    e.clear();
    CHECK(!d.flag_echo(&w, &w, &w, 1, 1000, e) && !e.empty());
    e.clear();
    CHECK(!d.sync(e) && !e.empty());
    e.clear();
    CHECK(d.host_alloc(4096, e) == nullptr && !e.empty());
    e.clear();
    CHECK(!d.probe(0, &w, 4, 1, r, ms, e) && !e.empty());
    d.host_free(&w);   // nothing to free on a closed card
    std::vector<uint64_t> addr;
    e.clear();
    CHECK(!d.arena({4096}, addr, e) && !e.empty());
    e.clear();
    CHECK(!d.fill(0, &w, 4, e) && !e.empty());
    e.clear();
    CHECK(!d.read(0, &w, 4, e) && !e.empty());
    e.clear();
    CHECK(!d.refill_record(e) && !e.empty());
    e.clear();
    CHECK(d.refill_done(true, &e) == -1 && !e.empty() && d.refill_done(false) == -1);
    strata_xtier_fmt fmt = {};
    strata_xtier_meta meta = {};
    const float* rows = nullptr;
    e.clear();
    CHECK(!d.decode(fmt, nullptr, 1, meta, e) && !e.empty());
    e.clear();
    CHECK(!d.decode_wait(&rows, e) && !e.empty());
    double ms3[3] = {};
    CHECK(d.timing(ms3) == -1);
    e.clear();
    CHECK(!d.prefill_open(64, 2560, 4096, 4096, 8, 4096, e) && !e.empty());
    const strata_xtier_prefill_args pa = {};
    e.clear();
    CHECK(!d.prefill_layer(pa, e) && !e.empty());
    std::vector<strata_xtier_prefill_record> recs(3);
    e.clear();
    CHECK(!d.prefill_stats(recs, e) && !e.empty() && recs.empty());
    SharedHost sh;
    e.clear();
    CHECK(!sh.alloc(d, 4096, e) && !e.empty() && sh.host() == nullptr && sh.variant() == 0);
    d.close();

    std::printf("the skip rule\n");
    CHECK(xtier_is_skip(missing_text));
    CHECK(xtier_is_skip("no AMD GPU"));
    CHECK(xtier_is_skip("no AMD GPU (hipGetDeviceCount: no ROCm-capable device is detected)"));
    CHECK(!xtier_is_skip("/p/libstrata_xtier_hip.so: version `GLIBCXX_3.4.32' not found (required by /p/libstrata_xtier_hip.so) (is /opt/rocm/lib on LD_LIBRARY_PATH?)"));
    CHECK(!xtier_is_skip("libamdhip64.so.7: cannot open shared object file: No such file or directory (is /opt/rocm/lib on LD_LIBRARY_PATH?)"));
    CHECK(!xtier_is_skip("AMD tier library /p/lib.so is from another engine version"));
    CHECK(!xtier_is_skip("AMD card 3 does not exist, the system has 1"));
    CHECK(!xtier_is_skip(""));

    std::printf("%s\n", g_fail == 0 ? "xtier_loader_test: ok" : "xtier_loader_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
