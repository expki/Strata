// The CUDA engine's side of the AMD tier: the --peer-device value and the loader for strata_xtier_hip.
#include "strata/core/xtier.hpp"

#include "strata/kernels/cpu/expert.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <mutex>
#include <system_error>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

static_assert(STRATA_XTIER_CAP == strata::kernels::cpu::MAXT * 10, "the library's entries follow MAXT x top-10");
static_assert(strata::kernels::cpu::H <= STRATA_XTIER_MAX_EMBD && strata::kernels::cpu::FF <= STRATA_XTIER_MAX_FF,
              "the library's buffers are sized for the engine's widths");

namespace strata::core::xtier {
namespace {

std::string own_folder() {
#if defined(_WIN32)
    char p[MAX_PATH] = {};
    const DWORD n = GetModuleFileNameA(nullptr, p, MAX_PATH);
    std::string s(p, n);
    const size_t k = s.find_last_of("\\/");
    return k == std::string::npos ? std::string() : s.substr(0, k + 1);
#else
    char p[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", p, sizeof(p) - 1);
    std::string s(p, n > 0 ? (size_t) n : 0);
    const size_t k = s.find_last_of('/');
    return k == std::string::npos ? std::string() : s.substr(0, k + 1);
#endif
}

void* sym(void* h, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress((HMODULE) h, name));
#else
    return dlsym(h, name);
#endif
}

// Digits only, so a sign, a space or a trailing letter is refused.
bool parse_card(const std::string& s, int& out) {
    if (s.empty() || s.size() > 4) return false;
    int v = 0;
    for (const char c : s) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + (c - '0');
    }
    out = v;
    return true;
}

}  // namespace

// A value that is not a positive number means the default.
uint64_t wait_ms() {
    constexpr uint64_t kDefault = 30000, kMax = 1000000000ull;
    const char* v = std::getenv("STRATA_XTIER_WAIT_MS");
    if (v == nullptr || v[0] == '\0') return kDefault;
    char* end = nullptr;
    const long long n = std::strtoll(v, &end, 10);
    return end != v && *end == '\0' && n > 0 ? std::min<uint64_t>((uint64_t) n, kMax) : kDefault;
}

bool parse_peer_device(const std::string& v, PeerSpec& out, std::string& err) {
    const std::string hip = "hip:";
    int n = 0;
    if (v.compare(0, hip.size(), hip) == 0) {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
        err = "this is the AMD engine: --peer-device takes a card number";
        return false;
#elif defined(__APPLE__)
        err = "this is the Mac engine: --peer-device takes a card number";
        return false;
#else
        if (!parse_card(v.substr(hip.size()), n)) {
            err = "--peer-device hip:N takes the number of an AMD card (0 or more), not '" + v + "'";
            return false;
        }
        out = PeerSpec{};
        out.hip = n;
        return true;
#endif
    }
    if (parse_card(v, n) && n == 0) {   // "0" was always accepted and meant no peer
        out = PeerSpec{};
        return true;
    }
    if (!parse_card(v, n) || n < 1) {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(__APPLE__)
        const char* shape = "a card number (1 or more)";
#else
        const char* shape = "a card number (1 or more) or hip:N";
#endif
        err = std::string("--peer-device takes ") + shape + ", not '" + v + "'";
        return false;
    }
    out = PeerSpec{};
    out.cuda = n;
    return true;
}

HipCard claim_hip_card(int n) {
    HipCard c;
    c.n = n;
    if (std::getenv("HIP_VISIBLE_DEVICES") != nullptr) {
        c.open = n;
        return c;
    }
#if defined(_WIN32)
    _putenv_s("HIP_VISIBLE_DEVICES", std::to_string(n).c_str());
#else
    setenv("HIP_VISIBLE_DEVICES", std::to_string(n).c_str(), 1);
#endif
    c.open = 0;
    c.engine_set = true;
    return c;
}

std::string library_path() {
    const char* v = std::getenv("STRATA_XTIER_LIB");
    if (v != nullptr && v[0] != '\0') return v;
    return own_folder() + STRATA_XTIER_LIB_NAME;
}

const Api* load_from(const std::string& path, std::string& err) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        err = "AMD tier library " + path + " not found (build it with -DSTRATA_XTIER=ON in the HIP tree, or set STRATA_XTIER_LIB)";
        return nullptr;
    }
#if defined(_WIN32)
    void* h = (void*) LoadLibraryA(path.c_str());
    if (h == nullptr) {
        err = "AMD tier library " + path + " did not load (Windows error " + std::to_string(GetLastError()) + ")";
        return nullptr;
    }
#else
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr) {
        const char* why = dlerror();
        err = std::string(why != nullptr ? why : "dlopen failed") + " (is /opt/rocm/lib on LD_LIBRARY_PATH?)";
        return nullptr;
    }
#endif
    Api a;
    a.abi = (decltype(a.abi)) sym(h, "strata_xtier_abi");
    const int abi = a.abi != nullptr ? a.abi() : -1;
    if (abi / 100000 != STRATA_XTIER_ABI_VERSION || abi % 100000 != strata_xtier_structs_size()) {
        err = "AMD tier library " + path + " is from another engine version";
        return nullptr;
    }
    a.open = (decltype(a.open)) sym(h, "strata_xtier_open");
    a.close = (decltype(a.close)) sym(h, "strata_xtier_close");
    a.mem_info = (decltype(a.mem_info)) sym(h, "strata_xtier_mem_info");
    a.reg = (decltype(a.reg)) sym(h, "strata_xtier_register");
    a.unreg = (decltype(a.unreg)) sym(h, "strata_xtier_unregister");
    a.last_error = (decltype(a.last_error)) sym(h, "strata_xtier_last_error");
    a.flag_echo = (decltype(a.flag_echo)) sym(h, "strata_xtier_flag_echo");
    a.sync = (decltype(a.sync)) sym(h, "strata_xtier_sync");
    a.host_alloc = (decltype(a.host_alloc)) sym(h, "strata_xtier_host_alloc");
    a.host_free = (decltype(a.host_free)) sym(h, "strata_xtier_host_free");
    a.probe = (decltype(a.probe)) sym(h, "strata_xtier_probe");
    a.arena = (decltype(a.arena)) sym(h, "strata_xtier_arena");
    a.fill = (decltype(a.fill)) sym(h, "strata_xtier_fill");
    a.read = (decltype(a.read)) sym(h, "strata_xtier_read");
    a.refill_record = (decltype(a.refill_record)) sym(h, "strata_xtier_refill_record");
    a.refill_done = (decltype(a.refill_done)) sym(h, "strata_xtier_refill_done");
    a.decode = (decltype(a.decode)) sym(h, "strata_xtier_decode");
    a.decode_wait = (decltype(a.decode_wait)) sym(h, "strata_xtier_decode_wait");
    a.timing = (decltype(a.timing)) sym(h, "strata_xtier_timing");   // optional: without it Device::timing gives -1
    a.prefill_open = (decltype(a.prefill_open)) sym(h, "strata_xtier_prefill_open");
    a.prefill_layer = (decltype(a.prefill_layer)) sym(h, "strata_xtier_prefill_layer");
    if (a.open == nullptr || a.close == nullptr || a.mem_info == nullptr || a.reg == nullptr || a.unreg == nullptr ||
        a.last_error == nullptr || a.flag_echo == nullptr || a.sync == nullptr || a.host_alloc == nullptr ||
        a.host_free == nullptr || a.probe == nullptr || a.arena == nullptr || a.fill == nullptr || a.read == nullptr ||
        a.refill_record == nullptr || a.refill_done == nullptr || a.decode == nullptr || a.decode_wait == nullptr ||
        a.prefill_open == nullptr || a.prefill_layer == nullptr) {
        err = "AMD tier library " + path + " is from another engine version";
        return nullptr;
    }
    static std::mutex mu;
    static std::deque<Api> kept;   // the library stays loaded, so its table stays too
    std::lock_guard<std::mutex> lk(mu);
    kept.push_back(a);
    err.clear();
    return &kept.back();
}

const Api* load(std::string& err) {
    static std::mutex mu;
    static bool done = false;
    static const Api* api = nullptr;
    static std::string why;
    std::lock_guard<std::mutex> lk(mu);
    if (!done) {
        api = load_from(library_path(), why);
        done = true;
    }
    err = why;
    return api;
}

Device::~Device() { close(); }

std::string Device::text() const {
    const char* m = api_->last_error(h_);
    return m != nullptr && m[0] != '\0' ? m : "AMD tier call failed";
}

bool Device::open(int hip_device, std::string& err) {
    close();
    hung_ = false;
    api_ = load(err);
    if (api_ == nullptr) return false;
    char buf[512] = {};
    h_ = api_->open(hip_device, buf, sizeof(buf));
    if (h_ == nullptr) {
        err = buf[0] != '\0' ? buf : "AMD card did not start";
        return false;
    }
    return true;
}

void Device::close() {
    if (h_ != nullptr && !hung_) api_->close(h_);   // a hung card's streams would never drain, and its memory is left as it is
    h_ = nullptr;
}

bool Device::mem_info(uint64_t& free_b, uint64_t& total_b, std::string& err) {
    if (h_ == nullptr) {
        err = "AMD card is not open";
        return false;
    }
    if (api_->mem_info(h_, &free_b, &total_b) != 0) {
        err = text();
        return false;
    }
    return true;
}

bool Device::register_host(void* host, uint64_t bytes, void** hip_dev, std::string& err) {
    if (h_ == nullptr) {
        err = "AMD card is not open";
        return false;
    }
    if (api_->reg(h_, host, bytes, hip_dev) != 0) {
        err = text();
        return false;
    }
    return true;
}

void Device::unregister_host(void* host) {
    if (h_ != nullptr) api_->unreg(h_, host);
}

bool Device::flag_echo(const uint32_t* in_hip, uint32_t* out_hip, uint32_t* err_hip, uint32_t seq, uint64_t timeout_ns,
                       std::string& err) {
    if (h_ == nullptr) {
        err = "AMD card is not open";
        return false;
    }
    if (api_->flag_echo(h_, in_hip, out_hip, err_hip, seq, timeout_ns) != 0) {
        err = text();
        return false;
    }
    return true;
}

bool Device::sync(std::string& err) {
    if (h_ == nullptr) {
        err = "AMD card is not open";
        return false;
    }
    if (api_->sync(h_) != 0) {
        err = text();
        return false;
    }
    return true;
}

void* Device::host_alloc(uint64_t bytes, std::string& err, bool coherent) {
    if (h_ == nullptr) {
        err = "AMD card is not open";
        return nullptr;
    }
    void* p = api_->host_alloc(h_, bytes, coherent ? 1 : 0);
    if (p == nullptr) err = text();
    return p;
}

void Device::host_free(void* host) {
    if (h_ != nullptr) api_->host_free(h_, host);
}

bool Device::probe(int op, void* host, uint64_t bytes, uint32_t seed, uint64_t& result, float& ms, std::string& err) {
    if (h_ == nullptr) {
        err = "AMD card is not open";
        return false;
    }
    if (api_->probe(h_, op, host, bytes, seed, &result, &ms) != 0) {
        err = text();
        return false;
    }
    return true;
}

bool Device::arena(const std::vector<uint64_t>& slot_bytes, std::vector<uint64_t>& slot_addr, std::string& err) {
    if (h_ == nullptr) {
        err = "AMD card is not open";
        return false;
    }
    std::vector<uint64_t> addr(slot_bytes.size());
    if (api_->arena(h_, slot_bytes.data(), (int64_t) slot_bytes.size(), addr.data()) != 0) {
        err = text();
        return false;
    }
    slot_addr = std::move(addr);
    return true;
}

bool Device::fill(int64_t slot, const void* host, uint64_t bytes, std::string& err) {
    if (h_ == nullptr) {
        err = "AMD card is not open";
        return false;
    }
    if (api_->fill(h_, slot, host, bytes) != 0) {
        err = text();
        return false;
    }
    return true;
}

bool Device::read(int64_t slot, void* host, uint64_t bytes, std::string& err) {
    if (h_ == nullptr) {
        err = "AMD card is not open";
        return false;
    }
    if (api_->read(h_, slot, host, bytes) != 0) {
        err = text();
        return false;
    }
    return true;
}

bool Device::refill_record(std::string& err) {
    if (h_ == nullptr) {
        err = "AMD card is not open";
        return false;
    }
    if (api_->refill_record(h_) != 0) {
        err = text();
        return false;
    }
    return true;
}

int Device::refill_done(bool wait, std::string* err) {
    if (h_ == nullptr) {
        if (err != nullptr) *err = "AMD card is not open";
        return -1;
    }
    const int r = api_->refill_done(h_, wait ? 1 : 0);
    if (r < 0 && err != nullptr) *err = text();
    return r < 0 ? -1 : r;
}

bool Device::decode(const strata_xtier_fmt& f, const float* x, int64_t n_tok, const strata_xtier_meta& m, std::string& err) {
    if (h_ == nullptr) {
        err = "AMD card is not open";
        return false;
    }
    if (api_->decode(h_, &f, x, n_tok, &m) != 0) {
        err = text();
        return false;
    }
    return true;
}

bool Device::decode_wait(const float** rows, std::string& err) {
    if (h_ == nullptr) {
        err = "AMD card is not open";
        return false;
    }
    const uint64_t ms = wait_ms();
    const int r = api_->decode_wait(h_, rows, ms * 1000000ull);
    if (r == STRATA_XTIER_TIMEOUT) {
        hung_ = true;
        err = "the AMD tier did not answer within STRATA_XTIER_WAIT_MS (" + std::to_string(ms) + " ms)";
        return false;
    }
    if (r != 0) {
        err = text();
        return false;
    }
    return true;
}

int Device::timing(double ms[3]) { return h_ != nullptr && api_->timing != nullptr ? api_->timing(h_, ms) : -1; }

bool Device::prefill_open(int64_t T_max, int64_t N, uint64_t gu_max, uint64_t d_max, std::string& err) {
    if (h_ == nullptr) {
        err = "AMD card is not open";
        return false;
    }
    if (api_->prefill_open(h_, T_max, N, gu_max, d_max) != 0) {
        err = text();
        return false;
    }
    return true;
}

bool Device::prefill_layer(const strata_xtier_prefill_args& a, std::string& err) {
    if (h_ == nullptr) {
        err = "AMD card is not open";
        return false;
    }
    if (api_->prefill_layer(h_, &a) != 0) {
        err = text();
        return false;
    }
    return true;
}

SharedHost::~SharedHost() { release(); }

void SharedHost::release() {
    if (d_ != nullptr && d_->hung()) {   // the card may still write it, so it stays mapped and allocated
        d_ = nullptr;
        kind_ = 0;
        host_ = cdev_ = hdev_ = nullptr;
        bytes_ = 0;
        cuda_reg_ = hip_reg_ = false;
        return;
    }
    if (hip_reg_) d_->unregister_host(host_);
    if (cuda_reg_) (void) cudaHostUnregister(host_);
    if (host_ != nullptr) {
        if (kind_ == 'C') d_->host_free(host_);
        else if (kind_ == 'A') (void) cudaFreeHost(host_);
#if !defined(_WIN32)
        else munmap(host_, bytes_);
#endif
    }
    d_ = nullptr;
    kind_ = 0;
    host_ = cdev_ = hdev_ = nullptr;
    bytes_ = 0;
    cuda_reg_ = hip_reg_ = false;
}

bool SharedHost::alloc(Device& d, size_t bytes, std::string& err, bool coherent) {
    release();
    if (!d.valid() || bytes == 0) {
        err = "shared host memory: no AMD card or no size";
        return false;
    }
    std::string why;
#if defined(_WIN32)
    const char kinds[] = "CA";
#else
    const char kinds[] = "CAB";
#endif
    for (const char kind : std::string(kinds)) {
        d_ = &d;
        kind_ = kind;
        bytes_ = bytes;
        std::string e;
        bool ok = true;
        if (kind == 'C') {   // the HIP runtime's pinned memory, which gave the NVIDIA cards the fastest copies into it
            host_ = d.host_alloc(bytes, e, coherent);
            ok = host_ != nullptr;
        } else if (kind == 'A') {
            ok = cudaHostAlloc(&host_, bytes, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess;
            if (!ok) host_ = nullptr;
#if !defined(_WIN32)
        } else {
            void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            ok = p != MAP_FAILED;
            host_ = ok ? p : nullptr;
            if (!ok) e = "mmap failed";
#endif
        }
        if (ok && kind != 'A')
            ok = cuda_reg_ = cudaHostRegister(host_, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess;
        if (ok) ok = cudaHostGetDevicePointer(&cdev_, host_, 0) == cudaSuccess;
        if (ok && kind == 'C') hdev_ = host_;   // the HIP runtime made it, so the card's pointer is the host pointer
        else if (ok) ok = hip_reg_ = d.register_host(host_, bytes, &hdev_, e);
        if (ok) return true;
        // a failed CUDA call stays the last error until asked, and a kernel's launch check would take it as its own
        why += std::string(why.empty() ? "" : ", ") + kind + ": " + (e.empty() ? cudaGetErrorString(cudaGetLastError()) : e);
        release();
    }
    err = "shared host memory of " + std::to_string(bytes) + " bytes did not register with both cards (" + why + ")";
    return false;
}

}  // namespace strata::core::xtier
