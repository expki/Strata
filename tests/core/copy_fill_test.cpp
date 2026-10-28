// The page-locked RAM copy filled layer by layer, on a CPU-only PC with the device calls wrapped at the link.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <linux/magic.h>
#include <sys/mman.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {
std::map<uintptr_t, size_t> g_registered;   // start -> bytes of each live registration
std::map<uintptr_t, size_t> g_host_alloc;   // start -> bytes of each cudaHostAlloc
int g_register_calls = 0;
int g_fail_call = -1;                       // the registration call that fails (-1: none)
}  // namespace

// ---- the wrapped device calls: memory is ordinary memory and every operation completes at once.
extern "C" cudaError_t __wrap_cudaMalloc(void** p, size_t n) {
    *p = std::malloc(n);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
extern "C" cudaError_t __wrap_cudaFree(void* p) { std::free(p); return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) {
    std::memcpy(d, s, n);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t) {
    std::memcpy(d, s, n);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaDeviceSynchronize(void) { return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaGetLastError(void) { return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaSetDevice(int) { return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaMemGetInfo(size_t* free_b, size_t* total_b) {
    *free_b = 64ull << 30;
    *total_b = 128ull << 30;
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaHostAlloc(void** p, size_t n, unsigned) {
    *p = std::malloc(n);
    if (*p != nullptr) g_host_alloc[(uintptr_t) *p] = n;
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
extern "C" cudaError_t __wrap_cudaFreeHost(void* p) {
    g_host_alloc.erase((uintptr_t) p);
    std::free(p);
    return cudaSuccess;
}
extern "C" const char* __wrap_cudaGetErrorString(cudaError_t) { return "wrapped error"; }
// a page in two registrations is refused, as the driver refuses it
extern "C" cudaError_t __wrap_cudaHostRegister(void* p, size_t n, unsigned) {
    if (g_register_calls++ == g_fail_call) return cudaErrorMemoryAllocation;
    const uintptr_t a = (uintptr_t) p & ~(uintptr_t) 4095, e = ((uintptr_t) p + n + 4095) & ~(uintptr_t) 4095;
    for (const auto& [s, len] : g_registered) {
        const uintptr_t sa = s & ~(uintptr_t) 4095, se = (s + len + 4095) & ~(uintptr_t) 4095;
        if (a < se && sa < e) return cudaErrorHostMemoryAlreadyRegistered;
    }
    g_registered[(uintptr_t) p] = n;
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaHostUnregister(void* p) {
    return g_registered.erase((uintptr_t) p) == 1 ? cudaSuccess : cudaErrorHostMemoryNotRegistered;
}
extern "C" cudaError_t __wrap_cudaHostGetDevicePointer(void** d, void* p, unsigned) {
    for (const auto& [s, len] : g_host_alloc)
        if ((uintptr_t) p >= s && (uintptr_t) p < s + len) {
            *d = p;
            return cudaSuccess;
        }
    for (const auto& [s, len] : g_registered)
        if ((uintptr_t) p >= s && (uintptr_t) p < s + len) {
            *d = p;
            return cudaSuccess;
        }
    return cudaErrorInvalidValue;
}

namespace fs = std::filesystem;

namespace {

constexpr int64_t kLayers = 4;
constexpr int64_t kExperts = 4;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;
    explicit TempDirectory(const fs::path& base = fs::temp_directory_path()) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = base / ("strata-copy-fill-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }
    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

uint8_t truth_byte(int64_t l, int64_t e, size_t i) {
    return (uint8_t) ((i * 131u + (size_t) (l * kExperts + e) * 37u) ^ (i >> 8));
}

// the GPU cache holds experts 0 and 1 of layer 0 and all of layer 2, so the copy has an empty layer
bool on_gpu(int64_t l, int64_t e) { return (l == 0 && e < 2) || l == 2; }

struct Rig {
    std::vector<std::vector<uint8_t>> truth;
    TempDirectory dir;
    strata::core::FileExpertSource src;
    strata::core::ExpertCache cache;
    size_t blob = 0;

    // cached = false gives the file's pages back before the copy, so every expert is read with O_DIRECT
    explicit Rig(bool cached) {
        std::string err;
        require(strata::kernels::cpu::expert_layout_load(dir.path.string(), kLayers, kExperts, err), err);
        blob = (size_t) strata::kernels::cpu::BLOB;
        truth.resize((size_t) (kLayers * kExperts));
        {
            std::ofstream out(dir.path / "experts.bin", std::ios::binary);
            for (int64_t l = 0; l < kLayers; ++l)
                for (int64_t e = 0; e < kExperts; ++e) {
                    std::vector<uint8_t>& t = truth[(size_t) (l * kExperts + e)];
                    t.resize(blob);
                    for (size_t i = 0; i < blob; ++i) t[i] = truth_byte(l, e, i);
                    out.write((const char*) t.data(), (std::streamsize) blob);
                }
            require((bool) out, "fixture write failed");
        }
        const int fd = ::open((dir.path / "experts.bin").c_str(), O_RDONLY);
        require(fd >= 0 && ::fsync(fd) == 0, "fixture fsync failed");   // clean pages, so a drop can give them back
        if (!cached) (void) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
        require(src.open(dir.path.string(), kLayers, kExperts, err), err);
        require(cache.open(6, kLayers, kExperts, blob, err), err);
        for (int64_t l = 0; l < kLayers; ++l)
            for (int64_t e = 0; e < kExperts; ++e)
                if (on_gpu(l, e)) {
                    const int32_t s = cache.admit(l, e);
                    require(s >= 0 && cache.fill_slot_blocking(s, truth[(size_t) (l * kExperts + e)].data(), err), err);
                }
    }

    bool holds_truth(int64_t l, int64_t e) {
        const uint8_t* b = src.blob(l, e);
        return b != nullptr && std::memcmp(b, truth[(size_t) (l * kExperts + e)].data(), blob) == 0;
    }

    // pages wholly inside each expert's range of the file: how many of them the page cache holds, and how many there are
    std::pair<size_t, size_t> cached_pages(int64_t l, int64_t e) {
        const fs::path file = dir.path / "experts.bin";
        const int fd = ::open(file.c_str(), O_RDONLY);
        const size_t size = (size_t) fs::file_size(file);
        void* m = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        require(m != MAP_FAILED, "fixture map failed");
        const size_t a = ((size_t) (l * kExperts + e) * blob + 4095) / 4096, z = (size_t) (l * kExperts + e + 1) * blob / 4096;
        std::vector<unsigned char> vec(size / 4096 + 1);
        require(mincore(m, size, vec.data()) == 0, "mincore failed");
        size_t held = 0;
        for (size_t p = a; p < z; ++p) held += vec[p] & 1;
        munmap(m, size);
        return {held, z - a};
    }
};

void test_page_align_layers() {
    using strata::core::detail::kNoCacheComplement;
    const std::vector<uint64_t> blob{5000, 5000, 7000};
    std::vector<uint64_t> offsets{0, kNoCacheComplement, 5000,   // layer 0: two experts
                                  kNoCacheComplement, kNoCacheComplement, kNoCacheComplement,   // layer 1: none
                                  10000, 17000, kNoCacheComplement};   // layer 2: two experts
    uint64_t bytes = 24000;
    const std::vector<uint64_t> starts = strata::core::detail::page_align_layers(3, 3, blob, offsets, bytes);
    require(starts == std::vector<uint64_t>({0, 12288, 12288, 28672}), "layer starts are not on pages");
    require(offsets == std::vector<uint64_t>({0, kNoCacheComplement, 5000, kNoCacheComplement, kNoCacheComplement,
                                              kNoCacheComplement, 12288, 19288, kNoCacheComplement}),
            "the blobs are not laid out in layer order from each start");
    require(bytes == 28672, "the copy's size is not its last start");
    std::cout << "page_align_layers: PASS\n";
}

void check_copy(Rig& rig, int64_t pinned_layers) {
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kExperts; ++e) {
            if (on_gpu(l, e)) {
                require(!rig.src.has_resident(l, e), "an expert the GPU holds is in the copy");
                continue;
            }
            require(rig.src.has_resident(l, e) && rig.holds_truth(l, e), "a copied blob differs from the file's");
            const bool pin = l < pinned_layers;
            require(rig.src.pinned(l, e) == pin, "an expert's pinned state is wrong");
            require((rig.src.device_alias(l, e) != nullptr) == pin, "an expert's device alias is wrong");
            if (pin) require(rig.src.device_alias(l, e) == rig.src.blob(l, e), "the device alias is not the host address");
        }
}

void test_cached_fill() {
    g_registered.clear();
    g_register_calls = 0;
    g_fail_call = -1;
    Rig rig(true);
    std::string err;
    require(rig.src.pin_cache_complement(rig.cache, err, true, {}, -1, 0, 0), err);
    const auto& cf = rig.src.copy_fill();
    require(cf.by_layer && cf.experts == 10, "the copy did not go layer by layer");
    require(cf.drive_bytes == 0, "a cached expert was read from the drive");
    require(g_registered.size() == 3 && g_register_calls == 3, "one registration per layer with experts");
    for (const auto& [s, len] : g_registered) require(s % 4096 == 0 && len % 4096 == 0, "a registration is not whole pages");
    check_copy(rig, kLayers);
    require(rig.src.pinned_bytes() == rig.src.resident_bytes(), "the whole copy is not pinned");
    // the copied experts' file pages are given back, the GPU's stay cached (tmpfs keeps every page)
    struct statfs fsinfo{};
    const bool drops = statfs(rig.dir.path.c_str(), &fsinfo) == 0 && fsinfo.f_type != TMPFS_MAGIC;
    for (int64_t l = 0; l < kLayers && drops; ++l)
        for (int64_t e = 0; e < kExperts; ++e) {
            const auto [held, pages] = rig.cached_pages(l, e);
            if (on_gpu(l, e)) require(held == pages, "an expert the copy does not hold lost its cached pages");
            else require(held == 0, "a copied expert's file pages are still cached");
        }
    rig.src.close();
    require(g_registered.empty(), "close left a layer registered");
    std::cout << "cached fill: 10 experts bitwise, 3 registrations, copied pages dropped: PASS\n";
}

void test_uncached_fill() {
    g_registered.clear();
    g_register_calls = 0;
    g_fail_call = -1;
    Rig rig(false);
    std::string err;
    require(rig.src.pin_cache_complement(rig.cache, err, true, {}, -1, 0, 0), err);
    const int probe = ::open((rig.dir.path / "experts.bin").c_str(), O_RDONLY | O_DIRECT);
    if (probe >= 0) {
        ::close(probe);
        require(rig.src.copy_fill().drive_bytes == 10 * (uint64_t) rig.blob, "the uncached experts were not read unbuffered");
    }
    check_copy(rig, kLayers);
    std::cout << "uncached fill: " << rig.src.copy_fill().drive_bytes << " bytes unbuffered, bitwise: PASS\n";
}

void test_failed_registration() {
    g_registered.clear();
    g_register_calls = 0;
    g_fail_call = 1;   // layer 1's registration: layer 0 stays pinned
    Rig rig(true);
    std::string err;
    require(rig.src.pin_cache_complement(rig.cache, err, true, {}, -1, 0, 0), err);
    require(g_registered.size() == 1, "registration went on past the refused layer");
    check_copy(rig, 1);
    require(rig.src.complement_pinned() && rig.src.pinned_bytes() > 0 && rig.src.pinned_bytes() < rig.src.resident_bytes(),
            "the registered prefix is not reported");
    rig.src.close();
    require(g_registered.empty(), "close left a layer registered");
    std::cout << "failed registration at layer 1: layer 0 pinned, the rest resident and pageable: PASS\n";
}

void test_host_alloc_switch() {
    g_registered.clear();
    g_register_calls = 0;
    g_fail_call = -1;
    setenv("STRATA_PIN_ALLOC", "host", 1);
    Rig rig(true);
    std::string err;
    require(rig.src.pin_cache_complement(rig.cache, err, true, {}, -1, 0, 0), err);
    unsetenv("STRATA_PIN_ALLOC");
    require(!rig.src.copy_fill().by_layer && g_register_calls == 0, "STRATA_PIN_ALLOC=host did not keep one allocation");
    check_copy(rig, kLayers);
    std::cout << "STRATA_PIN_ALLOC=host: one allocation: PASS\n";
}

}  // namespace

int main() {
    try {
        test_page_align_layers();
        test_cached_fill();
        test_uncached_fill();
        test_failed_registration();
        test_host_alloc_switch();
        std::cout << "copy_fill_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "copy_fill_test: " << error.what() << '\n';
        return 1;
    }
}
