// The ring sized by the chunk: 96 slots for the bench's 2K prompt, about 200 for 8K, about 500 for 20K, never past the cap. CPU only.
#include "strata/prefill/chunk_ring.hpp"

#include <cstdio>
#include <cstdlib>

using strata::prefill::chunk_ring_slots;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

}  // namespace

int main() {
    check(chunk_ring_slots(2048, 512) == 96, "a 2K chunk keeps the 96-slot floor");
    check(chunk_ring_slots(1024, 512) == 96, "the smallest streaming chunk keeps the floor");
    check(chunk_ring_slots(8192, 512) == 204, "an 8K chunk takes about 200 slots");
    check(chunk_ring_slots(20224, 512) == 505, "the 20K prompt's chunk takes about 500 slots");
    check(chunk_ring_slots(24576, 512) == 512, "a 24K chunk stops at the cap");
    check(chunk_ring_slots(24576, 384) == 384, "a smaller cap holds");
    check(chunk_ring_slots(8192, 64) == 96, "a cap under the floor gives the floor");
    std::printf("chunk_ring_test: %d checks PASS\n", checks);
    return 0;
}
