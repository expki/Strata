// The NVIDIA card's extra expert cache reserve beside the pinned RAM copy, from what the box measured after the cache was sized.
#pragma once

#include <cstdint>

namespace strata::program {

/// The lowest request start the reserve keeps free, with room above the engine's 256 MiB LOW line.
constexpr uint64_t kRequestFloorBytes = 350ull << 20;
/// The verify windows' captures after load: 138 to 142 MiB with the copy, 130 without (RTX 3090, Swift 1.5 Q6_K).
constexpr uint64_t kVerifyCaptureBytes = 140ull << 20;
/// What the default 700 MiB reserve left free at load without the copy on the same card (431 MiB).
constexpr uint64_t kLoadSpareBytes = 431ull << 20;

/// What the copy's CUDA registration takes on the card: 2 MiB a GiB (198 MiB measured for 97.86 GiB).
constexpr uint64_t copy_register_bytes(uint64_t copy) { return copy / 512; }

/// The reserve to add beside a copy of `copy` bytes: what lifts the lowest request start to the floor after its registration and the captures.
constexpr uint64_t copy_vram_reserve(uint64_t copy, uint64_t captures) {
    const uint64_t need = kRequestFloorBytes + captures + copy_register_bytes(copy);
    return need > kLoadSpareBytes ? need - kLoadSpareBytes : 0;
}

}  // namespace strata::program
