// tier_stream: the staging ring's entries, a layer's cap, the stager's copies landing layer by layer across chunks and on a stop, and each stage's slots of the card's ring. CPU only.
#include "strata/prefill/tier_stream.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <dirent.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

using namespace strata::prefill;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

constexpr size_t kSlot = 64;
constexpr int kLayers = 4;
using Clock = std::chrono::steady_clock;

uint8_t pattern(int chunk, int l, int e) { return (uint8_t) (1 + chunk * 37 + l * 11 + e); }

// One copy into the ring, as the test checks it: its layer and expert, its slot, and the byte it fills the slot with
struct Copy { int l, e; int64_t slot; uint8_t byte; };

// A stage and the tier around one ring: the stage plans and frees, the stager's threads fill slots, the tier reads each layer once its word is up
struct Rig {
    TierRing ring;
    TierLanding land;
    std::vector<uint8_t> host, src;
    uint32_t words[kLayers] = {};
    std::vector<int64_t> first;
    std::vector<Copy> copies;
    std::unique_ptr<TierStager> stager;   // last, so it stops before the rest goes

    Rig(int64_t slots, int threads, int batch) : ring(slots), host((size_t) slots * kSlot, 0), first(kLayers + 1, 0) {
        land.left = std::make_unique<std::atomic<int32_t>[]>(kLayers);
        land.words = words;
        stager = std::make_unique<TierStager>(land, threads, batch, nullptr);
    }

    // A chunk's copies, n[l] for layer l, each layer's first entry, then the stager's threads start on them (ungated: the entries a walk without the cap takes)
    void chunk(int c, const std::vector<int>& n, bool ungated = false) {
        land.gen += 1;
        copies.clear();
        int total = 0;
        for (int k : n) total += k;
        src.assign((size_t) total * kSlot, 0);
        std::vector<TierCopy> js;
        int64_t next = ring.entries();
        for (int l = 0; l < kLayers; ++l) {
            first[(size_t) l] = ungated ? next : ring.entries();
            ring.start_layer();
            for (int e = 0; e < n[(size_t) l]; ++e) {
                TierRing::Entry en{next % ring.slots(), next - ring.slots() + 1};
                if (!ungated) check(ring.take(en), "rig: the ring gives a layer within its cap every entry");
                ++next;
                uint8_t* from = src.data() + copies.size() * kSlot;
                std::memset(from, pattern(c, l, e), kSlot);
                js.push_back(TierCopy{from, kSlot, nullptr, l, e, host.data() + (size_t) en.slot * kSlot, en.free_at});
                copies.push_back({l, e, en.slot, pattern(c, l, e)});
            }
            land.left[l].store(n[(size_t) l], std::memory_order_relaxed);
        }
        first[kLayers] = ungated ? next : ring.entries();
        stager->start(std::move(js), false);
    }
    // Every copy of the chunk done, none waiting for a slot any more
    void stop() { stager->drain(); }
    bool up(int l) const { return __atomic_load_n(&words[l], __ATOMIC_ACQUIRE) == land.gen; }
    // The tier reads layer l: its word within the wait, then every slot it staged holds that layer's bytes
    bool read(int l, double wait_ms) {
        const Clock::time_point until = Clock::now() + std::chrono::microseconds((int64_t) (wait_ms * 1000));
        while (!up(l))
            if (Clock::now() > until) return false;
            else std::this_thread::yield();
        for (const Copy& cp : copies) {
            if (cp.l != l) continue;
            const uint8_t* s = host.data() + (size_t) cp.slot * kSlot;
            if (!std::all_of(s, s + kSlot, [&](uint8_t b) { return b == cp.byte; })) return false;
        }
        return true;
    }
};

void fits_half() {
    check(tier_layer_fits(16, 32) && tier_layer_fits(0, 2), "fits: a layer may stage half the ring");
    check(!tier_layer_fits(17, 32) && !tier_layer_fits(-1, 32), "fits: not more than half, and not a negative count");
    const TierRing r(32);
    check(r.slots() == 32 && r.cap() == 16, "fits: the cap is half the ring");
    check(tier_layer_fits(r.cap(), r.slots()) && !tier_layer_fits(r.cap() + 1, r.slots()), "fits: the cap is the most that fits");
}

void slots_in_pieces() {
    // a ring pinned in pieces: a slot is whole pages and a piece whole slots, so the card's copy of a slot never crosses into the next piece
    for (const uint64_t blob : {1ull, 4096ull, 4097ull, 1382400ull, 2621696ull, 5222400ull}) {
        const uint64_t slot = tier_slot_bytes(blob);
        check(slot >= blob && slot - blob < 4096 && slot % 4096 == 0, "pieces: a slot holds its blob in whole pages");
        for (const uint64_t want : {1ull << 20, 32ull << 20, 100ull << 20}) {
            const uint64_t piece = tier_piece_bytes(slot, want);
            check(piece % slot == 0 && piece % 4096 == 0 && piece >= slot, "pieces: a piece is whole slots and whole pages");
            check(piece <= std::max(want, slot) && (piece == slot || piece + slot > want), "pieces: as many slots as the wanted size holds");
            bool inside = true;
            for (uint64_t k = 0; k < 64; ++k) inside = inside && (k * slot) / piece == ((k + 1) * slot - 1) / piece;
            check(inside, "pieces: no slot crosses a piece");
        }
    }
}

void entries_in_order() {
    TierRing r(8);
    check(r.entries() == 0, "entries: none at first");
    for (int64_t k = 0; k < 10; ++k) {
        if (k % 4 == 0) r.start_layer();
        TierRing::Entry e{-1, -1};
        check(r.take(e), "entries: a layer within its cap takes each entry");
        check(e.slot == k % 8, "entries: entry k takes slot k mod the ring");
        check(e.free_at == k - 7, "entries: and waits until the tier read the entry a ring before it");
    }
    check(r.entries() == 10, "entries: counted");
}

void layer_gate() {
    // a layer takes at most half the ring: the take past it is refused and changes nothing, and the next layer takes again
    TierRing::Entry e{-1, -1};
    TierRing g(8);
    g.start_layer();
    for (int i = 0; i < 4; ++i) check(g.take(e), "gate: a layer takes up to its cap");
    check(e.slot == 3 && e.free_at == -4, "gate: the cap's last entry");
    const TierRing::Entry kept = e;
    check(!g.take(e), "gate: the take past the cap is refused");
    check(e.slot == kept.slot && e.free_at == kept.free_at && g.entries() == 4, "gate: a refused take changes no entry and takes none");
    check(!g.take(e) && g.entries() == 4, "gate: and stays refused for the rest of the layer");
    g.start_layer();
    check(g.take(e) && e.slot == 4 && e.free_at == -3 && g.entries() == 5, "gate: the next layer goes on from the entries taken");
    g.start_layer();
    g.start_layer();
    for (int i = 0; i < 4; ++i) check(g.take(e), "gate: an empty layer between leaves the next its whole cap");
    check(!g.take(e) && g.entries() == 9, "gate: and not more");
    TierRing none;
    none.start_layer();
    check(!none.take(e), "gate: a ring that was never sized takes nothing");
}

// Whether every entry of each layer after the first waits for no more than the layer before's first entry
bool lands_ahead(const std::vector<int>& n, int64_t slots) {
    TierRing r(slots);
    int64_t before = -1;
    for (int c : n) {
        const int64_t f = r.entries();
        r.start_layer();
        for (int i = 0; i < c; ++i) {
            TierRing::Entry e{};
            if (!r.take(e) || (before >= 0 && e.free_at > before)) return false;
        }
        before = f;
    }
    return true;
}

void half_lands_ahead() {
    // at the cap a layer's copies all land once the routing of the layer before freed its slots, so they overlap the tier's work
    check(lands_ahead({4, 4, 4, 3, 4}, 8), "ahead: layers at the cap land while the tier reads the layer before");
    check(!lands_ahead({4, 4, 5, 4}, 8), "ahead: a layer above the cap does not get its entries");
}

void two_chunks() {
    Rig rig(8, 2, 4);
    const std::vector<int> n[2] = {{4, 4, 2, 4}, {4, 0, 4, 1}};
    for (int c = 0; c < 2; ++c) {
        rig.chunk(c, n[c]);
        for (int l = 0; l < kLayers; ++l) {
            rig.land.free_to(rig.first[(size_t) l]);   // the routing of layer l: the tier read every layer before it
            if (n[c][(size_t) l] > 0) check(rig.read(l, 5000), "chunks: each layer's word comes up and its slots hold its bytes");
            else check(!rig.up(l), "chunks: a layer that staged nothing raises no word");
        }
        rig.stop();
    }
    check(rig.ring.entries() == 4 + 4 + 2 + 4 + 4 + 4 + 1, "chunks: the entries go on from chunk to chunk");
}

void stop_path() {
    // a request fails after layer 0: layer 2's copies wait for slots the tier never frees, and a stop lets them finish
    Rig rig(8, 2, 4);
    rig.chunk(0, {4, 4, 4, 0});
    rig.land.free_to(rig.first[0]);
    check(rig.read(0, 5000) && rig.read(1, 5000), "stop: the first two layers land into free slots");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    check(!rig.up(2), "stop: the third layer waits for the first to be read");
    rig.stop();
    check(rig.up(2), "stop: then its copies finish and nothing is left waiting");
}

void beyond_ring_waits() {
    // a layer that stages more than the ring waits for its own read: its word never comes until a stop
    Rig rig(8, 2, 4);
    rig.chunk(0, {9, 0, 0, 0}, true);
    rig.land.free_to(rig.first[0]);
    check(!rig.read(0, 100), "beyond: a layer above the ring does not land before the tier reads it");
    rig.stop();
    check(rig.up(0), "beyond: a stop releases it");
}

void layers_apart() {
    // one thread taking runs of up to 16 copies: a run stops at a layer's end, so a layer lands while the next one's copies still wait
    Rig rig(8, 1, 16);
    rig.chunk(0, {2, 4, 4, 0});
    rig.land.free_to(rig.first[0]);
    check(rig.read(0, 5000) && rig.read(1, 5000), "apart: the second layer lands with only the first one's routing");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    check(!rig.up(2), "apart: the third layer's copies wait for the first to be read");
    rig.land.free_to(rig.first[1]);
    check(rig.read(2, 5000), "apart: and land once it is");
    rig.stop();
}

void empty_chunk() {
    // a chunk that stages nothing starts no copies, and its drain returns at once
    Rig rig(8, 2, 4);
    rig.chunk(0, {4, 0, 0, 0});
    rig.land.free_to(rig.first[0]);
    check(rig.read(0, 5000), "empty: the chunk before lands");
    rig.stop();
    rig.chunk(1, {0, 0, 0, 0});
    rig.stop();
    check(!rig.up(0) && rig.ring.entries() == 4, "empty: no copy and no entry");
}

void card_ring_size() {
    // the card's ring: a layer's copies for every stage that may stream, as far as its room goes, and none under 16 slots
    const std::vector<int64_t> three{1, 1, 1};
    check(tier_card_ring(36, three, 99) == 36, "card ring: the room bounds it");
    check(tier_card_ring(1000, three, 99) == 297, "card ring: a layer's copies for each stage");
    check(tier_card_ring(1000, {1}, 99) == 99, "card ring: one stage");
    check(tier_card_ring(15, three, 99) == 0 && tier_card_ring(1000, three, 5) == 0, "card ring: none under 16 slots");
    check(tier_card_ring(16, three, 99) == 16, "card ring: 16 slots are enough");
    check(tier_card_ring(1000, {}, 99) == 0 && tier_card_ring(-1, three, 99) == 0, "card ring: no stages or no room");
    check(tier_card_ring(1000, {0, 531, 1525}, 128) == 257, "card ring: a stage with nothing to stream adds one slot, not a layer's");
}

void ring_slots() {
    // the staging ring: two layers of the most RAM experts one of the stage's layers holds, at most a 96th of the RAM available
    constexpr uint64_t kMiB = 1ull << 20, kGiB = 1ull << 30;
    check(tier_ring_slots(10, 512, 48 * kGiB, kMiB) == 20, "ring slots: a stage with few RAM experts gets two layers of them");
    check(tier_ring_slots(300, 512, 48 * kGiB, kMiB) == 256, "ring slots: a stage with many gets the cap, half the experts");
    check(tier_ring_slots(300, 512, 9 * kGiB, kMiB) == 96, "ring slots: or a 96th of the RAM available");
    check(tier_ring_slots(300, 512, 9 * kGiB + 96 * kMiB, kMiB) == 96, "ring slots: an even count");
    check(tier_ring_slots(0, 512, 48 * kGiB, kMiB) == 0, "ring slots: a stage with no RAM experts opens no ring");
    check(tier_ring_slots(10, 512, 2 * kGiB, kMiB) == 0, "ring slots: none when a 96th of the RAM holds under 32 slots");
    check(tier_ring_slots(10, 512, 48 * kGiB, 0) == 0 && tier_ring_slots(-1, 512, 48 * kGiB, kMiB) == 0, "ring slots: no slot size or a bad count");
    for (int64_t w = 1; w <= 300; ++w) {   // whatever it holds, a ring a TierRing takes, and a layer of all its RAM experts fits under the cap
        const int64_t n = tier_ring_slots(w, 512, 48 * kGiB, kMiB);
        if (n < 2 || n % 2 != 0 || n > 256 || (2 * w <= 256 && n != 2 * w)) check(false, "ring slots: even, at least 2, two layers under the cap");
    }
}

#if defined(__linux__)
// Each named copy thread's nice value, read from the process's own task list
std::vector<int> stage_thread_nice() {
    std::vector<int> out;
    DIR* d = opendir("/proc/self/task");
    if (d == nullptr) return out;
    while (const dirent* t = readdir(d)) {
        if (t->d_name[0] == '.') continue;
        char path[300], name[32] = {};
        std::snprintf(path, sizeof path, "/proc/self/task/%s/comm", t->d_name);
        if (FILE* f = std::fopen(path, "r")) {
            if (std::fgets(name, sizeof name, f) != nullptr && std::strncmp(name, "strata-stage", 12) == 0) {
                errno = 0;
                const int n = getpriority(PRIO_PROCESS, (id_t) std::atoi(t->d_name));
                if (errno == 0) out.push_back(n);
            }
            std::fclose(f);
        }
    }
    closedir(d);
    return out;
}
#endif

void copy_threads_yield() {
#if defined(__linux__)
    // the copy threads run below the thread that started them, so a prompt thread on the same core gets it first
    const int mine = getpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid));
    check(mine < 19, "copy threads: the test runs above the lowest priority");
    Rig rig(8, 3, 1);
    std::vector<int> nice;
    for (const Clock::time_point until = Clock::now() + std::chrono::seconds(2); Clock::now() < until;) {
        nice = stage_thread_nice();
        if (nice.size() == 3 && std::all_of(nice.begin(), nice.end(), [&](int n) { return n > mine; })) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(nice.size() == 3, "copy threads: three named threads");
    for (const int n : nice) check(n > mine && n <= 19, "copy threads: below the thread that started them");
#endif
}

void card_room() {
    // the smallest ring the stages' openers keep (the room the card opens before its experts fill is peer_room_test's)
    check(kTierCardMin == 16 && tier_card_ring(kTierCardMin, {1, 1, 1}, 128) == kTierCardMin && tier_card_ring(kTierCardMin - 1, {1, 1, 1}, 128) == 0,
          "card room: the smallest ring the card's ring rule opens");
}

void card_share_equal() {
    // 36 slots for three stages of equal weight: 12 each, apart, whichever stage opens first
    const std::vector<int64_t> even{1, 1, 1};
    for (int i = 0; i < 3; ++i) {
        const TierCardShare s = tier_card_share(36, even, i, 99);
        check(s.base == 12 * i && s.slots == 12, "card share: an equal part at the stage's place");
    }
    const TierCardShare one = tier_card_share(36, {1}, 0, 99), cap = tier_card_share(36, {1}, 0, 20);
    check(one.base == 0 && one.slots == 36 && cap.slots == 20, "card share: one stage takes the ring up to a layer");
}

void card_share_by_weight() {
    // 36 slots for stages of weight 3, 8 and 37 give 2, 6 and 28 (36 x 37 / 48 = 27.75 takes the slot left over)
    const std::vector<int64_t> layers{3, 8, 37};
    const TierCardShare a = tier_card_share(36, layers, 0, 128), b = tier_card_share(36, layers, 1, 128), c = tier_card_share(36, layers, 2, 128);
    check(a.base == 0 && a.slots == 2 && b.base == 2 && b.slots == 6 && c.base == 8 && c.slots == 28, "card share: parts by the stages' layers");
    // a ring of a layer's copies for each stage gives each a layer's, what the largest part's cap leaves going to the others
    for (int i = 0; i < 3; ++i) {
        const TierCardShare s = tier_card_share(384, layers, i, 128);
        check(s.base == 128 * i && s.slots == 128, "card share: a part capped at a layer leaves the rest to the others");
    }
    // every part capped at a layer once the ring holds a layer's copies for each, the 8 slots past them unused
    const TierCardShare d = tier_card_share(200, layers, 0, 64), e = tier_card_share(200, layers, 1, 64), f = tier_card_share(200, layers, 2, 64);
    check(d.slots == 64 && e.slots == 64 && f.slots == 64 && f.base == 128, "card share: every part at a layer when the ring holds them");
    const TierCardShare g = tier_card_share(150, layers, 0, 100), h = tier_card_share(150, layers, 1, 100), k = tier_card_share(150, layers, 2, 100);
    // the 3090's part capped at 100 leaves 50, shared 3 to 8: 13 and 36, and the slot left over to the larger remainder
    check(k.slots == 100 && g.slots == 14 && h.slots == 36 && h.base == 14 && k.base == 50, "card share: the rest of a capped part by weight");
    check(tier_card_share(36, {3, 0, 37}, 1, 128).slots == 1, "card share: a stage of no weight keeps one slot");
}

void card_share_box() {
    // the test box's RAM experts a stage at load (its layers' experts less its own and the tier's): NORMAL's 36 slots give the 3090 the most
    const std::vector<int64_t> normal{88, 531, 1525};
    const TierCardShare a = tier_card_share(36, normal, 0, 128), b = tier_card_share(36, normal, 1, 128), c = tier_card_share(36, normal, 2, 128);
    check(a.slots == 1 && b.base == 1 && b.slots == 9 && c.base == 10 && c.slots == 26, "card share: NORMAL by RAM experts, the 3090 the most");
    // RAMHEAVY's ring is a layer's copies for each stage whatever room the card has past it, so every stage keeps 128 there
    const std::vector<int64_t> ramheavy{116, 2792, 13435};
    bool keeps = true;
    for (int64_t room = 384; room <= 2000; room += 17) {
        const int64_t ring = tier_card_ring(room, ramheavy, tier_card_layer(512));
        for (int i = 0; i < 3; ++i) {
            const TierCardShare s = tier_card_share(ring, ramheavy, i, tier_card_layer(512));
            keeps = keeps && s.base == 128 * i && s.slots == 128;
        }
    }
    check(keeps, "card share: RAMHEAVY keeps 128 a stage");
    // a stage of many layers that holds all its experts streams nothing: it keeps one slot and no share, even of slots the others leave
    const std::vector<int64_t> none{0, 531, 1525};
    const TierCardShare held = tier_card_share(36, none, 0, 128), n1 = tier_card_share(36, none, 1, 128), n2 = tier_card_share(36, none, 2, 128);
    check(held.base == 0 && held.slots == 1 && n1.base == 1 && n1.slots == 9 && n2.base == 10 && n2.slots == 26,
          "card share: a stage with no RAM experts keeps one slot, the others share 35");
    const int64_t wide = tier_card_ring(1000, none, 128);
    const TierCardShare w0 = tier_card_share(wide, none, 0, 128), w1 = tier_card_share(wide, none, 1, 128), w2 = tier_card_share(wide, none, 2, 128);
    check(w0.slots == 1 && w1.base == 1 && w1.slots == 128 && w2.base == 129 && w2.slots == 128, "card share: no share of what the others leave");
    check(tier_card_share(384, none, 0, 128).slots == 1, "card share: one slot of a ring with room past the others' layers");
}

void card_share_order() {
    // the stages open in any order: each gets the same slots, and the first one sizes the ring from the room it reads
    const int order[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    const int64_t layer = tier_card_layer(512);
    for (const int64_t room : {36, 1000}) {   // a ring bound by the room, and one bound by the stages' layers
        std::vector<TierCardShare> first;
        for (const auto& o : order) {
            int64_t ring = 0;
            std::vector<TierCardShare> got(3);
            for (const int i : o) {
                if (ring == 0) ring = tier_card_ring(room, {3, 8, 37}, layer);
                got[(size_t) i] = tier_card_share(ring, {3, 8, 37}, i, layer);
            }
            if (first.empty()) first = got;
            bool same = true;
            for (int i = 0; i < 3; ++i) same = same && got[(size_t) i].base == first[(size_t) i].base && got[(size_t) i].slots == first[(size_t) i].slots;
            check(same, room == 36 ? "card share: the same split in every order" : "card share: the same split in every order, bound by layers");
        }
        check(first[0].slots >= 1 && first[1].slots >= 1 && first[2].slots >= 1 && first[2].base + first[2].slots <= tier_card_ring(room, {3, 8, 37}, layer),
              "card share: every stage streams");
        check(room != 1000 || (first[0].slots == 128 && first[2].base == 256), "card share: a ring bound by layers gives each a layer");
    }
    check(layer == 128 && tier_card_layer(256) == 64 && tier_card_layer(3) == 0, "card share: a layer's copies are the most any stage streams");
}

void card_copies() {
    // a stage's copies of a chunk: a layer's entries take slots of their own in its part, and no tag comes twice in any chunk or stream
    const TierCardShare part{12, 12};
    const uint64_t tags[2] = {tier_tag_base(0), tier_tag_base(1)};
    std::vector<uint64_t> seen;
    TierRing ring(24);
    const int64_t per_layer[] = {12, 5, 0, 12, 7, 12, 1};
    bool apart = true, inside = true;
    for (int chunk = 0; chunk < 3; ++chunk)
        for (const int64_t n : per_layer) {
            ring.start_layer();
            std::vector<char> used(36, 0);
            for (int64_t j = 0; j < n; ++j) {
                const int64_t k = ring.entries();
                TierRing::Entry en{};
                if (!ring.take(en)) break;
                for (const uint64_t base : tags) {
                    const TierCardCopy c = tier_card_copy(part, base, k);
                    seen.push_back(c.tag);
                    inside = inside && c.slot >= part.base && c.slot < part.base + part.slots;
                    if (base == tags[0]) apart = apart && used[(size_t) c.slot]++ == 0;
                }
            }
        }
    check(inside, "card copies: every copy in the stage's part");
    check(apart, "card copies: a layer's copies in slots of their own");
    std::sort(seen.begin(), seen.end());
    check(std::adjacent_find(seen.begin(), seen.end()) == seen.end() && seen.front() != 0, "card copies: no tag twice, and none 0");
}

void card_share_room() {
    // wherever the card has the 16 slots a ring needs, every stage gets some, apart from the others and inside the ring
    for (int64_t ring = 16; ring <= 300; ++ring)
        for (int stages = 1; stages <= 4; ++stages) {
            std::vector<char> owner((size_t) ring, 0);
            bool ok = true;
            const std::vector<int64_t> w = {3, 8, 37, 1};
            const std::vector<int64_t> weights(w.begin(), w.begin() + stages);
            for (int i = 0; i < stages; ++i) {
                const TierCardShare s = tier_card_share(ring, weights, i, 99);
                ok = ok && s.slots >= 1 && s.base >= 0 && s.base + s.slots <= ring;
                for (int64_t k = s.base; ok && k < s.base + s.slots; ++k) ok = owner[(size_t) k]++ == 0;
            }
            check(ok, "card share: every stage some slots, none shared");
        }
    // too small a ring or a bad place gives none
    check(tier_card_share(2, {1, 1, 1}, 0, 99).slots == 0 && tier_card_share(36, {1, 1, 1}, 3, 99).slots == 0, "card share: none past the ring or the stages");
    check(tier_card_share(36, {}, 0, 99).slots == 0 && tier_card_share(36, {1, 1, 1}, -1, 99).slots == 0, "card share: none without a place");
    check(tier_card_share(36, {1, 1, 1}, 0, 0).slots == 0, "card share: none for a stage that streams no layer");
}

}  // namespace

int main() {
    fits_half();
    slots_in_pieces();
    entries_in_order();
    layer_gate();
    half_lands_ahead();
    two_chunks();
    stop_path();
    beyond_ring_waits();
    layers_apart();
    empty_chunk();
    card_ring_size();
    ring_slots();
    card_room();
    copy_threads_yield();
    card_share_equal();
    card_share_by_weight();
    card_share_box();
    card_share_order();
    card_copies();
    card_share_room();
    std::printf("tier_landing_test: %d checks passed\n", checks);
    return 0;
}
