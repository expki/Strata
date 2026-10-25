// A stand-in for the AMD tier library that answers with another interface version.
#include "strata/core/xtier_abi.hpp"

#if defined(_WIN32)
#define XTIER_STUB_EXPORT extern "C" __declspec(dllexport)
#else
#define XTIER_STUB_EXPORT extern "C" __attribute__((visibility("default")))
#endif

XTIER_STUB_EXPORT int strata_xtier_abi(void) { return 999 * 100000; }
