// Optional optimized CPU Top-K baseline with stable score/index ordering.
// Native API consumes FP32 scores; host FP16->FP32 conversion is counted in
// the Python benchmark end-to-end timing.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#if defined(_WIN32)
#define EXPORT_API extern "C" __declspec(dllexport)
#else
#define EXPORT_API extern "C" __attribute__((visibility("default")))
#endif

EXPORT_API int cats_topk_f32(const float* scores, std::int64_t n,
                             std::int64_t k, std::int64_t* out_idx) noexcept {
    if (scores == nullptr || out_idx == nullptr || n < 1 || k < 1 || k > n) {
        return 1;
    }
    try {
        for (std::int64_t i = 0; i < n; ++i) {
            if (!std::isfinite(scores[i])) return 2;
        }
        std::vector<std::int64_t> ids(static_cast<std::size_t>(n));
        std::iota(ids.begin(), ids.end(), static_cast<std::int64_t>(0));
        auto better = [&](std::int64_t a, std::int64_t b) {
            if (scores[a] > scores[b]) return true;
            if (scores[a] < scores[b]) return false;
            return a < b;
        };
        if (k < n) {
            std::nth_element(ids.begin(), ids.begin() + k, ids.end(), better);
        }
        std::sort(ids.begin(), ids.begin() + k, better);
        std::copy_n(ids.begin(), k, out_idx);
        return 0;
    } catch (...) {
        return 3;
    }
}
