#include "ssl-encoder.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool encode(const ssl_model & model, const std::vector<float> & features, int frames,
                   std::vector<float> & output) {
    if (!ssl_encode_features(model, features.data(), frames, output) ||
        output.size() != (size_t) frames * model.n_embd ||
        !std::all_of(output.begin(), output.end(), [](float x) { return std::isfinite(x); })) {
        fprintf(stderr, "teacher did not produce finite per-frame features\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: ssl-reference-state TEACHER.gguf [--gpu]\n");
        return 2;
    }
    const bool gpu = argc == 3 && std::strcmp(argv[2], "--gpu") == 0;
    if (argc == 3 && !gpu) return 2;
    ssl_model model;
    if (!ssl_load(model, argv[1], gpu)) return 1;
    if (gpu && ggml_backend_dev_type(ggml_backend_get_device(model.backend)) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        fprintf(stderr, "requested GPU is unavailable; refusing CPU fallback\n");
        ssl_free(model);
        return 1;
    }
    // Match a padded 20-second lane: its scratch reuse exposed corrupt cached indices.
    constexpr int frames = 999;
    std::vector<float> a(frames * 160), b(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = 0.2f * std::sin((float) (i + 1) * 0.123f) + 0.005f * (float) ((int) (i % 17) - 8);
        b[i] = -0.5f * a[i] + 0.02f * (float) (i % 7);
    }
    std::vector<float> first, other, repeated, shorter, restored;
    bool ok = encode(model, a, frames, first) && encode(model, b, frames, other) &&
              encode(model, a, frames, repeated);
    if (ok && (first == other || first != repeated)) {
        fprintf(stderr, "cached teacher changed a repeated input or ignored a new input\n");
        ok = false;
    }
    if (ok) {
        std::vector<float> short_input(a.begin(), a.begin() + 17 * 160);
        ok = encode(model, short_input, 17, shorter) && encode(model, a, frames, restored);
        if (ok && first != restored) {
            fprintf(stderr, "teacher graph-size transition corrupted persistent inputs\n");
            ok = false;
        }
    }
    ssl_free(model);
    if (ok) printf("teacher A/B/A replay and %d/17/%d graph-size transition: exact\n", frames, frames);
    return ok ? 0 : 1;
}
