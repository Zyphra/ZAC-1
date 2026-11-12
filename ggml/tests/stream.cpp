// Streaming decode with early partial-frame decode: complete frames are bit-identical next to partial ones, a missing
// residual code leaves its codebook's whole term out (table row and res.{c}.bias), and a codec without res.{i}.bias
// refuses missing cells but still decodes complete frames.
//
// usage: zac-stream <dummy-zac.gguf> [--gpu]   (fixture: scripts/make-dummy-zac.py; --gpu also runs the checks on the
// GPU backend)
#include "zac.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(cond, label) do { if (!(cond)) { fprintf(stderr, "FAIL: %s (%s:%d)\n", label, __FILE__, __LINE__); exit(1); } } while (0)

static constexpr int NCB = 10;   // the release geometry: 10 codebooks

static std::vector<int32_t> lcg_codes(int rows) {
    std::vector<int32_t> c((size_t) rows * NCB);
    uint32_t x = 12345;
    for (auto & v : c) { x = x * 1664525u + 1013904223u; v = (int32_t) ((x >> 8) % 4096); }
    return c;
}

static std::vector<float> stream(const zac_model & m, const std::vector<int32_t> & codes, int H, std::vector<int> ends,
                                 bool * ok = nullptr) {
    std::vector<float> pcm;
    zac_stream st;
    bool good = true;
    for (int e : ends) good = good && zac_stream_decode(m, st, codes.data(), H, e, pcm);
    good = good && zac_stream_flush(m, st, pcm);
    if (ok) *ok = good;
    else CHECK(good, "stream decode");
    return pcm;
}

static double max_diff(const std::vector<float> & a, const std::vector<float> & b, size_t n) {
    double d = 0;
    for (size_t i = 0; i < n; ++i) d = std::max(d, std::fabs((double) a[i] - (double) b[i]));
    return d;
}

static void test_codec(const char * zac_path, bool gpu) {
    zac_model m;
    CHECK(zac_load(m, zac_path, gpu), "load the dummy codec");
    CHECK(zac_partial_frames(m), "the dummy codec has res.{i}.bias");
    const int n = 12, L = m.latent_dim;
    const std::vector<int32_t> codes = lcg_codes(40);

    // Complete frames are bit-identical through the masked graph: decoding [0, n) with frame n - 1's last codebook
    // missing emits, ahead of what that frame touches, exactly what decoding [0, n - 1) alone emits.
    {
        std::vector<float> part, full;
        zac_stream a, b;
        CHECK(zac_stream_decode(m, a, codes.data(), n + NCB - 2, n, part), "partial chunk");
        CHECK(zac_stream_decode(m, b, codes.data(), 40, n - 1, full), "complete chunk");
        CHECK(!full.empty() && part.size() >= full.size(), "the partial chunk emits as far");
        CHECK(max_diff(part, full, full.size()) == 0.0, "complete frames bit-identical next to a partial one");
    }

    // A missing residual code drops its codebook's whole term: with table row X of every residual codebook set to
    // -res.{c}.bias (code X adds nothing), a frame missing its codes decodes like one coded X.
    const int X = 123;
    for (int c = 0; c + 1 < NCB; ++c) {
        std::vector<float> row(m.res_cb_bias.begin() + (size_t) c * L, m.res_cb_bias.begin() + (size_t) (c + 1) * L);
        for (float & v : row) v = -v;
        ggml_tensor * t = m.get("res." + std::to_string(c) + ".table");
        CHECK(t && t->type == GGML_TYPE_F32, "F32 residual table");
        ggml_backend_tensor_set(t, row.data(), (size_t) X * L * sizeof(float), (size_t) L * sizeof(float));
    }
    std::vector<int32_t> coded(codes.begin(), codes.begin() + (size_t) (n + NCB - 1) * NCB), minus = coded, zero = coded;
    for (int r = n; r < n + NCB - 1; ++r)
        for (int c = 1; c < NCB; ++c) {
            coded[(size_t) r * NCB + c] = X;
            minus[(size_t) r * NCB + c] = -1;
            zero[(size_t) r * NCB + c] = 0;
        }
    const std::vector<float> missing = stream(m, codes, n, {n});   // rows past H: frames n - 9 .. n - 1 partial
    const std::vector<float> neg = stream(m, minus, n + NCB - 1, {n}); // the same cells as -1
    const std::vector<float> as_x = stream(m, coded, n + NCB - 1, {n});
    const std::vector<float> as_0 = stream(m, zero, n + NCB - 1, {n});
    CHECK(missing.size() == as_x.size() && neg.size() == as_x.size() && as_0.size() == as_x.size(), "lengths");
    double peak = 0;
    for (float v : as_x) peak = std::max(peak, (double) std::fabs(v));
    CHECK(peak > 0, "nonzero audio");
    CHECK(max_diff(missing, neg, missing.size()) == 0.0, "rows past H and -1 codes are the same missing cells");
    const double d_x = max_diff(missing, as_x, missing.size()), d_0 = max_diff(missing, as_0, missing.size());
    fprintf(stderr, "codec: peak %.3e, missing vs code-X %.3e, missing vs code-0 %.3e\n", peak, d_x, d_0);
    CHECK(d_x <= 1e-4 * peak, "a missing code leaves its table row and res.{c}.bias out");
    CHECK(d_0 > 1e-2 * peak, "a missing code is not a code");

    // Without res.{i}.bias the codec cannot leave codebooks out: missing cells fail, complete frames still decode.
    zac_model old;
    CHECK(zac_clone(old, m), "clone");
    old.res_cb_bias.clear();
    CHECK(!zac_partial_frames(old), "no res.{i}.bias: no partial frames");
    bool ok = true;
    stream(old, codes, n, {n}, &ok);
    CHECK(!ok, "missing cells need res.{i}.bias");
    stream(old, codes, 40, {n}, &ok);
    CHECK(ok, "complete frames decode without res.{i}.bias");
    zac_free(old);
    zac_free(m);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <dummy-zac.gguf> [--gpu]\n", argv[0]);
        return 1;
    }
    test_codec(argv[1], /*gpu=*/false);
    if (argc > 2 && std::string(argv[2]) == "--gpu") test_codec(argv[1], /*gpu=*/true);
    printf("PASS\n");
    return 0;
}
