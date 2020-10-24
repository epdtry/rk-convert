#include <astcenc.h>
#include <cstdio>
#include <cassert>
#include <vector>

#define CHECK_ERR(err, msg) \
    do { \
        if (err) { \
            printf(msg ": error %d\n", (int)err); \
            return 1; \
        } \
    } while (0)

extern "C" {
    
int astcwrap_decode(
    uint8_t* input, size_t input_len,
    uint8_t* output, size_t output_len,
    int x_size, int y_size
) {
    astcenc_error err;

    astcenc_config cfg;
    err = astcenc_config_init(
        ASTCENC_PRF_LDR,                // profile
        8, 8, 1,                        // block_x/y/z
        ASTCENC_PRE_FAST,               // preset
        ASTCENC_FLG_DECOMPRESS_ONLY,    // flags
        cfg);
    CHECK_ERR(err, "astcenc_config_init");

    astcenc_context* ctx = nullptr;
	err = astcenc_context_alloc(cfg, 1 /* thread count*/, &ctx);
    CHECK_ERR(err, "astcenc_config_init");

    std::vector<uint8_t*> rows;
    for (int y = 0; y < y_size; ++y) {
        rows.push_back(&output[y * x_size * 4]);
    }
    std::vector<uint8_t**> planes;
    planes.push_back(rows.data());

    astcenc_image img = {
        (unsigned int)x_size,
        (unsigned int)y_size,
        1,          // dim_z
        0,          // dim_pad
        ASTCENC_TYPE_U8,
        planes.data(),
    };
    assert(output_len == img.dim_x * img.dim_y * img.dim_z * 4);
    astcenc_swizzle swizzle = {
        ASTCENC_SWZ_R,
        ASTCENC_SWZ_G,
        ASTCENC_SWZ_B,
        ASTCENC_SWZ_A,
    };
    err = astcenc_decompress_image(
        ctx,
        input, input_len,
        img,
        swizzle);
    CHECK_ERR(err, "astcenc_decompress_image");

    return 0;
}

}
