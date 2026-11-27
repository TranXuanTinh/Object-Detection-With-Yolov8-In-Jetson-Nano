#include "service/preprocessor.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstring>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace edge_ai {

namespace {

// Convert one canvas row of BGR (3 bytes/px) or BGRx (4 bytes/px) into planar RGB float32 [0, 1].
inline void convertRowToRgbPlanar(const uint8_t* row_ptr,
                                  int width,
                                  int channels,
                                  float* r_row,
                                  float* g_row,
                                  float* b_row) {
    const float inv_255 = 1.0f / 255.0f;
    int x = 0;

#if defined(__aarch64__) || defined(__ARM_NEON)
    const float32x4_t v_inv255 = vdupq_n_f32(inv_255);
    if (channels == 3) {
        for (; x + 16 <= width; x += 16) {
            uint8x16x3_t bgr = vld3q_u8(row_ptr + x * 3);
            uint8x16_t src_planes[3] = { bgr.val[2], bgr.val[1], bgr.val[0] }; // R, G, B
            float* dst_planes[3] = { r_row + x, g_row + x, b_row + x };
            for (int c = 0; c < 3; ++c) {
                uint16x8_t lo = vmovl_u8(vget_low_u8(src_planes[c]));
                uint16x8_t hi = vmovl_u8(vget_high_u8(src_planes[c]));
                vst1q_f32(dst_planes[c] + 0,  vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))), v_inv255));
                vst1q_f32(dst_planes[c] + 4,  vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo))), v_inv255));
                vst1q_f32(dst_planes[c] + 8,  vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))), v_inv255));
                vst1q_f32(dst_planes[c] + 12, vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi))), v_inv255));
            }
        }
    } else if (channels == 4) {
        for (; x + 16 <= width; x += 16) {
            uint8x16x4_t bgrx = vld4q_u8(row_ptr + x * 4);
            uint8x16_t src_planes[3] = { bgrx.val[2], bgrx.val[1], bgrx.val[0] }; // R, G, B
            float* dst_planes[3] = { r_row + x, g_row + x, b_row + x };
            for (int c = 0; c < 3; ++c) {
                uint16x8_t lo = vmovl_u8(vget_low_u8(src_planes[c]));
                uint16x8_t hi = vmovl_u8(vget_high_u8(src_planes[c]));
                vst1q_f32(dst_planes[c] + 0,  vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))), v_inv255));
                vst1q_f32(dst_planes[c] + 4,  vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo))), v_inv255));
                vst1q_f32(dst_planes[c] + 8,  vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))), v_inv255));
                vst1q_f32(dst_planes[c] + 12, vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi))), v_inv255));
            }
        }
    }
#endif

    for (; x < width; ++x) {
        const uint8_t* px = row_ptr + x * channels;
        b_row[x] = static_cast<float>(px[0]) * inv_255;
        g_row[x] = static_cast<float>(px[1]) * inv_255;
        r_row[x] = static_cast<float>(px[2]) * inv_255;
    }
}

} // namespace

Preprocessor::Preprocessor(int input_size)
    : input_size_(input_size) {
}

bool Preprocessor::process(const cv::Mat& src_bgr, std::vector<float>& dst_chw, PreprocessMeta& meta) {
    if (src_bgr.empty()) return false;

    const int channels = src_bgr.channels();
    if (channels != 3 && channels != 4) {
        return false;
    }

    meta.orig_width = src_bgr.cols;
    meta.orig_height = src_bgr.rows;

    // 1. Calculate letterbox scale & paddings
    float r = std::min(static_cast<float>(input_size_) / meta.orig_width,
                       static_cast<float>(input_size_) / meta.orig_height);

    int new_unpad_w = static_cast<int>(std::round(meta.orig_width * r));
    int new_unpad_h = static_cast<int>(std::round(meta.orig_height * r));

    meta.scale = r;
    meta.pad_x = (input_size_ - new_unpad_w) * 0.5f;
    meta.pad_y = (input_size_ - new_unpad_h) * 0.5f;

    // 2. Resize into reusable member buffer
    const cv::Mat* resized_ptr = &src_bgr;
    if (meta.orig_width != new_unpad_w || meta.orig_height != new_unpad_h) {
        cv::resize(src_bgr, resized_, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
        resized_ptr = &resized_;
    }

    // 3. Reuse square canvas and pad with neutral gray (114)
    canvas_.create(input_size_, input_size_, resized_ptr->type());
    canvas_.setTo(cv::Scalar(114, 114, 114, 255));
    int top = static_cast<int>(std::round(meta.pad_y));
    int left = static_cast<int>(std::round(meta.pad_x));
    resized_ptr->copyTo(canvas_(cv::Rect(left, top, new_unpad_w, new_unpad_h)));

    // 4. Fused BGR/BGRx -> RGB swap, [0, 1] normalization, and HWC -> CHW planar write
    size_t channel_size = static_cast<size_t>(input_size_ * input_size_);
    dst_chw.resize(3 * channel_size);

    float* r_plane = dst_chw.data();
    float* g_plane = dst_chw.data() + channel_size;
    float* b_plane = dst_chw.data() + 2 * channel_size;

    for (int y = 0; y < input_size_; ++y) {
        const uint8_t* row_ptr = canvas_.ptr<uint8_t>(y);
        int row_offset = y * input_size_;
        convertRowToRgbPlanar(row_ptr,
                              input_size_,
                              channels,
                              r_plane + row_offset,
                              g_plane + row_offset,
                              b_plane + row_offset);
    }

    return true;
}

// Convert float to FP16 representation
static uint16_t floatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 31) & 0x1;
    uint32_t exp = (x >> 23) & 0xFF;
    uint32_t mant = x & 0x7FFFFF;

    if (exp == 0) return static_cast<uint16_t>(sign << 15);
    if (exp == 0xFF) return static_cast<uint16_t>((sign << 15) | 0x7C00);

    int new_exp = static_cast<int>(exp) - 127 + 15;
    if (new_exp >= 31) return static_cast<uint16_t>((sign << 15) | 0x7C00);
    if (new_exp <= 0) return static_cast<uint16_t>(sign << 15);

    return static_cast<uint16_t>((sign << 15) | (new_exp << 10) | (mant >> 13));
}

bool Preprocessor::processFP16(const cv::Mat& src_bgr, std::vector<uint16_t>& dst_chw_fp16, PreprocessMeta& meta) {
    std::vector<float> float_buf;
    if (!process(src_bgr, float_buf, meta)) return false;

    dst_chw_fp16.resize(float_buf.size());
    for (size_t i = 0; i < float_buf.size(); ++i) {
        dst_chw_fp16[i] = floatToHalf(float_buf[i]);
    }
    return true;
}

} // namespace edge_ai

