#include "quantization.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace llm {
Quantized quantize(const std::vector<float> &w, size_t r, size_t c, size_t g) {
    if (!r || !c || !g || r > std::numeric_limits<size_t>::max() / c || w.size() != r * c)
        throw std::invalid_argument("quantization shape");
    size_t ng = c / g + (c % g != 0);
    Quantized q{r, c, g, std::vector<int8_t>(w.size()), std::vector<float>(r * ng)};
    for (size_t row = 0; row < r; ++row)
        for (size_t k = 0; k < ng; ++k) {
            size_t begin = k * g, end = begin + std::min(g, c - begin);
            float a = 0;
            for (size_t j = begin; j < end; ++j) {
                float v = w[row * c + j];
                if (!std::isfinite(v))
                    throw std::invalid_argument("nonfinite weight");
                a = std::max(a, std::abs(v));
            }
            float s = a == 0 ? 1 : a / 127;
            q.scales[row * ng + k] = s;
            for (size_t j = begin; j < end; ++j)
                q.values[row * c + j] = static_cast<int8_t>(
                    std::clamp(std::round(w[row * c + j] / s), -127.0f, 127.0f));
        }
    return q;
}
std::vector<float> dequantize(const Quantized &q) {
    if (!q.rows || !q.cols || !q.group || q.rows > std::numeric_limits<size_t>::max() / q.cols)
        throw std::invalid_argument("quantization metadata");
    size_t ng = q.cols / q.group + (q.cols % q.group != 0);
    if (q.values.size() != q.rows * q.cols || q.scales.size() != q.rows * ng)
        throw std::invalid_argument("quantization data size");
    std::vector<float> w(q.values.size());
    for (size_t r = 0; r < q.rows; ++r)
        for (size_t c = 0; c < q.cols; ++c)
            w[r * q.cols + c] = q.values[r * q.cols + c] * q.scales[r * ng + c / q.group];
    return w;
}
} // namespace llm
