//! Generic CUDA DSP kernels.
//!
//! Standard building blocks for frequency-domain processing pipelines.
//! No application-specific logic. Compiled once via NVRTC at init.

/// Hann window applied to overlapping frames.
/// Each thread handles one sample in one frame.
pub const KERNEL_WINDOW_FRAMES: &str = r#"
extern "C" __global__ void window_frames(
    const float* __restrict__ signal,
    float* __restrict__ output,
    const int signal_len,
    const int frame_size,
    const int hop,
    const int n_frames
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_frames * frame_size;
    if (idx >= total) return;

    int frame = idx / frame_size;
    int sample = idx % frame_size;
    int src = frame * hop + sample;

    float val = (src < signal_len) ? signal[src] : 0.0f;
    float w = 0.5f * (1.0f - cosf(2.0f * 3.14159265359f * (float)sample / (float)(frame_size - 1)));
    output[idx] = val * w;
}
"#;

/// Compute magnitude from complex R2C output, transposed for frequency-axis processing.
/// Input: [n_frames][n_bins] as interleaved float2 (re, im).
/// Output: [n_bins][n_frames] (transposed).
pub const KERNEL_MAGNITUDE: &str = r#"
extern "C" __global__ void magnitude(
    const float* __restrict__ complex_data,
    float* __restrict__ mag_out,
    const int n_bins,
    const int n_frames
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_bins * n_frames;
    if (idx >= total) return;

    int frame = idx / n_bins;
    int bin = idx % n_bins;
    int src = (frame * n_bins + bin) * 2;
    float re = complex_data[src];
    float im = complex_data[src + 1];

    int dst = bin * n_frames + frame;
    mag_out[dst] = sqrtf(re * re + im * im);
}
"#;

/// 2D median filter on a [rows][cols] matrix.
/// Supports horizontal (across cols) and vertical (across rows) directions.
/// Insertion sort for the sliding window (optimal for small kernel sizes <= 31).
pub const KERNEL_MEDIAN_FILTER: &str = r#"
extern "C" __global__ void median_filter(
    const int n_rows,
    const int n_cols,
    const int kernel_size,
    const int horizontal,
    const float* __restrict__ input,
    float* __restrict__ output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_rows * n_cols;
    if (idx >= total) return;

    int row = idx / n_cols;
    int col = idx % n_cols;
    int half = kernel_size / 2;

    float window[31];
    int count = 0;

    int center, axis_len;
    if (horizontal) {
        center = col;
        axis_len = n_cols;
    } else {
        center = row;
        axis_len = n_rows;
    }

    int start = (center >= half) ? center - half : 0;
    int end = (center + half + 1 < axis_len) ? center + half + 1 : axis_len;

    for (int i = start; i < end; i++) {
        int src_idx;
        if (horizontal) {
            src_idx = row * n_cols + i;
        } else {
            src_idx = i * n_cols + col;
        }
        float val = input[src_idx];
        int j = count;
        while (j > 0 && window[j - 1] > val) {
            window[j] = window[j - 1];
            j--;
        }
        window[j] = val;
        count++;
    }

    output[idx] = window[count / 2];
}
"#;

/// Soft mask: a^2 / (a^2 + b^2 + epsilon), applied to complex frequency-domain data.
/// a_mag and b_mag are [n_bins][n_frames]. complex data is [n_frames][n_bins] as float2.
/// Generic formulation: callers decide what a and b represent.
pub const KERNEL_SOFT_MASK: &str = r#"
extern "C" __global__ void soft_mask(
    const float* __restrict__ a_mag,
    const float* __restrict__ b_mag,
    const float* __restrict__ complex_in,
    float* __restrict__ complex_out,
    const int n_bins,
    const int n_frames
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_bins * n_frames;
    if (idx >= total) return;

    int frame = idx / n_bins;
    int bin = idx % n_bins;

    int mag_idx = bin * n_frames + frame;
    float aa = a_mag[mag_idx] * a_mag[mag_idx];
    float bb = b_mag[mag_idx] * b_mag[mag_idx];
    float mask = aa / (aa + bb + 1e-10f);

    int c_idx = (frame * n_bins + bin) * 2;
    complex_out[c_idx]     = complex_in[c_idx]     * mask;
    complex_out[c_idx + 1] = complex_in[c_idx + 1] * mask;
}
"#;

/// Overlap-add for ISTFT reconstruction.
/// Accumulates Hann-windowed frames into output signal using atomicAdd.
pub const KERNEL_OVERLAP_ADD: &str = r#"
extern "C" __global__ void overlap_add(
    const float* __restrict__ frames,
    float* __restrict__ output,
    const int frame_size,
    const int hop,
    const int n_frames,
    const int output_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_frames * frame_size;
    if (idx >= total) return;

    int frame = idx / frame_size;
    int sample = idx % frame_size;
    int dst = frame * hop + sample;
    if (dst >= output_len) return;

    float w = 0.5f * (1.0f - cosf(2.0f * 3.14159265359f * (float)sample / (float)(frame_size - 1)));
    atomicAdd(&output[dst], frames[idx] * w);
}
"#;
