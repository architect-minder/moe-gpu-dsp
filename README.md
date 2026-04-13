# moe-gpu-dsp

Zero-copy GPU signal processing framework for Rust. Batch cuFFT, CUDA kernel dispatch, and full STFT/IFFT pipelines that stay entirely on GPU memory.

Built on [cudarc](https://docs.rs/cudarc) 0.19.

## What it does

Upload a signal once. Run all processing on GPU. Download the result once. No CPU round-trips between stages.

```text
Signal -> GPU upload -> Hann window -> Batch FFT R2C
  -> Magnitude -> Processing kernels -> Soft mask
  -> Batch FFT C2R -> Overlap-add -> GPU download -> Output
```

## Kernels included

| Kernel | Purpose |
|--------|---------|
| `window_frames` | Hann windowing with hop-based frame extraction |
| `magnitude` | Complex to float magnitude with transpose |
| `median_filter` | 2D median filter (horizontal or vertical) |
| `soft_mask` | a^2 / (a^2 + b^2 + eps) applied to complex data |
| `overlap_add` | ISTFT reconstruction via atomicAdd |

All kernels are compiled once at init via NVRTC (typically < 0.1s).

## Usage

```toml
[dependencies]
moe-gpu-dsp = { git = "https://github.com/architect-minder/moe-gpu-dsp", features = ["cuda"] }
```

```rust
use moe_gpu_dsp::{GpuDsp, DspConfig};

// Init GPU + compile kernels
let dsp = GpuDsp::new(DspConfig::default()).unwrap();

// Upload + window
let (_sig, windowed, n_frames) = dsp.window_frames(&audio_signal);

// Batch FFT (all frames at once)
let complex = dsp.batch_fft_r2c(&windowed, n_frames);

// Magnitude (transposed for frequency-axis processing)
let mag = dsp.magnitude(&complex, n_frames);
let complex_floats = GpuDsp::complex_as_floats(&complex, n_frames * dsp.n_bins());

// Median filter (horizontal = time axis, vertical = frequency axis)
let h_filtered = dsp.median_filter(&mag, dsp.n_bins(), n_frames, 17, true);
let v_filtered = dsp.median_filter(&mag, dsp.n_bins(), n_frames, 17, false);

// Soft mask + IFFT + overlap-add
let mut masked = dsp.soft_mask(&h_filtered, &v_filtered, &complex_floats,
                                dsp.n_bins(), n_frames);
let output = dsp.batch_ifft_c2r_ola(&mut masked, n_frames, audio_signal.len());
```

## Requirements

- NVIDIA GPU (tested on RTX 3070, sm_86)
- CUDA toolkit installed (13.x)
- cudarc 0.19

Configure GPU architecture in `DspConfig::gpu_arch` (default: `"sm_86"`).

## Performance

On RTX 3070, processing a 3-minute audio signal (7.4M samples, 14,562 STFT frames):

- NVRTC kernel compilation: 0.03s (cached after first run)
- Full pipeline (window + FFT + magnitude + 2x median filter + soft mask + IFFT + overlap-add): **99ms**

## License

MIT OR Apache-2.0
