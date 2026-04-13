# moe-gpu-dsp

[![crates.io](https://img.shields.io/crates/v/moe-gpu-dsp.svg)](https://crates.io/crates/moe-gpu-dsp)
[![docs.rs](https://docs.rs/moe-gpu-dsp/badge.svg)](https://docs.rs/moe-gpu-dsp)
[![CI](https://github.com/architect-minder/moe-gpu-dsp/actions/workflows/ci.yml/badge.svg)](https://github.com/architect-minder/moe-gpu-dsp/actions)

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
moe-gpu-dsp = { version = "0.1", features = ["cuda"] }
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

## Setup

### 1. Install CUDA toolkit

**Linux / WSL:**
```bash
# Ubuntu/Debian
sudo apt-get install -y cuda-toolkit-13-1

# Verify
nvcc --version
```

**Windows:** Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and install.

### 2. Set environment variables

The build needs to find your CUDA installation:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_PATH=/usr/local/cuda
```

If your CUDA toolkit version differs from your driver version (common on WSL), pin it:

```bash
# Check driver version: nvidia-smi
# Set toolkit version to match driver (e.g. driver 13.1 = 13010)
export CUDARC_CUDA_VERSION=13010
```

### 3. Add to your project

```toml
[dependencies]
moe-gpu-dsp = { version = "0.1", features = ["cuda"] }
```

### 4. Build

```bash
cargo build --features cuda
```

### 5. Run

At runtime, the CUDA libraries must be on `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cargo run --features cuda
```

### Configure GPU architecture

Default target is `sm_86` (RTX 3070/3080/3090). Change it in `DspConfig`:

```rust
let config = DspConfig {
    gpu_arch: "sm_89",  // RTX 4090
    ..Default::default()
};
```

Common architectures:

| GPU | Architecture |
|-----|-------------|
| RTX 2070/2080 | `sm_75` |
| RTX 3070/3080/3090 | `sm_86` |
| RTX 4070/4080/4090 | `sm_89` |
| A100 | `sm_80` |
| H100 | `sm_90` |

### WSL-specific notes

- CUDA toolkit goes in WSL. Do NOT install Linux GPU drivers in WSL (the Windows driver bridges automatically).
- If you get `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, your toolkit version is newer than your driver. Install an older toolkit or update the Windows NVIDIA driver.
- cuFFT libraries are in `/usr/local/cuda/lib64/`. Make sure `LD_LIBRARY_PATH` includes this at runtime.

### Graceful fallback

Without the `cuda` feature, `GpuDsp::new()` returns `None`. You can fall back to CPU:

```rust
let dsp = GpuDsp::new(DspConfig::default());
if dsp.is_none() {
    // CPU fallback
}
```

## Performance

On RTX 3070, processing a 3-minute audio signal (7.4M samples, 14,562 STFT frames):

- NVRTC kernel compilation: 0.03s (cached after first run)
- Full pipeline (window + FFT + magnitude + 2x median filter + soft mask + IFFT + overlap-add): **99ms**

## License

MIT OR Apache-2.0
