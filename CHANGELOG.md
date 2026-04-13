# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-04-13

### Added
- Zero-copy GPU DSP pipeline: upload once, process entirely on GPU, download once
- Batch cuFFT R2C/C2R via cudarc 0.19
- 5 CUDA kernels: Hann windowing, magnitude, 2D median filter, soft mask, overlap-add
- NVRTC runtime kernel compilation (< 0.1s)
- Configurable GPU architecture (`sm_75` through `sm_90`)
- Graceful fallback when `cuda` feature is disabled (`GpuDsp::new()` returns `None`)
- Full STFT/ISTFT pipeline with overlap-add reconstruction
- Dual license: MIT OR Apache-2.0
