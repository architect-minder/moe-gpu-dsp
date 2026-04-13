//! # moe-gpu-dsp
//!
//! MoE-routed GPU signal processing framework for Rust.
//!
//! Provides a zero-copy GPU pipeline for frequency-domain processing:
//! upload signal once, run all processing on GPU, download result once.
//!
//! Built on [cudarc](https://docs.rs/cudarc) for CUDA bindings and cuFFT for
//! batch FFT operations.
//!
//! ## Pipeline
//!
//! ```text
//! Signal -> [GPU upload] -> Window -> Batch FFT R2C -> Magnitude
//!   -> Processing kernels (median filter, soft mask, custom)
//!   -> Batch FFT C2R -> Overlap-add -> [GPU download] -> Output
//! ```
//!
//! ## Quick start
//!
//! ```ignore
//! use moe_gpu_dsp::{GpuDsp, DspConfig};
//!
//! let dsp = GpuDsp::new(DspConfig::default()).unwrap();
//! let (signal_dev, windowed, n_frames) = dsp.window_frames(&signal);
//! let complex = dsp.batch_fft_r2c(&windowed, n_frames);
//! // ... run processing kernels on GPU ...
//! let output = dsp.batch_ifft_c2r_ola(&mut masked, n_frames, signal.len());
//! ```

#[cfg(feature = "cuda")]
mod kernels;
#[cfg(feature = "cuda")]
mod pipeline;

#[cfg(feature = "cuda")]
pub use pipeline::{DspConfig, GpuDsp};

#[cfg(feature = "cuda")]
pub use kernels::{
    KERNEL_MAGNITUDE, KERNEL_MEDIAN_FILTER, KERNEL_OVERLAP_ADD, KERNEL_SOFT_MASK,
    KERNEL_WINDOW_FRAMES,
};

// Re-export cudarc types callers need
#[cfg(feature = "cuda")]
pub use cudarc::cufft::sys as cufft_sys;
#[cfg(feature = "cuda")]
pub use cudarc::driver::{CudaFunction, CudaSlice, CudaView, LaunchConfig, PushKernelArg};

/// Stub when CUDA feature is not enabled.
#[cfg(not(feature = "cuda"))]
pub struct GpuDsp;

#[cfg(not(feature = "cuda"))]
impl GpuDsp {
    pub fn new(_config: DspConfig) -> Option<Self> {
        None
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Clone, Debug, Default)]
pub struct DspConfig;
