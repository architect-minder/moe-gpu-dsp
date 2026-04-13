//! GPU DSP pipeline: batch cuFFT, kernel dispatch, zero-copy processing.

use cudarc::driver::{CudaContext, CudaStream, CudaSlice, CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::cufft::{CudaFft, sys as cufft_sys};
use crate::kernels::*;

/// Pipeline configuration.
#[derive(Clone, Debug)]
pub struct DspConfig {
    /// FFT frame size (default: 2048).
    pub frame_size: usize,
    /// Hop size between frames (default: 512).
    pub hop: usize,
    /// CUDA GPU architecture target (default: "sm_86").
    pub gpu_arch: &'static str,
}

impl Default for DspConfig {
    fn default() -> Self {
        Self { frame_size: 2048, hop: 512, gpu_arch: "sm_86" }
    }
}

/// GPU-accelerated DSP pipeline.
///
/// Compiles all kernels at init, reuses across calls. All processing stays on
/// GPU memory between stages. Only the initial upload and final download touch
/// the CPU.
pub struct GpuDsp {
    pub stream: std::sync::Arc<CudaStream>,
    pub config: DspConfig,
    pub window_func: CudaFunction,
    pub magnitude_func: CudaFunction,
    pub median_func: CudaFunction,
    pub soft_mask_func: CudaFunction,
    pub overlap_add_func: CudaFunction,
}

impl GpuDsp {
    /// Initialize GPU, compile all DSP kernels via NVRTC.
    ///
    /// Returns `None` if no CUDA device is available or compilation fails.
    pub fn new(config: DspConfig) -> Option<Self> {
        let t = std::time::Instant::now();
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => { eprintln!("[moe-gpu-dsp] CUDA init failed: {e}"); return None; }
        };
        let stream = ctx.default_stream();

        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(config.gpu_arch),
            ..Default::default()
        };

        let src = format!(
            "{}\n{}\n{}\n{}\n{}",
            KERNEL_WINDOW_FRAMES, KERNEL_MAGNITUDE, KERNEL_MEDIAN_FILTER,
            KERNEL_SOFT_MASK, KERNEL_OVERLAP_ADD
        );

        let ptx = match cudarc::nvrtc::compile_ptx_with_opts(&src, opts) {
            Ok(p) => p,
            Err(e) => { eprintln!("[moe-gpu-dsp] NVRTC failed: {e}"); return None; }
        };

        let module = match ctx.load_module(ptx) {
            Ok(m) => m,
            Err(e) => { eprintln!("[moe-gpu-dsp] module load failed: {e}"); return None; }
        };

        let f = |name: &str| -> Option<CudaFunction> {
            match module.load_function(name) {
                Ok(f) => Some(f),
                Err(e) => { eprintln!("[moe-gpu-dsp] {name}: {e}"); None }
            }
        };

        let name = ctx.name().unwrap_or_else(|_| "unknown".into());
        eprintln!("[moe-gpu-dsp] {} ready ({:.2}s)", name, t.elapsed().as_secs_f64());

        Some(Self {
            stream,
            config,
            window_func: f("window_frames")?,
            magnitude_func: f("magnitude")?,
            median_func: f("median_filter")?,
            soft_mask_func: f("soft_mask")?,
            overlap_add_func: f("overlap_add")?,
        })
    }

    /// Launch configuration for a 1D grid of `total` threads.
    pub fn launch_cfg(total: usize) -> LaunchConfig {
        let bs = 256u32;
        LaunchConfig {
            grid_dim: (((total as u32) + bs - 1) / bs, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Number of frequency bins = frame_size / 2 + 1.
    pub fn n_bins(&self) -> usize { self.config.frame_size / 2 + 1 }

    /// Number of STFT frames for a signal of `len` samples.
    pub fn n_frames(&self, len: usize) -> usize {
        if len >= self.config.frame_size {
            (len - self.config.frame_size) / self.config.hop + 1
        } else { 0 }
    }

    /// Upload signal to GPU and apply Hann windowing.
    ///
    /// Returns (signal_dev, windowed_dev, n_frames).
    pub fn window_frames(&self, signal: &[f32]) -> (CudaSlice<f32>, CudaSlice<f32>, usize) {
        let len = signal.len();
        let nf = self.n_frames(len);
        let fs = self.config.frame_size;
        let total = nf * fs;

        let mut sig = self.stream.alloc_zeros::<f32>(len).unwrap();
        self.stream.memcpy_htod(signal, &mut sig).unwrap();

        let mut out: CudaSlice<f32> = self.stream.alloc_zeros(total).unwrap();
        unsafe {
            self.stream.launch_builder(&self.window_func)
                .arg(&sig).arg(&mut out)
                .arg(&(len as i32)).arg(&(fs as i32))
                .arg(&(self.config.hop as i32)).arg(&(nf as i32))
                .launch(Self::launch_cfg(total)).unwrap();
        }
        (sig, out, nf)
    }

    /// Batch R2C FFT on windowed frames.
    pub fn batch_fft_r2c(&self, windowed: &CudaSlice<f32>, n_frames: usize)
        -> CudaSlice<cufft_sys::float2>
    {
        let nb = self.n_bins();
        let plan = CudaFft::plan_1d(
            self.config.frame_size as i32,
            cufft_sys::cufftType::CUFFT_R2C,
            n_frames as i32,
            self.stream.clone(),
        ).expect("[moe-gpu-dsp] R2C plan failed");

        let mut out: CudaSlice<cufft_sys::float2> =
            self.stream.alloc_zeros(n_frames * nb).unwrap();
        plan.exec_r2c(windowed, &mut out).expect("[moe-gpu-dsp] R2C failed");
        out
    }

    /// Compute magnitude spectrogram from complex FFT output.
    ///
    /// Output is transposed: [n_bins][n_frames] for frequency-axis processing.
    pub fn magnitude(&self, complex: &CudaSlice<cufft_sys::float2>, n_frames: usize)
        -> CudaSlice<f32>
    {
        let nb = self.n_bins();
        let total = nb * n_frames;
        let complex_floats = unsafe {
            complex.transmute::<f32>(n_frames * nb * 2).unwrap()
        };
        let mut mag: CudaSlice<f32> = self.stream.alloc_zeros(total).unwrap();
        unsafe {
            self.stream.launch_builder(&self.magnitude_func)
                .arg(&complex_floats).arg(&mut mag)
                .arg(&(nb as i32)).arg(&(n_frames as i32))
                .launch(Self::launch_cfg(total)).unwrap();
        }
        mag
    }

    /// Reinterpret complex FFT output as raw float pairs for kernel access.
    pub fn complex_as_floats(complex: &CudaSlice<cufft_sys::float2>, count: usize)
        -> cudarc::driver::CudaView<'_, f32>
    {
        unsafe { complex.transmute::<f32>(count * 2).unwrap() }
    }

    /// Run median filter on a [rows x cols] matrix.
    ///
    /// `horizontal=true` filters across columns (time axis).
    /// `horizontal=false` filters across rows (frequency axis).
    pub fn median_filter(&self, input: &CudaSlice<f32>, n_rows: usize, n_cols: usize,
                         kernel_size: i32, horizontal: bool) -> CudaSlice<f32>
    {
        let total = n_rows * n_cols;
        let mut out: CudaSlice<f32> = self.stream.alloc_zeros(total).unwrap();
        let h: i32 = if horizontal { 1 } else { 0 };
        unsafe {
            self.stream.launch_builder(&self.median_func)
                .arg(&(n_rows as i32)).arg(&(n_cols as i32))
                .arg(&kernel_size).arg(&h)
                .arg(input).arg(&mut out)
                .launch(Self::launch_cfg(total)).unwrap();
        }
        out
    }

    /// Apply soft mask: a^2 / (a^2 + b^2 + eps) to complex frequency-domain data.
    pub fn soft_mask(&self, a_mag: &CudaSlice<f32>, b_mag: &CudaSlice<f32>,
                     complex_floats: &CudaSlice<f32>, n_bins: usize, n_frames: usize)
        -> CudaSlice<f32>
    {
        let total = n_bins * n_frames;
        let mut out: CudaSlice<f32> = self.stream.alloc_zeros(total * 2).unwrap();
        unsafe {
            self.stream.launch_builder(&self.soft_mask_func)
                .arg(a_mag).arg(b_mag)
                .arg(complex_floats).arg(&mut out)
                .arg(&(n_bins as i32)).arg(&(n_frames as i32))
                .launch(Self::launch_cfg(total)).unwrap();
        }
        out
    }

    /// Batch C2R IFFT + overlap-add. Returns time-domain signal on host.
    ///
    /// `masked_floats` is consumed (transmuted to complex for cuFFT).
    pub fn batch_ifft_c2r_ola(&self, masked_floats: &mut CudaSlice<f32>,
                               n_frames: usize, output_len: usize) -> Vec<f32>
    {
        let nb = self.n_bins();
        let fs = self.config.frame_size;
        let hop = self.config.hop;
        let total_w = n_frames * fs;

        let plan = CudaFft::plan_1d(
            fs as i32,
            cufft_sys::cufftType::CUFFT_C2R,
            n_frames as i32,
            self.stream.clone(),
        ).expect("[moe-gpu-dsp] C2R plan failed");

        let mut cx = unsafe {
            masked_floats.transmute_mut::<cufft_sys::float2>(n_frames * nb).unwrap()
        };
        let mut istft: CudaSlice<f32> = self.stream.alloc_zeros(total_w).unwrap();
        plan.exec_c2r(&mut cx, &mut istft).expect("[moe-gpu-dsp] C2R failed");

        let mut out: CudaSlice<f32> = self.stream.alloc_zeros(output_len).unwrap();
        unsafe {
            self.stream.launch_builder(&self.overlap_add_func)
                .arg(&istft).arg(&mut out)
                .arg(&(fs as i32)).arg(&(hop as i32))
                .arg(&(n_frames as i32)).arg(&(output_len as i32))
                .launch(Self::launch_cfg(total_w)).unwrap();
        }

        let mut result = self.stream.clone_dtoh(&out).unwrap();
        let scale = 1.0 / fs as f32;
        for v in &mut result { *v *= scale; }
        result
    }
}
