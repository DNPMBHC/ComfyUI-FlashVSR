# Changelog

All notable changes to ComfyUI-FlashVSR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added official ComfyUI node definitions for English and Simplified Chinese under `locales/`.
- Added `fix_method` selection with `wavelet` as the quality-first default and `adain` as an explicit alternative.

### Changed

- Updated repository, documentation, issue tracker, CLI, and Comfy Registry metadata for the maintained fork.
- Replaced the tracked local model-path configuration with `model_paths.yaml.example`; local `model_paths.yaml` files are now ignored.
- Sanitized the sample workflow by removing machine-specific input/output paths and stale node-source metadata.
- Removed the stale `node.zip` source snapshot from version control.
- Reduced validated Full-mode VAE choices to `Wan2.1` and `LightVAE_W2.1`; the official Wan2.2 VAE is incompatible with FlashVSR's latent contract.
- Clarified decoder ownership: Full mode uses the selected VAE; Tiny and Tiny-Long use the streaming TCDecoder.
- Quality defaults are now BF16 when supported, wavelet color correction, `kv_ratio=3.5`, and `local_range=11`.
- DiT spatial tiling and stateless external frame chunking are no longer enabled automatically. Nonzero CLI `frame_chunk_size` values are rejected because they reset temporal state.

### Performance

- Added architecture-aware `auto` attention selection with RTX 50 series support through mask-preserving Block Sparse Attention.
- Added explicit `sage_attention` and `flash_attention_3` backends while preserving explicit user selection.
- Added lazy binary-extension loading, version/capability reporting, and mask-preserving runtime fallbacks.
- FlashAttention 3 is restricted to known-compatible Hopper devices unless `FLASHVSR_ENABLE_EXPERIMENTAL_FA3=1` is set.

### Bug Fixes

- Optional attention packages no longer override the node-selected backend merely because they are installed.
- Broken or incompatible CUDA extensions fall back without being retried for every attention call.
- Attention backend fallbacks preserve the model's draft/local block-mask semantics.
- Full-mode decoder selection no longer silently advertises unsupported TAE/LightTAE mappings.
- Full and Tiny pipelines now preserve decoder temporal context, apply FP32 color correction, and reject invalid short internal frame sequences explicitly.
- CLI encoding now honors the selected FFmpeg codec/quality settings and exits nonzero after processing or finalization failures.


## [1.3.0] - 2026-02-03

### 🚀 New Features

- **Custom Model Paths Configuration**: Added `model_paths.yaml` configuration file to specify custom directories for FlashVSR models.
  - Allows storing models on different drives (e.g., D:, E:) without code modification
  - Supports Windows (forward slashes and backslashes), Linux, and Mac path formats
  - Automatic directory creation if path doesn't exist
  - Path expansion support (`~` for home directory, environment variables)

- **Auto-Download to Custom Paths**: Model auto-download now respects the custom path configured in `model_paths.yaml`.
  - When models don't exist, they automatically download to the configured directory
  - All model files (DiT, LQ_proj, TCDecoder, VAE) download to custom path
  - Falls back to default ComfyUI models directory if config is empty

### ⚡ Performance

- **Configuration Caching**: Thread-safe module-level caching prevents repeated file I/O operations
- **Optimized Path Resolution**: Configuration loaded once at startup and cached for all subsequent operations

### 🛠 Technical Implementation

- Added `load_model_paths_config()`: Thread-safe YAML configuration loader with caching
- Added `get_flashvsr_model_base_dir()`: Helper function to resolve model base directory
- Updated `model_download()` to use custom path from configuration
- Updated `init_pipeline()` to construct paths using configuration
- Thread safety implemented with `threading.Lock()` for concurrent node loading
- Security: Uses `yaml.safe_load()` to prevent code injection

### 📖 Documentation

- Added comprehensive `model_paths.yaml` with:
  - Detailed instructions for Windows/Linux/Mac users
  - Multiple practical examples for different scenarios
  - Auto-download workflow explanation
  - Turkish and English documentation
- Updated README.md with:
  - "Step 3: Custom Model Paths" installation section
  - Auto-download support callout with examples
  - CLI usage examples with `--models_dir` parameter
- Added inline code comments explaining the configuration flow

### 🔧 Dependencies

- Added `pyyaml` to requirements.txt for YAML configuration parsing

---

## [1.2.9] - 2026-01-25

### 🐛 Bug Fixes

- Users with high VRAM can now set keep_models_on_cpu=False to keep models in VRAM throughout processing, eliminating offload overhead between tiles (~30-40% speedup for tiled operations on 24GB+ GPUs).

## [1.2.8] - 2026-01-20

### 🐛 Bug Fixes

- FlashVSR nodes return half-precision tensors on CUDA, breaking downstream nodes that expect float32 on CPU. This causes type mismatch errors in FILM interpolator and other processing nodes.
- The tiled processing path creates final_output_canvas with dtype=torch.float16, causing accumulated results to remain in half precision despite tensor2video() converting to float32. Output tensors vary between fp16/bf16/CUDA depending on processing path.

## [1.2.7] - 2025-12-23

### ✨ New Features

- **5 New VAE Models**: Added support for `Wan2.2`, `LightVAE_W2.1`, `TAE_W2.2`, `LightTAE_HY1.5` (plus original `Wan2.1`).
- **Smart Resource Calculator**: Added "Pre-flight" check to analyze VRAM/RAM usage and recommend optimal settings before processing.
- **Auto-Download**: Added automatic downloading for missing VAE models from HuggingFace (`lightx2v/Autoencoders` repository).
- **Unified UI**: Replaced complex `vae_type` and `alt_vae` inputs with a single `vae_model` dropdown.

### 🐛 Bug Fixes

- **Fixed Black Borders**: Implemented correct "Padding → Process → Crop" logic to handle VAE dimension requirements without corrupting the output.
- **Fixed Video Glitches**: Corrected Tensor Permutation logic (`B,C,F,H,W` vs `B,H,W,C`) to prevent visual artifacts.
- **Fixed Model Loading**: Solved the issue where selecting "Wan2.2" incorrectly loaded "Wan2.1_VAE.pth".
- **Fixed OOM False Positive**: OOM recovery no longer triggers prematurely (adjusted threshold to 95%).

### ⚡ Optimization

- **Unified Pipeline**: Merged `tiny`, `full`, and `tiny-long` logic into a single optimized architecture.
- **OOM Protection**: Improved memory management with 95% VRAM threshold using `torch.cuda.mem_get_info()`.
- **Lossless Resize**: Uses `NEAREST` interpolation for integer scaling (0.5, 0.25) to avoid blur.
- **Aggressive Garbage Collection**: Added `torch.cuda.empty_cache()` before/after heavy VAE operations.

### 🛠 Refactoring

- **Explicit VAE Instantiation**: No more state_dict inspection/guessing - strict class mapping based on user selection.
- **VAE_MODEL_MAP**: Centralized configuration for all 5 VAE options (class, file, URL, dimensions).
- **Summary Logging**: End-of-processing report with total time, peak VRAM, and resolutions.
- **Debug Logging**: Shows `selected_model` vs `loaded_model_path` for verification.

### 📖 Documentation

- Updated README with VAE Selection Guide and VRAM tier recommendations.
- Added Best Practices section for Low/Medium/High VRAM configurations.
- Added Pre-Flight Resource Check documentation with example output.
- Added direct download links for all 5 VAE models.

---

## [1.1.0] - 2025-12-22

### 🚀 New Features

- **Wan2.2 VAE Support**: Integrated Wan2.2 VAE with optimized normalization statistics.
- **LightX2V VAE Integration**: Added LightX2V VAE for ~50% VRAM reduction and 2-3x faster inference.
- **VAE Type Selection**: Added `vae_model` dropdown in both Init Pipeline and Ultra-Fast nodes.
- **Factory Function**: Added `create_video_vae()` for programmatic VAE selection.

### ⚡ Performance

- **VRAM Reduction**: LightX2V reduces peak VRAM usage by approximately 50%.
- **Speed Improvement**: LightX2V provides 2-3x faster VAE decode times.

### 🛠 Refactoring

- **Backward Compatibility**: All new VAE types maintain full compatibility with existing Wan2.1 weights.
- **Architecture Constants**: Added `VAE_FULL_DIM`, `VAE_LIGHT_DIM`, `VAE_Z_DIM` for clarity.

---

## [1.0.3] - 2025-12-08

### ⚡ Performance

- **tiny-long Optimization**: Ported all VRAM optimizations and Tiled VAE support to `tiny-long` mode.
- **Windows Optimization**: Speed improvements for Windows environments.
- **Codebase Cleanup**: General synchronization and cleanup.

---

## [1.0.2] - 2025-12-07

### 🚀 New Features

- **Frame Chunking**: Added `frame_chunk_size` option to split large videos into chunks.
- **Attention Mode Selection**: Support for `flash_attention_2`, `sdpa`, `sparse_sage`, and `block_sparse`.
- **Debug Mode**: Added `enable_debug` option for extensive logging.

### 🐛 Bug Fixes

- **OOM Auto-Recovery**: If OOM occurs, automatically retries with `tiled_vae=True`, then `tiled_dit=True`.
- **Non-Tiled Output**: Fixed bug where output was undefined in non-tiled processing path.
- **Progress Bar**: Fixed display in ComfyUI using `cqdm` wrapper.
- **Full Mode VAE**: Added fallback mechanism to manually load VAE if model manager fails.

### ⚡ Performance

- **Deferred VAE Loading**: In `full` mode, VAE loading is deferred until strictly necessary.
- **90% VRAM Warning**: Added proactive warning when VRAM usage approaches 90%.
- **Memory Cleanup**: Added `torch.cuda.ipc_collect()` for better cleanup.

### 🛠 Refactoring

- **Progress Bar**: Rewrote to use single-line in-place updates (`\r`).
- **Default Settings**: Updated defaults for 16GB cards (`unload_dit=True`, tiled options enabled).
- **Documentation**: Expanded tooltips for all node parameters.

---

## [1.0.1] - 2025-12-06

### 🐛 Bug Fixes

- **Shape Mismatch**: Fixed error for small input frames with correct padding logic.
- **VRAM Cleanup**: VRAM immediately freed at start of processing to prevent OOM.

### 🚀 New Features

- **CPU Offload**: Added `keep_models_on_cpu` option to keep models in RAM.
- **FPS Reporting**: Added accurate FPS calculation and peak VRAM reporting.

### ⚡ Performance

- **Native PyTorch**: Replaced `einops` operations with native PyTorch ops where possible.
- **Conv3d Workaround**: Added memory optimization for Conv3d operations.

---

## [1.0.0] - 2025-10-21

### 🚀 Initial Release

- **Video Super Resolution**: ComfyUI node for FlashVSR video upscaling.
- **Three Modes**: `tiny`, `tiny-long`, and `full` processing modes.
- **Tiling Support**: Spatial tiling for VAE and DiT to reduce VRAM.
- **Sparse_Sage Attention**: Replaced Block-Sparse-Attention for RTX 50 series support.
- **Long Video Pipeline**: Streaming mode for very long videos with minimal VRAM spike.
