# ComfyUI-FlashVSR

**High-performance Video Super Resolution for ComfyUI with VRAM optimization.**

Run FlashVSR on 8GB-24GB+ GPUs with quality-first defaults, two validated Full-mode VAE decoders, and automatic model downloads.

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)

---

Maintained fork of [ComfyUI-FlashVSR_Stable](https://github.com/naxci1/ComfyUI-FlashVSR_Stable), published at [github.com/DNPMBHC/ComfyUI-FlashVSR](https://github.com/DNPMBHC/ComfyUI-FlashVSR).

The node interface includes ComfyUI's official `locales/en/nodeDefs.json` and `locales/zh/nodeDefs.json` translations.

---

## ✨ Key Features

- **🎬 Video Super Resolution**: 2x or 4x upscaling using FlashVSR diffusion models
- **🧠 Validated Decoders**: Full mode supports Wan2.1 and LightVAE_W2.1; Tiny and Tiny-Long use the streaming TCDecoder
- **📊 Pre-Flight Resource Check**: Intelligent VRAM estimation with settings recommendations
- **⚡ Auto-Download**: Models download automatically from HuggingFace if missing
- **🛡️ Quality-Safe OOM Handling**: Full mode can use native VAE tiling; DiT tiling and stateless frame chunking are never enabled automatically
- **🎨 Quality-First Defaults**: BF16 where supported, wavelet color correction, `kv_ratio=3.5`, and `local_range=11`
- **🔧 Unified Pipeline**: All modes share optimized processing logic

---

## 📋 Quick Links

- [Changelog](CHANGELOG.md) - Full version history
- [Sample Workflow](./workflow/FlashVSR.json)
- [HuggingFace Models](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1)

---

## Performance & VRAM Optimization

This node is optimized for various hardware configurations. Here are some guidelines:

### VRAM Tiers & Settings

| VRAM | Recommended Mode | Decoder / Tiling | Frame Processing | Precision | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **24GB+** | `full` | Wan2.1, tiling off | One stateful sequence | `bf16`/`auto` | Highest fidelity. |
| **16GB** | `full` or `tiny-long` | Full native VAE tiling if needed | One stateful sequence | `bf16`/`auto` | Enable `keep_models_on_cpu` before reducing resolution. |
| **12GB** | `tiny-long` or `full` + LightVAE | Full native VAE tiling if needed | One stateful sequence | `bf16`/`auto` | Keep `tiled_dit=False`; reduce `resize_factor` only if necessary. |
| **8GB** | `tiny-long` | TCDecoder streaming | One stateful sequence | `bf16`/`auto` | Prefer offload, then cautiously reduce `resize_factor`. |

### Performance Enhancements
- **Attention Mode**: Use `auto`. RTX 50 series prefers mask-preserving Block Sparse Attention so the model keeps the official draft/local topology. Dense Sage/FlashAttention backends still accelerate compatible cross-attention, while masked self-attention uses the mask-preserving SDPA fallback when necessary.
- **Precision**: `auto` selects `bf16` on supported GPUs. BF16 is the quality-first default for RTX 3000/4000/5000 series because it preserves more dynamic range than FP16.
- **Decoder Roles**: `full` uses the selected Wan/LightVAE decoder. `tiny` and `tiny-long` always use TCDecoder; their VAE selection does not change the decoder.
- **Long Videos**: Use `tiny-long` for stateful streaming. Stateless external `frame_chunk_size` processing resets temporal attention and decoder state, so it is disabled; the CLI rejects nonzero values.
- **Resize Input**: Keep `resize_factor=1.0` for full quality. Lower it only as a last-resort VRAM trade-off because downscaling permanently discards source detail before super-resolution.

### Pre-Flight Resource Check (NEW)

Before processing, FlashVSR now performs an intelligent pre-flight check that:

1. **Estimates VRAM Requirements**: Calculates approximate VRAM needed based on resolution, frames, scale, and settings.
2. **Checks Available Resources**: Uses `torch.cuda.mem_get_info()` for accurate real-time VRAM availability.
3. **Provides Recommendations**: If OOM is predicted, suggests optimal settings.

Example console output:
```
============================================================
🔍 PRE-FLIGHT RESOURCE CHECK
💻 RAM: 15.4GB / 95.8GB
💾 VRAM Available: 14.2GB
📊 Estimated VRAM Required: 12.8GB
✅ Safe to proceed. Estimated ~12.8GB needed, 14.2GB available.
============================================================
```

If VRAM is insufficient:
```
⚠️ Current settings require ~18.5GB but only 8.0GB available.
💡 Recommended Optimal Settings:
  • mode = tiny-long
  • keep_models_on_cpu = True
  • tiled_vae = True  (Full mode only)
  • resize_factor = 0.6  (last resort)
```

---

## 🎨 VAE Model Selection

The `vae_model` setting applies to **Full mode only**. Tiny and Tiny-Long use TCDecoder for their streaming decode path.

### VAE Type Comparison

| VAE Type | VRAM Usage | Speed | Quality | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Wan2.1** | 8-12 GB | Baseline | ⭐⭐⭐⭐⭐ | Maximum quality, 24GB+ VRAM |
| **LightVAE_W2.1** | 4-5 GB | 2-3x faster | ⭐⭐⭐⭐ | 8-16GB VRAM, speed priority |

The official Wan2.2 VAE is not selectable because it uses 48-channel latents with 16x spatial compression; FlashVSR produces 16-channel latents with 8x compression.

### VAE Selection Guide

| Your VRAM | Recommended Mode / Decoder | Additional Settings |
| :--- | :--- | :--- |
| **8GB** | `tiny-long` / TCDecoder | Keep models on CPU; reduce `resize_factor` only if still required |
| **12GB** | `tiny-long`, or `full` / LightVAE_W2.1 | Use native Full VAE tiling when needed |
| **16GB** | `full` / LightVAE_W2.1 or Wan2.1 | Native VAE tiling is optional |
| **24GB+** | `full` / Wan2.1 | Disable tiling for maximum quality |

### Auto-Download

Both supported Full-mode VAE models auto-download from HuggingFace if not found locally:

| VAE Selection | File | Direct Download Link |
| :--- | :--- | :--- |
| **Wan2.1** | `Wan2.1_VAE.pth` | [Download](https://huggingface.co/lightx2v/Autoencoders/blob/main/Wan2.1_VAE.pth) |
| **LightVAE_W2.1** | `lightvaew2_1.pth` | [Download](https://huggingface.co/lightx2v/Autoencoders/blob/main/lightvaew2_1.pth) |

---

## 📖 Best Practices / Settings Guide

### Low VRAM (8-12GB) Configuration

```
Mode: tiny-long
Decoder: TCDecoder (fixed for Tiny/Tiny-Long)
Tiled DiT: ❌ Disabled
Frame Chunk Size: 0 (stateful streaming)
Precision: BF16 / Auto
Color Fix: wavelet
Resize Factor: 1.0; lower only if offload is insufficient
Keep Models on CPU: ✅ Enabled
```

### Medium VRAM (16GB) Configuration

```
Mode: full or tiny-long
Full VAE: LightVAE_W2.1 or Wan2.1
Full Native VAE Tiling: Enable only if needed
Tiled DiT: ❌ Disabled
Frame Chunk Size: 0
Precision: BF16 / Auto
Color Fix: wavelet
KV Ratio / Local Range: 3.5 / 11
Resize Factor: 1.0
Keep Models on CPU: Optional
```

### High VRAM (24GB+) Configuration

```
Mode: full
VAE: Wan2.1
Tiled VAE: ❌ Disabled
Tiled DiT: ❌ Disabled
Frame Chunk Size: 0 (one stateful sequence)
Precision: BF16 / Auto
Color Fix: wavelet
KV Ratio / Local Range: 3.5 / 11
Resize Factor: 1.0
Keep Models on CPU: ❌ Disabled
```

### Processing Summary

At the end of each run, you'll see a summary:

```
============================================================
📊 PROCESSING SUMMARY
⏱️ Total Processing Time: 130.08s (1.54 FPS)
📥 Input Resolution: 276x206 (200 frames)
📤 Output Resolution: 552x412 (200 frames)
📈 Peak VRAM Used: 12.4 GB
============================================================
```

---

## 🔧 Node Parameters

Hover over any input in ComfyUI to see tooltips. Full parameter list:

| Parameter | Description |
| :--- | :--- |
| **model** | FlashVSR model version |
| **mode** | `tiny`/`tiny-long` use TCDecoder; `full` uses the selected VAE for highest quality |
| **vae_model** | Full-mode decoder: `Wan2.1` or `LightVAE_W2.1`; official Wan2.2 latents are incompatible |
| **scale** | Upscaling factor: 2x or 4x |
| **color_fix** | Enable color correction. |
| **fix_method** | `wavelet` (default, preserves high-frequency detail) or `adain` |
| **tiled_vae** | Native Full-mode VAE tiling. Reduces VRAM with a smaller quality impact than DiT tiling. |
| **tiled_dit** | Deprecated compatibility input. Nonzero values are ignored because independent DiT tiles lose full-frame context. |
| **tile_size** | Tile dimensions. Smaller = less VRAM. |
| **overlap** | Tile overlap for seamless blending. |
| **unload_dit** | Unload DiT before VAE decode. |
| **kv_ratio** | Key/value cache ratio. Default `3.5`; supported range `1.0-10.0`. |
| **local_range** | Local attention range: `7`, `9`, or `11` (default). |
| **frame_chunk_size** | Must remain `0`. Stateless external chunks reset temporal state; CLI nonzero values are rejected. |
| **enable_debug** | Verbose console logging. |
| **keep_models_on_cpu** | Offload to system RAM when idle. |
| **resize_factor** | Keep at `1.0` for full quality; lower values trade source detail for VRAM. |
| **attention_mode** | `auto` prefers mask-preserving Block Sparse on RTX 50. Dense backends accelerate compatible attention calls; masked self-attention preserves the official topology through Block Sparse or masked SDPA. |

---

## 💻 Command-Line Interface (CLI)

FlashVSR includes a full-featured CLI that mirrors all ComfyUI node parameters for standalone video upscaling.

### Quick Start

```bash
# Basic 2x upscale
python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2

# 4x Full-mode upscale with native VAE tiling for lower VRAM
python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 4 \
    --mode full --vae_model LightVAE_W2.1 --tiled_vae --unload_dit \
    --fix_method wavelet

# Long video with stateful streaming
python cli_main.py --input long_video.mp4 --output upscaled.mp4 \
    --mode tiny-long --fix_method wavelet

# Low VRAM last resort: reduce input only after Tiny-Long/offload is insufficient
python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2 \
    --mode tiny-long --resize_factor 0.75 --fix_method wavelet

# Custom models directory
python cli_main.py --input video.mp4 --output upscaled.mp4 \
    --models_dir /path/to/your/models
```

### CLI Arguments Reference

All arguments map 1:1 with ComfyUI node inputs. Run `python cli_main.py --help` for full details.

#### Required Arguments

| Argument | Description |
| :--- | :--- |
| `--input`, `-i` | Input video file path (e.g., `video.mp4`) |
| `--output`, `-o` | Output video file path (e.g., `upscaled.mp4`) |

#### Pipeline Initialization (from FlashVSRNodeInitPipe)

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | choice | `FlashVSR-v1.1` | Model version: `FlashVSR`, `FlashVSR-v1.1` |
| `--mode` | choice | `full` | Operation mode: `tiny`, `tiny-long`, `full` |
| `--vae_model` | choice | `Wan2.1` | Full-mode VAE decoder: `Wan2.1` or `LightVAE_W2.1`; Tiny modes use TCDecoder |
| `--force_offload` | flag | `True` | Force offload models to CPU after execution |
| `--no_force_offload` | flag | - | Disable force offloading |
| `--precision` | choice | `auto` | Precision: `fp16`, `bf16`, `auto`; auto selects BF16 when supported |
| `--device` | string | `auto` | Device: `cuda:0`, `cuda:1`, `cpu`, `auto` |
| `--attention_mode` | choice | `auto` | RTX 50 auto prefers mask-preserving Block Sparse; dense backends remain available for compatible attention calls |

#### Processing Parameters (from FlashVSRNodeAdv)

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--scale` | int | `2` | Upscaling factor: `2` or `4` |
| `--color_fix` | flag | `True` | Apply color correction |
| `--no_color_fix` | flag | - | Disable color correction |
| `--fix_method` | choice | `wavelet` | Color correction: `wavelet` or `adain` |
| `--tiled_vae` | flag | `False` | Enable native Full-mode VAE tiling |
| `--tiled_dit` | flag | `False` | Deprecated compatibility flag; ignored to preserve full-frame context |
| `--tile_size` | int | `256` | Tile size for DiT processing (32-1024) |
| `--tile_overlap` | int | `24` | Overlap pixels between tiles (8-512) |
| `--unload_dit` | flag | `False` | Unload DiT before VAE decoding |
| `--sparse_ratio` | float | `2.0` | Sparse attention control (1.5-2.0) |
| `--kv_ratio` | float | `3.5` | Key/value cache ratio (1.0-10.0) |
| `--local_range` | int | `11` | Local attention range: `7`, `9`, or `11` |
| `--seed` | int | `0` | Random seed for reproducibility |
| `--frame_chunk_size` | int | `0` | Must remain 0; CLI rejects nonzero stateless frame chunking |
| `--enable_debug` | flag | `False` | Enable verbose logging |
| `--keep_models_on_cpu` | flag | `True` | Keep models in CPU RAM when idle |
| `--no_keep_models_on_cpu` | flag | - | Keep models in VRAM |
| `--resize_factor` | float | `1.0` | Resize input before processing (0.1-1.0) |

#### Video I/O Parameters

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--fps` | float | input FPS | Output video FPS |
| `--codec` | string | `libx264` | Video codec: `libx264`, `libx265`, `h264_nvenc` |
| `--crf` | int | `18` | Quality (0-51, lower = better) |
| `--start_frame` | int | `0` | Start frame index (0-indexed) |
| `--end_frame` | int | `-1` | End frame index (-1 = all frames) |
| `--models_dir` | string | `./models` | Custom models directory path |

---

## 🚀 Installation

### Step 1: Install the Node

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DNPMBHC/ComfyUI-FlashVSR.git
python -m pip install -r ComfyUI-FlashVSR/requirements.txt
```

> 📢 **Turing architecture or older GPUs (GTX 16 series, RTX 20 series, and earlier)**: Install `triton<3.3.0`:
> ```bash
> # Windows
> python -m pip install -U triton-windows<3.3.0
> # Linux
> python -m pip install -U triton<3.3.0
> ```

### Step 2: Download Models

Download the `FlashVSR` folder from [HuggingFace](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1):

```
ComfyUI/models/FlashVSR/
├── LQ_proj_in.ckpt
├── TCDecoder.ckpt
├── diffusion_pytorch_model_streaming_dmd.safetensors
└── Wan2.1_VAE.pth  (or auto-downloads)
```

> 💡 **VAE files auto-download** from HuggingFace if not present. Only the DiT model and other components need manual download.

### Step 3: Custom Model Paths (Optional)

By default, FlashVSR looks for models in `ComfyUI/models/FlashVSR/`. To use a different location (e.g., models on another drive):

1. Copy `model_paths.yaml.example` to `model_paths.yaml` in the `ComfyUI-FlashVSR` directory
2. Set `flashvsr_model_path` to your custom path
3. Restart ComfyUI

**Example configurations:**

```yaml
# Windows (D: drive)
flashvsr_model_path: "D:/AI/Models/FlashVSR"

# Windows (alternative syntax)
flashvsr_model_path: "E:\\ComfyUI\\models\\FlashVSR"

# Linux/Mac
flashvsr_model_path: "/home/user/models/FlashVSR"
flashvsr_model_path: "/mnt/storage/AI/FlashVSR"

# Use default (leave empty)
flashvsr_model_path: ""
```

> 📂 **Auto-Download Support**: If model files don't exist, they will automatically download to the directory specified in `model_paths.yaml`. The custom path will be created if needed.
> 
> **Example**: If you set `flashvsr_model_path: "D:/AI/Models"`, models will automatically download to `D:/AI/Models/FlashVSR/` on first use.

---

## 🖼️ Preview

![Workflow Preview](./workflow/image1.png)

### Sample Workflow

[Download Workflow JSON](./workflow/FlashVSR.json)

---

## 🏷️ Recent Changes

See [CHANGELOG.md](CHANGELOG.md) for full version history.

---

## 🙏 Acknowledgments

- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) @OpenImagingLab  
- [Sparse_SageAttention](https://github.com/jt-zhang/Sparse_SageAttention_API) @jt-zhang
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) @Wan-Video
- [LightX2V](https://github.com/ModelTC/LightX2V) @ModelTC
- [LightX2V Autoencoders](https://huggingface.co/lightx2v/Autoencoders) @lightx2v

---

## 📄 License

GPLv3 License - see [LICENSE](LICENSE) for details. Components under `src/` retain their original Apache-2.0 licensing where applicable.
