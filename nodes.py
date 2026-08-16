#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlashVSR ComfyUI Node - Video Super Resolution
===============================================
Supports the VAE implementations that are bundled and validated in this repo:
Wan2.1 and LightVAE_W2.1.

Key Fixes Applied:
- FIX 1: Merged VAE selection into a single validated 'vae_model' dropdown
- FIX 2: STRICT file path mapping - each VAE loads its own distinct file
- FIX 3: Black border fix - crop ONLY AFTER full VAE decode is complete
- FIX 4: Quality-preserving resize and crop semantics match the reference path
- FIX 5: VRAM optimization - 95% threshold before triggering OOM recovery
- FIX 6: Auto-download with CORRECT HuggingFace URLs
- FIX 7: Explicit VAE class instantiation - no guessing from state_dict
- FIX 8: Summary logging at end of processing
"""

import os, gc
import math
import torch
import folder_paths
import comfy.utils
import time
import sys
import psutil
import yaml
import threading

import numpy as np
import torch.nn.functional as F

from einops import rearrange
from huggingface_hub import snapshot_download, hf_hub_download
try:
    from .src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
    from .src.models.TCDecoder import build_tcdecoder
    from .src.models.utils import clean_vram, get_device_list, load_state_dict, normalize_checkpoint_state_dict, Buffer_LQ4x_Proj, Causal_LQ4x_Proj
    from .src.models.wan_video_dit import ATTENTION_MODES, attention_backend_status
    from .src.models.wan_video_vae import (
        WanVideoVAE, LightX2VVAE, create_video_vae,
        VAE_FULL_DIM, VAE_LIGHT_DIM, VAE_Z_DIM
    )
except ImportError:
    from src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
    from src.models.TCDecoder import build_tcdecoder
    from src.models.utils import clean_vram, get_device_list, load_state_dict, normalize_checkpoint_state_dict, Buffer_LQ4x_Proj, Causal_LQ4x_Proj
    from src.models.wan_video_dit import ATTENTION_MODES, attention_backend_status
    from src.models.wan_video_vae import (
        WanVideoVAE, LightX2VVAE, create_video_vae,
        VAE_FULL_DIM, VAE_LIGHT_DIM, VAE_Z_DIM
    )

try:
    import safetensors.torch
except ImportError:
    pass

# =============================================================================
# Decoder and quality options
# =============================================================================
VAE_MODEL_OPTIONS = ["Wan2.1", "LightVAE_W2.1"]
FIX_METHOD_OPTIONS = ["wavelet", "adain"]
ATTENTION_MODE_OPTIONS = list(ATTENTION_MODES)

# =============================================================================
# FIX 2 & 7: STRICT file path mapping with EXPLICIT class instantiation
# Each VAE selection loads a DISTINCT file and uses EXPLICIT class (no guessing)
# =============================================================================
VAE_MODEL_MAP = {
    "Wan2.1": {
        "class": WanVideoVAE, 
        "file": "Wan2.1_VAE.pth", 
        "internal_name": "wan2.1",
        "url": "https://huggingface.co/lightx2v/Autoencoders/resolve/main/Wan2.1_VAE.pth",
        "dim": VAE_FULL_DIM,
        "z_dim": VAE_Z_DIM,
        "use_full_arch": False
    },
    "LightVAE_W2.1": {
        "class": LightX2VVAE, 
        "file": "lightvaew2_1.pth",
        "internal_name": "lightx2v",
        "url": "https://huggingface.co/lightx2v/Autoencoders/resolve/main/lightvaew2_1.pth",
        "dim": VAE_LIGHT_DIM,
        "z_dim": VAE_Z_DIM,
        "use_full_arch": False
    },
}

# =============================================================================
# FIX 5: VRAM threshold for OOM recovery - set to 95%
# =============================================================================
VRAM_OOM_THRESHOLD = 0.95  # Only trigger OOM recovery when 95% VRAM is used

# =============================================================================
# Model Paths Configuration Loader
# =============================================================================
_cached_model_path = None  # Cache for model path to avoid repeated file reads
_config_loaded = False  # Flag to track if we've attempted to load config
_config_lock = threading.Lock()  # Thread-safe access to cached values

def load_model_paths_config():
    """
    Load model paths configuration from model_paths.yaml file.
    Returns the custom FlashVSR model path if configured, otherwise None.
    Uses caching to avoid repeated file I/O operations.
    Thread-safe implementation using a lock.
    """
    global _cached_model_path, _config_loaded
    
    # Return cached value if already loaded (thread-safe check)
    with _config_lock:
        if _config_loaded:
            return _cached_model_path
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "model_paths.yaml")
    
    # Check if file exists before entering try block
    if not os.path.exists(config_path):
        with _config_lock:
            _config_loaded = True
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        if config and isinstance(config, dict):
            flashvsr_path = config.get('flashvsr_model_path', '').strip()
            
            if flashvsr_path:
                # Expand user path (~/...) and environment variables
                flashvsr_path = os.path.expanduser(flashvsr_path)
                flashvsr_path = os.path.expandvars(flashvsr_path)
                
                # Convert to absolute path if it's not already
                # Use current_dir (plugin directory) as base for relative paths
                if not os.path.isabs(flashvsr_path):
                    flashvsr_path = os.path.abspath(os.path.join(current_dir, flashvsr_path))
                
                log(f"Custom FlashVSR model path loaded from config: {flashvsr_path}", 
                    message_type='info', icon="📂")
                
                with _config_lock:
                    _cached_model_path = flashvsr_path
                    _config_loaded = True
                return flashvsr_path
    except Exception as e:
        log(f"Warning: Could not load model_paths.yaml: {e}. Using default path.", 
            message_type='warning', icon="⚠️")
    
    with _config_lock:
        _config_loaded = True
    return None

device_choices = get_device_list()

def log(message: str, message_type: str = 'normal', icon: str = "", end: str = "\n", in_place: bool = False):
    if icon:
        message = f"{icon} {message}"
        
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    else:
        message = message

    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        message.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        message = message.encode(encoding, errors="replace").decode(encoding, errors="replace")

    if in_place:
        # Clear line before printing
        sys.stdout.write("\r\033[K" + message)
        sys.stdout.flush()
    else:
        print(message, end=end, flush=True)

def get_vram_info():
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return vram_used, vram_reserved, vram_total
    return 0, 0, 0

def log_resource_usage(prefix="Resource Usage", end="\n", in_place=False):
    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024 ** 3)
    ram_total = ram.total / (1024 ** 3)
    
    msg = f"[{prefix}] RAM: {ram_used:.1f}/{ram_total:.1f}G"
    
    if torch.cuda.is_available():
        vram_used, vram_reserved, vram_total = get_vram_info()
        msg += f" | VRAM: {vram_used:.1f}/{vram_reserved:.1f}/{vram_total:.1f}G"
        
    log(msg, message_type='info', icon="📊", end=end, in_place=in_place)


# =============================================================================
# FIX 5 & 9: VRAM Estimation, Pre-Flight Resource Check & Settings Recommender
# Calculate approximate VRAM requirements and provide optimal settings
# =============================================================================
def estimate_vram_usage(width, height, num_frames, scale, tiled_vae=False, tiled_dit=False, 
                         chunk_size=0, mode="full"):
    """
    Estimate approximate VRAM usage for the given video parameters.
    Returns estimated VRAM in GB. Enhanced to consider chunk_size and mode.
    
    =============================================================================
    FIX: Accurate VRAM Estimation with Safety Factor
    =============================================================================
    Previous estimates were ~4.5GB when actual usage was ~15GB.
    This was because we ignored:
    - Intermediate Activations: PyTorch stores outputs for every layer
    - VAE Upscaling: VAE decoding expands data significantly  
    - Workspace Memory: CUDA context overhead
    
    Solution: Apply Safety_Factor = 4.0 to the raw tensor calculations
    to account for these overheads.
    """
    # Safety factor to account for intermediate activations, VAE upscaling overhead,
    # and CUDA workspace memory. Empirically determined from observed ~15GB actual
    # usage when estimates were ~4.5GB.
    SAFETY_FACTOR = 4.0
    
    # Base model memory varies by mode
    if mode == "full":
        base_model_gb = 5.0  # Full VAE + DiT
    elif mode == "tiny-long":
        base_model_gb = 3.5  # TCDecoder is lighter than full VAE
    else:  # tiny
        base_model_gb = 4.0
    
    # Per-frame latent memory (scaled output resolution)
    output_h, output_w = height * scale, width * scale
    
    # Latent dimensions (8x downsampled)
    latent_h, latent_w = output_h // 8, output_w // 8
    
    # Frames to process at once (if chunked, use chunk_size)
    effective_frames = chunk_size if chunk_size > 0 and chunk_size <= num_frames else num_frames
    
    # Input tensor size - use 4 bytes to account for float32 intermediates during processing
    # Even though final tensors are bf16/fp16, operations often use float32 internally
    input_tensor_bytes = output_h * output_w * 3 * effective_frames * 4
    input_tensor_gb = (input_tensor_bytes * SAFETY_FACTOR) / (1024 ** 3)
    
    # Approximate memory per frame in latent space (16 channels, bf16)
    bytes_per_frame = latent_h * latent_w * 16 * 2  # bf16 = 2 bytes
    total_latent_gb = (bytes_per_frame * effective_frames * SAFETY_FACTOR) / (1024 ** 3)
    
    # DiT attention memory (quadratic with sequence length)
    seq_len = latent_h * latent_w * (effective_frames // 4)
    attention_gb = (seq_len * seq_len * 2 * SAFETY_FACTOR) / (1024 ** 3) * 0.001  # Rough estimate
    
    # VAE decode memory - this is where most intermediate activations live
    vae_decode_gb = (output_h * output_w * 3 * effective_frames * 2 * SAFETY_FACTOR) / (1024 ** 3)
    
    # Apply tiling reductions
    if tiled_dit:
        attention_gb *= 0.3  # Tiling reduces peak attention memory
    if tiled_vae:
        vae_decode_gb *= 0.4  # Tiling reduces peak VAE memory
    
    total_estimated = base_model_gb + input_tensor_gb + total_latent_gb + attention_gb + vae_decode_gb
    return total_estimated


def get_optimal_settings(width, height, num_frames, scale, available_vram_gb, mode="full"):
    """
    Calculate optimal settings (chunk_size, resize_factor, tiling) based on VRAM.
    
    Returns dict with recommended settings.
    """
    # Target VRAM usage: 85% of available to leave headroom
    target_vram = available_vram_gb * 0.85
    
    # Start with default settings
    recommended = {
        "chunk_size": 0,  # 0 = process all at once
        "resize_factor": 1.0,
        "tiled_vae": False,
        "tiled_dit": False,
        "warning": None
    }
    
    # Test current settings
    estimated = estimate_vram_usage(width, height, num_frames, scale, 
                                     tiled_vae=False, tiled_dit=False, 
                                     chunk_size=0, mode=mode)
    
    if estimated <= target_vram:
        # Settings are fine
        return recommended
    
    # Only the Full Wan VAE has a quality-preserving, decoder-native tiler.
    # Tiny modes use TCDecoder and must not be spatially sliced.
    use_tiled_vae = mode == "full"
    if use_tiled_vae:
        estimated_tiled_vae = estimate_vram_usage(
            width, height, num_frames, scale,
            tiled_vae=True, tiled_dit=False,
            chunk_size=0, mode=mode,
        )
        if estimated_tiled_vae <= target_vram:
            recommended["tiled_vae"] = True
            return recommended

    # External temporal chunking resets model state, and DiT spatial tiling
    # changes the model's global attention context. Prefer a lower working
    # resolution over silently changing inference semantics.
    for resize in [0.8, 0.6, 0.5, 0.4, 0.3]:
        new_h, new_w = int(height * resize), int(width * resize)
        estimated_resized = estimate_vram_usage(new_w, new_h, num_frames, scale,
                                                 tiled_vae=use_tiled_vae, tiled_dit=False,
                                                 chunk_size=0, mode=mode)
        if estimated_resized <= target_vram:
            recommended["tiled_vae"] = use_tiled_vae
            recommended["resize_factor"] = resize
            return recommended

    # Even with max reduction still risky
    recommended["tiled_vae"] = use_tiled_vae
    recommended["resize_factor"] = 0.3
    recommended["warning"] = (
        "VRAM is critically low. Use Tiny-Long or a lower input resolution; "
        "quality-changing DiT/chunk fallbacks are intentionally disabled."
    )
    return recommended


def check_resources(width, height, num_frames, scale, chunk_size, resize_factor, 
                    tiled_vae, tiled_dit, mode="full"):
    """
    =============================================================================
    FIX 9: Pre-Flight Resource Calculator
    =============================================================================
    
    Performs intelligent pre-flight check before loading heavy models.
    
    1. Gets hardware stats (VRAM, RAM) using torch.cuda.mem_get_info()
    2. Estimates required memory based on video parameters
    3. Simulates if current settings will cause OOM
    4. Provides optimal settings recommendations
    
    Returns:
        dict with keys:
        - estimated_vram_gb: float
        - available_vram_gb: float
        - ram_used_gb: float
        - ram_total_gb: float
        - will_oom: bool
        - recommended_settings: dict (if will_oom)
        - message: str
    """
    result = {
        "estimated_vram_gb": 0.0,
        "available_vram_gb": 0.0,
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "will_oom": False,
        "recommended_settings": None,
        "message": ""
    }
    
    # Get RAM info
    ram = psutil.virtual_memory()
    result["ram_used_gb"] = ram.used / (1024 ** 3)
    result["ram_total_gb"] = ram.total / (1024 ** 3)
    
    # Get VRAM info
    if torch.cuda.is_available():
        vram_free, vram_total = torch.cuda.mem_get_info()
        result["available_vram_gb"] = vram_free / (1024 ** 3)
        vram_total_gb = vram_total / (1024 ** 3)
    else:
        result["message"] = "CUDA not available. Running on CPU may be very slow."
        return result
    
    # Calculate effective dimensions after resize
    effective_h = int(height * resize_factor) if resize_factor < 1.0 else height
    effective_w = int(width * resize_factor) if resize_factor < 1.0 else width
    
    # Estimate VRAM usage
    result["estimated_vram_gb"] = estimate_vram_usage(
        effective_w, effective_h, num_frames, scale,
        tiled_vae=tiled_vae, tiled_dit=tiled_dit,
        chunk_size=chunk_size, mode=mode
    )
    
    # Check if OOM likely
    if result["estimated_vram_gb"] > result["available_vram_gb"] * VRAM_OOM_THRESHOLD:
        result["will_oom"] = True
        result["recommended_settings"] = get_optimal_settings(
            effective_w, effective_h, num_frames, scale, 
            result["available_vram_gb"], mode
        )
    
    # Build message
    if result["will_oom"]:
        rec = result["recommended_settings"]
        msg_parts = []
        if rec["chunk_size"] != chunk_size and rec["chunk_size"] > 0:
            msg_parts.append(f"chunk_size={rec['chunk_size']}")
        if rec["resize_factor"] != resize_factor:
            msg_parts.append(f"resize_factor={rec['resize_factor']:.1f}")
        if rec["tiled_vae"] and not tiled_vae:
            msg_parts.append("tiled_vae=True")
        if rec["tiled_dit"] and not tiled_dit:
            msg_parts.append("tiled_dit=True")
        
        if msg_parts:
            result["message"] = f"⚠️ Current settings require ~{result['estimated_vram_gb']:.1f}GB but only {result['available_vram_gb']:.1f}GB available. Recommended: {', '.join(msg_parts)}"
        else:
            result["message"] = f"⚠️ VRAM critically low. Estimated ~{result['estimated_vram_gb']:.1f}GB needed, only {result['available_vram_gb']:.1f}GB available."
    else:
        result["message"] = f"✅ Safe to proceed. Estimated ~{result['estimated_vram_gb']:.1f}GB needed, {result['available_vram_gb']:.1f}GB available."
    
    return result


def log_preflight_check(width, height, num_frames, scale, chunk_size, resize_factor,
                         tiled_vae, tiled_dit, mode="full"):
    """
    Log pre-flight resource check results.
    """
    result = check_resources(width, height, num_frames, scale, chunk_size, resize_factor,
                              tiled_vae, tiled_dit, mode)
    
    log("=" * 60, message_type='info')
    log("PRE-FLIGHT RESOURCE CHECK", message_type='info', icon="🔍")
    log(f"RAM: {result['ram_used_gb']:.1f}GB / {result['ram_total_gb']:.1f}GB", message_type='info', icon="💻")
    log(f"VRAM Available: {result['available_vram_gb']:.1f}GB", message_type='info', icon="💾")
    log(f"Estimated VRAM Required: {result['estimated_vram_gb']:.1f}GB", message_type='info', icon="📊")
    
    if result["will_oom"]:
        log(result["message"], message_type='warning', icon="⚠️")
        if result["recommended_settings"]:
            rec = result["recommended_settings"]
            log("Recommended Optimal Settings:", message_type='info', icon="💡")
            if rec["chunk_size"] > 0:
                log(f"  • chunk_size = {rec['chunk_size']}", message_type='info')
            if rec["resize_factor"] < 1.0:
                log(f"  • resize_factor = {rec['resize_factor']:.1f}", message_type='info')
            if rec["tiled_vae"]:
                log(f"  • tiled_vae = True", message_type='info')
            if rec["tiled_dit"]:
                log(f"  • tiled_dit = True", message_type='info')
            if rec.get("warning"):
                log(f"  ⚠️ {rec['warning']}", message_type='warning')
    else:
        log(result["message"], message_type='finish', icon="✅")
    
    log("=" * 60, message_type='info')
    
    return result


def log_vram_advisory(width, height, num_frames, scale, tiled_vae, tiled_dit, mode="full"):
    """
    Log advisory message about VRAM usage.
    Enhanced to use the new pre-flight check.
    """
    if not torch.cuda.is_available():
        return
    
    estimated_vram = estimate_vram_usage(width, height, num_frames, scale, tiled_vae, tiled_dit, mode=mode)
    available_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    current_used = torch.cuda.memory_allocated() / (1024 ** 3)
    free_vram = available_vram - current_used
    
    log(f"VRAM Advisory: Estimated ~{estimated_vram:.1f}GB needed, Available: {free_vram:.1f}GB free of {available_vram:.1f}GB total", 
        message_type='info', icon="💡")
    
    if estimated_vram > free_vram * 0.9:
        if mode == "full" and not tiled_vae:
            recommendation = "Enable the Wan VAE's native tiler or reduce the working resolution."
        else:
            recommendation = "Reduce the working resolution or use Tiny-Long."
        log(
            f"High VRAM usage expected. {recommendation}",
            message_type='warning', icon="⚠️",
        )
    elif estimated_vram < free_vram * 0.5:
        log("✅ Safe to proceed. VRAM usage should be comfortable.", message_type='info', icon="✅")

def get_flashvsr_model_base_dir():
    """
    Get the base directory for FlashVSR models.
    Checks model_paths.yaml first, falls back to ComfyUI models directory.
    """
    custom_path = load_model_paths_config()
    if custom_path:
        return custom_path
    return folder_paths.models_dir

def model_download(model_name="JunhaoZhuang/FlashVSR"):
    base_dir = get_flashvsr_model_base_dir()
    model_dir = os.path.join(base_dir, model_name.split("/")[-1])
    if not os.path.exists(model_dir):
        log(f"Downloading model '{model_name}' from huggingface...", message_type='info', icon="⬇️")
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)


# =============================================================================
# FIX 6: Auto-download VAE models if missing - UPDATED URLs
# =============================================================================
def download_vae_if_missing(vae_file: str, model_path: str, vae_config: dict) -> str:
    """
    Check if VAE file exists. If not, attempt to download it using the URL in vae_config.
    
    Args:
        vae_file: The filename of the VAE (e.g., 'Wan2.1_VAE.pth')
        model_path: The directory where VAE should be saved
        vae_config: The VAE configuration from VAE_MODEL_MAP (must contain 'url' key)
    
    Returns:
        Full path to the VAE file
    """
    vae_path = os.path.join(model_path, vae_file)
    
    if os.path.exists(vae_path):
        log(f"VAE file found: {vae_file}", message_type='info', icon="✅")
        return vae_path
    
    log(f"VAE file '{vae_file}' not found. Attempting auto-download...", message_type='warning', icon="⬇️")
    
    # Get URL from config (FIX 6: Use EXACT URLs from VAE_MODEL_MAP)
    url = vae_config.get("url")
    
    if url:
        try:
            log(f"Downloading from: {url}", message_type='info', icon="🌐")
            # Ensure directory exists
            os.makedirs(model_path, exist_ok=True)
            torch.hub.download_url_to_file(url, vae_path, progress=True)
            log(f"Successfully downloaded VAE: {vae_file}", message_type='finish', icon="✅")
            return vae_path
        except Exception as e:
            log(f"Download failed: {e}", message_type='error', icon="❌")
    
    raise RuntimeError(
        f'VAE file "{vae_file}" not found and auto-download failed.\n'
        f'Please manually download it and save to: {vae_path}\n'
        f'Download URL: {url}'
    )


# =============================================================================
# FIX 7: Fixed tensor2video for correct video output
# Ensures proper tensor permutation: VAE output (B, C, F, H, W) -> video (F, H, W, C)
# CRITICAL: This is called AFTER VAE decode is complete - no cropping here
# =============================================================================
def tensor2video(frames: torch.Tensor):
    """
    Convert VAE output tensor to video format.
    
    Input: (B, C, F, H, W) - Batch, Channels, Frames, Height, Width (VAE output)
    Output: (F, H, W, C) - Frames, Height, Width, Channels (video format)
    
    The tensor is normalized from [-1, 1] to [0, 1] for display.
    
    NOTE: This function does NOT crop - cropping happens in process_chunk() 
    AFTER this conversion is complete.
    """
    # Handle different input shapes
    if frames.dim() == 5:
        # Expected shape: (B, C, F, H, W)
        video_squeezed = frames.squeeze(0)  # (C, F, H, W)
        video_permuted = video_squeezed.permute(1, 2, 3, 0)  # (F, H, W, C)
    elif frames.dim() == 4:
        # Shape: (C, F, H, W) or (F, C, H, W) - need to detect
        if frames.shape[0] == 3 or frames.shape[0] == 4:
            # Likely (C, F, H, W)
            video_permuted = frames.permute(1, 2, 3, 0)  # (F, H, W, C)
        else:
            # Likely (F, C, H, W)
            video_permuted = frames.permute(0, 2, 3, 1)  # (F, H, W, C)
    else:
        raise ValueError(f"Unexpected tensor shape: {frames.shape}")
    
    # Normalize from [-1, 1] to [0, 1]
    video_final = (video_permuted.float() + 1.0) / 2.0
    # Clamp to valid range to avoid visual artifacts
    video_final = torch.clamp(video_final, 0.0, 1.0)
    
    return video_final

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def next_8n5(n):  # next 8n+5
    return 21 if n < 21 else ((n - 5 + 7) // 8) * 8 + 5

def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    """
    Compute scaled dimensions and target dimensions (aligned to multiple).
    
    =============================================================================
    FIX 3: Black Border Fix - Track original scaled dimensions
    =============================================================================
    Returns: sW, sH (scaled), tW, tH (model size), crop_left, crop_top.

    FlashVSR's reference path crops the scaled image to the largest valid
    multiple instead of reflecting pixels into the model context.  For inputs
    smaller than one model block we keep a single block and use edge padding as
    a robustness fallback.
    """
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    if sW < multiple:
        tW = multiple
    if sH < multiple:
        tH = multiple

    crop_left = max(0, (sW - tW) // 2)
    crop_top = max(0, (sH - tH) // 2)
    return sW, sH, tW, tH, crop_left, crop_top


def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int, crop_left: int, crop_top: int) -> torch.Tensor:
    """
    Upscale, then center-crop to the official model dimensions.
    """
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0) # HWC -> CHW -> BCHW
    
    sW, sH = w0 * scale, h0 * scale
    upscaled_tensor = F.interpolate(
        tensor_bchw,
        size=(sH, sW),
        mode='bicubic',
        align_corners=False,
        antialias=True,
    ).clamp_(0.0, 1.0)

    # Tiny inputs can be smaller than one model block. Replicate padding is a
    # bounded fallback; normal-sized inputs use a pure center crop.
    pad_left = max(0, (tW - sW) // 2)
    pad_right = max(0, tW - sW - pad_left)
    pad_top = max(0, (tH - sH) // 2)
    pad_bottom = max(0, tH - sH - pad_top)
    if pad_left or pad_right or pad_top or pad_bottom:
        upscaled_tensor = F.pad(
            upscaled_tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='replicate',
        )

    cropped_tensor = upscaled_tensor[
        :, :, crop_top:crop_top + tH, crop_left:crop_left + tW
    ]

    return cropped_tensor.squeeze(0)


def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    """
    Prepare input tensor with proper padding tracking.
    
    =============================================================================
    FIX 3: Black Border Fix - Track padding for later cropping
    =============================================================================
    Returns: vid_final, tH, tW, F, output_H, output_W, crop_top, crop_left
    """
    if image_tensor.ndim != 4:
        raise ValueError(
            f"FlashVSR input must have shape (frames, height, width, channels), got {tuple(image_tensor.shape)}."
        )
    N0, h0, w0, channels = image_tensor.shape
    if N0 < 1:
        raise ValueError("FlashVSR requires at least one input frame.")
    if channels != 3:
        raise ValueError(f"FlashVSR requires RGB input with 3 channels, got {channels}.")
    
    multiple = 128 # Keep 128 alignment for VAE/DiT blocks
    sW, sH, tW, tH, crop_left, crop_top = compute_scaled_and_target_dims(
        w0, h0, scale=scale, multiple=multiple
    )
    num_frames_with_padding = N0 + 4
    F = largest_8n1_leq(num_frames_with_padding)
    
    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")
    
    frames = []
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_center_crop(
            frame_slice,
            scale=scale,
            tW=tW,
            tH=tH,
            crop_left=crop_left,
            crop_top=crop_top,
        ).to('cpu').to(dtype) * 2.0 - 1.0
        frames.append(tensor_chw)
        del frame_slice

    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    
    del vid_stacked
    clean_vram()
    
    # The model sees the official cropped dimensions, so its output should be
    # consumed at tH/tW without trying to reinsert reflected padding.
    return vid_final, tH, tW, F, tH, tW, 0, 0

def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []
    
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride
            
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
                
            coords.append((x1, y1, x2, y2))
            
    return coords

def create_feather_mask(size, overlap, fade_left=True, fade_right=True, fade_top=True, fade_bottom=True):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    overlap = max(0, min(int(overlap), H // 2, W // 2))
    if overlap == 0:
        return mask

    ramp = torch.linspace(0, 1, overlap)
    if fade_left:
        mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    if fade_right:
        mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    if fade_top:
        mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    if fade_bottom:
        mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask

def init_pipeline(model, mode, device, dtype, vae_model="Wan2.1", attention_mode="auto"):
    """
    Initialize FlashVSR pipeline with specified model and VAE type.
    
    =============================================================================
    FIX 2 & 7: STRICT VAE file path mapping with EXPLICIT class instantiation
    =============================================================================
    Full mode uses the selected, validated Wan VAE. Tiny and Tiny-Long use
    TCDecoder by design and do not download or instantiate an unused VAE.
    """
    if mode == "full" and vae_model not in VAE_MODEL_MAP:
        if vae_model == "Wan2.2":
            raise ValueError(
                "Official Wan2.2 VAE is incompatible with FlashVSR: it uses "
                "48-channel latents at 16x spatial compression, while the "
                "FlashVSR DiT produces 16-channel latents at 8x compression."
            )
        raise ValueError(
            f"Unsupported VAE '{vae_model}'. Supported Full decoders: "
            f"{', '.join(VAE_MODEL_OPTIONS)}"
        )

    model_download(model_name="JunhaoZhuang/"+model)
    base_dir = get_flashvsr_model_base_dir()
    model_path = os.path.join(base_dir, model)
    if not os.path.exists(model_path):
        raise RuntimeError(f'Model directory does not exist!\nPlease save all weights to "{model_path}"')
    def first_existing(candidates, label):
        for filename in candidates:
            candidate = os.path.join(model_path, filename)
            if os.path.exists(candidate):
                return candidate
        raise RuntimeError(
            f"{label} does not exist in '{model_path}'. Tried: {', '.join(candidates)}"
        )

    if model == "FlashVSR-v1.1":
        ckpt_candidates = [
            "diffusion_pytorch_model_streaming_dmd_v11.safetensors",
            "diffusion_pytorch_model_streaming_dmd.safetensors",
        ]
        lq_candidates = ["LQ_proj_in_v11.ckpt", "LQ_proj_in.ckpt"]
    else:
        ckpt_candidates = ["diffusion_pytorch_model_streaming_dmd.safetensors"]
        lq_candidates = ["LQ_proj_in.ckpt"]

    ckpt_path = first_existing(ckpt_candidates, "DiT checkpoint")
    lq_path = first_existing(lq_candidates, "LQ projection checkpoint")
    tcd_path = None
    if mode != "full":
        tcd_path = first_existing(["TCDecoder.ckpt"], "TCDecoder checkpoint")

    vae_config = None
    vae_path = None
    if mode == "full":
        vae_config = VAE_MODEL_MAP[vae_model]
        vae_path = download_vae_if_missing(vae_config["file"], model_path, vae_config)
        log(
            f"Full decoder: '{vae_model}' -> '{vae_config['file']}' -> "
            f"{vae_config['class'].__name__}",
            message_type='info', icon="🔍",
        )
    else:
        log(
            f"Mode '{mode}' uses TCDecoder; VAE selection '{vae_model}' is not loaded.",
            message_type='info', icon="📦",
        )

    log(f"DiT checkpoint: {os.path.basename(ckpt_path)}", message_type='info', icon="📁")
    log(f"LQ projection: {os.path.basename(lq_path)}", message_type='info', icon="📁")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "posi_prompt.pth")

    def load_full_vae(config, checkpoint_path):
        vae_class = config["class"]
        init_kwargs = {
            "z_dim": config["z_dim"],
            "dim": config["dim"],
        }
        if vae_class is LightX2VVAE:
            init_kwargs["use_full_arch"] = config["use_full_arch"]

        vae = vae_class(**init_kwargs).eval().requires_grad_(False)
        raw_state = load_state_dict(checkpoint_path)
        target_state = vae.state_dict()
        vae_state = normalize_checkpoint_state_dict(raw_state, target_state)
        missing = sorted(set(target_state) - set(vae_state))
        unexpected = sorted(set(vae_state) - set(target_state))
        mismatched = [
            (key, tuple(vae_state[key].shape), tuple(target_state[key].shape))
            for key in sorted(set(target_state).intersection(vae_state))
            if tuple(vae_state[key].shape) != tuple(target_state[key].shape)
        ]
        if missing or unexpected or mismatched:
            details = []
            if missing:
                details.append(f"missing={missing[:8]}" + ("..." if len(missing) > 8 else ""))
            if unexpected:
                details.append(f"unexpected={unexpected[:8]}" + ("..." if len(unexpected) > 8 else ""))
            if mismatched:
                details.append(f"shape_mismatch={mismatched[:4]}" + ("..." if len(mismatched) > 4 else ""))
            raise RuntimeError(
                f"VAE '{vae_model}' does not match {os.path.basename(checkpoint_path)}: "
                + "; ".join(details)
            )
        vae.load_state_dict(vae_state, strict=True, assign=True)
        return vae.to(device="cpu", dtype=dtype)

    def load_tcdecoder(checkpoint_path):
        decoder = build_tcdecoder(
            new_channels=[512, 256, 128, 128],
            device=device,
            dtype=dtype,
            new_latent_channels=16 + 768,
        )
        state = load_state_dict(checkpoint_path)
        target_state = decoder.state_dict()
        state = normalize_checkpoint_state_dict(state, target_state)
        mismatched = [
            (key, tuple(state[key].shape), tuple(target_state[key].shape))
            for key in sorted(set(target_state).intersection(state))
            if tuple(state[key].shape) != tuple(target_state[key].shape)
        ]
        if mismatched:
            raise RuntimeError(
                f"TCDecoder checkpoint has incompatible tensor shapes: {mismatched[:4]}"
                + ("..." if len(mismatched) > 4 else "")
            )

        load_result = decoder.load_state_dict(state, strict=False)
        if load_result.missing_keys:
            missing = list(load_result.missing_keys)
            raise RuntimeError(
                f"TCDecoder checkpoint is missing required weights: {missing[:8]}"
                + ("..." if len(missing) > 8 else "")
            )
        if load_result.unexpected_keys:
            log(
                f"TCDecoder ignored {len(load_result.unexpected_keys)} unused checkpoint keys.",
                message_type='warning', icon="⚠️",
            )
        decoder.eval().requires_grad_(False)
        decoder.clean_mem()
        return decoder

    mm = ModelManager(torch_dtype=dtype, device="cpu")
    mm.load_models([ckpt_path])
    if mode == "full":
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
        pipe.vae = load_full_vae(vae_config, vae_path)
        pipe.set_decoder_mode("vae")
        log(f"Loaded VAE weights from: {vae_path}", message_type='info', icon="✅")
        log(f"VAE Type Active: {type(pipe.vae).__name__}", message_type='info', icon="📦")

        if hasattr(pipe.vae, "model"):
            if hasattr(pipe.vae.model, "encoder"):
                pipe.vae.model.encoder = None
            if hasattr(pipe.vae.model, "conv1"):
                pipe.vae.model.conv1 = None
    else:
        if mode == "tiny":
            pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        elif mode == "tiny-long":
            pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        else:
            raise ValueError("mode must be one of: tiny, tiny-long, full")
        pipe.model_names = ["dit"]
        pipe.TCDecoder = load_tcdecoder(tcd_path)
        log(f"Loaded TCDecoder weights from: {tcd_path}", message_type='info', icon="✅")

    if pipe.denoising_model() is None:
        raise RuntimeError(f"Unable to load DiT checkpoint: {ckpt_path}")

    effective_attention_mode = pipe.denoising_model().set_attention_mode(attention_mode)
    pipe.attention_mode = effective_attention_mode
    requested_attention_mode = getattr(
        pipe.denoising_model(), "requested_attention_mode", attention_mode
    )
    if requested_attention_mode == "auto":
        log(
            f"Auto attention selected '{effective_attention_mode}'.",
            message_type='info', icon="🧩",
        )
    elif effective_attention_mode != requested_attention_mode:
        log(
            f"Attention backend '{requested_attention_mode}' is unavailable; using '{effective_attention_mode}'.",
            message_type='warning', icon="⚠️")
    backend_status = attention_backend_status()
    log(
        "Attention capabilities: "
        f"arch={backend_status['cuda_arch']}, "
        f"Sage={backend_status['sage_attention']['version'] or 'off'}, "
        f"FA2={'on' if backend_status['flash_attention_2']['available'] else 'off'}, "
        f"FA3={'on' if backend_status['flash_attention_3']['available'] else 'off'}, "
        f"BlockSparse={'on' if backend_status['block_sparse_attention']['available'] else 'off'}",
        message_type='info', icon="🧩")
    
    if model == "FlashVSR":
        pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    else:
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu", weights_only=False), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit", "vae"] if mode == "full" else ["dit"])
    pipe.offload_model()

    # Log final pipeline info with VAE confirmation
    vae_info = f"VAE Model: {vae_model}" if mode == "full" else "Decoder: TCDecoder"
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        vae_info += f" ({type(pipe.vae).__name__})"
    
    log(f"Pipeline Initialized: Mode={mode}, Device={device}, Dtype={dtype}, Attention={effective_attention_mode}", message_type='info', icon="🔧")
    log(f"Model: {model}, {vae_info}", message_type='info', icon="📦")

    return pipe

class cqdm:
    def __init__(self, iterable=None, total=None, desc="Processing", enable_debug=False):
        self.desc = desc
        self.pbar = None
        self.iterable = None
        self.total = total
        self.enable_debug = enable_debug
        self.start_time = time.time()
        self.step_idx = 0
        
        if iterable is not None:
            try:
                self.total = len(iterable)
                self.iterable = iter(iterable)
            except TypeError:
                if self.total is None:
                    raise ValueError("Total must be provided for iterables with no length.")

        elif self.total is not None:
            pass
            
        else:
            raise ValueError("Either iterable or total must be provided.")
            
    def __iter__(self):
        if self.iterable is None:
            raise TypeError(f"'{type(self).__name__}' object is not iterable. Did you mean to use it with a 'with' statement?")
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self
    
    def __next__(self):
        if self.iterable is None:
            raise TypeError("Cannot call __next__ on a non-iterable cqdm object.")
        try:
            step_start = time.time()
            val = next(self.iterable)
            
            if self.pbar:
                self.pbar.update(1)
            
            self.step_idx += 1

            # Show a text progress bar in the log (single line using \r)
            perc = (self.step_idx / self.total) * 100
            bar_len = 20
            filled = int(bar_len * self.step_idx // self.total)
            bar = '█' * filled + '░' * (bar_len - filled)

            elapsed = time.time() - self.start_time
            rate = self.step_idx / elapsed if elapsed > 0 else 0

            msg = f"{self.desc}: {self.step_idx}/{self.total} |{bar}| {perc:.1f}%"

            if self.enable_debug:
                step_end = time.time()
                step_time = step_end - step_start
                msg += f" (Step: {step_time:.2f}s)"
                # Pass in_place=True to log_resource_usage to keep it on one line if possible
                # But note log_resource_usage prints Resource usage which is long.
                log_resource_usage(prefix=msg, in_place=True)
            else:
                print(f"\r{msg}", end="", flush=True)
                if self.step_idx == self.total:
                    print()

            return val
        except StopIteration:
            total_time = time.time() - self.start_time
            if self.enable_debug:
                # Use print with newline here to finalize the log block
                print(f"\n✅ Loop '{self.desc}' finished in {total_time:.2f}s", flush=True)
            raise
            
    def __enter__(self):
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self.pbar
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def __len__(self):
        return self.total

def process_chunk(pipe, frames, scale, color_fix, fix_method, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, force_offload, enable_debug, is_single_frame_input=False):
    """
    Processes a single chunk of frames.
    
    =============================================================================
    FIX 3: Black Border Fix - Proper cropping to remove padding
    =============================================================================
    """
    # Aggressive garbage collection before processing (FIX 5)
    clean_vram()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    _frames = frames
    _device = pipe.device
    dtype = pipe.torch_dtype
    denoising_model = pipe.denoising_model()
    requested_attention_mode = getattr(
        denoising_model,
        "requested_attention_mode",
        getattr(pipe, "attention_mode", None),
    )
    
    # Store original dimensions for cropping (FIX 3)
    original_H, original_W = frames.shape[1], frames.shape[2]
    target_H, target_W = original_H * scale, original_W * scale
    
    # Padding logic for the chunk (temporal padding)
    add = next_8n5(frames.shape[0]) - frames.shape[0]
    padding_frames = frames[-1:, :, :, :].repeat(add, 1, 1, 1)
    _frames = torch.cat([frames, padding_frames], dim=0)

    if tiled_dit:
        N, H, W, C = _frames.shape
        
        final_output_canvas = torch.zeros(
            (N, H * scale, W * scale, C), 
            dtype=torch.float32,
            device="cpu"
        )
        weight_sum_canvas = torch.zeros_like(final_output_canvas)
        tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
        
        log(f"Starting Tiled Processing: {len(tile_coords)} tiles", message_type='info', icon="🚀")
        
        # Create progress bar wrapper for tiled pipeline processing
        class cqdm_tile(cqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, enable_debug=enable_debug)
        
        for i, (x1, y1, x2, y2) in enumerate(cqdm(tile_coords, desc="Processing Tiles", enable_debug=enable_debug)):
            tile_start = time.time()
            if enable_debug:
                log(f"Processing tile {i+1}/{len(tile_coords)}: ({x1},{y1}) -> ({x2},{y2})", message_type='info', icon="🔄")
            
            input_tile = _frames[:, y1:y2, x1:x2, :]
            
            # Get tile dimensions including padding info (FIX 3)
            LQ_tile, th, tw, F, tile_sH, tile_sW, tile_pad_top, tile_pad_left = prepare_input_tensor(
                input_tile, _device, scale=scale, dtype=dtype
            )
            if not isinstance(pipe, FlashVSRTinyLongPipeline):
                LQ_tile = LQ_tile.to(_device)

            output_tile_gpu = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                progress_bar_cmd=cqdm_tile, LQ_video=LQ_tile, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                color_fix=color_fix, fix_method=fix_method,
                attention_mode=requested_attention_mode,
                unload_dit=unload_dit, force_offload=force_offload,
                enable_debug_logging=enable_debug
            )
            
            processed_tile_cpu = tensor2video(output_tile_gpu).to("cpu")
            
            # =================================================================
            # FIX 3: Crop output tile to remove padding before blending
            # =================================================================
            # Bounds checking to avoid IndexError
            max_crop_h = min(tile_pad_top + tile_sH, processed_tile_cpu.shape[1])
            max_crop_w = min(tile_pad_left + tile_sW, processed_tile_cpu.shape[2])
            actual_h = max_crop_h - tile_pad_top
            actual_w = max_crop_w - tile_pad_left
            
            if actual_h > 0 and actual_w > 0:
                processed_tile_cpu = processed_tile_cpu[:, tile_pad_top:max_crop_h, 
                                                           tile_pad_left:max_crop_w, :]
            
            if enable_debug:
                tile_end = time.time()
                tile_time = tile_end - tile_start
                log(f"Tile {i+1} completed in {tile_time:.2f}s", message_type='info', icon="⏱️")
            
            mask_nchw = create_feather_mask(
                (processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]),
                tile_overlap * scale,
                fade_left=x1 > 0,
                fade_right=x2 < W,
                fade_top=y1 > 0,
                fade_bottom=y2 < H,
            ).to("cpu")
            mask_nhwc = mask_nchw.permute(0, 2, 3, 1)
            out_x1, out_y1 = x1 * scale, y1 * scale
            
            tile_H_scaled = processed_tile_cpu.shape[1]
            tile_W_scaled = processed_tile_cpu.shape[2]
            out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
            final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
            weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
            
            del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
            clean_vram()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        final_output = final_output_canvas / weight_sum_canvas
    else:
        log("Preparing full frame processing...", message_type='info', icon="🎞️")
        if enable_debug:
            log_resource_usage(prefix="Pre-Preprocess")
        
        # Get padding info for cropping (FIX 3)
        LQ, th, tw, F, sH, sW, pad_top, pad_left = prepare_input_tensor(_frames, _device, scale=scale, dtype=dtype)
        if not isinstance(pipe, FlashVSRTinyLongPipeline):
            LQ = LQ.to(_device)
            
        log(f"Processing {frames.shape[0]} frames...", message_type='info', icon="🚀")
        
        process_start = time.time()

        class cqdm_debug(cqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, enable_debug=enable_debug)

        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
            progress_bar_cmd=cqdm_debug, LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
            color_fix=color_fix, fix_method=fix_method,
            attention_mode=requested_attention_mode,
            unload_dit=unload_dit, force_offload=force_offload,
            enable_debug_logging=enable_debug
        )

        process_end = time.time()
        
        if enable_debug:
            log(f"Inference completed in {process_end - process_start:.2f}s", message_type='info', icon="⏱️")
        final_output_tensor = tensor2video(video).to('cpu')
        
        # =====================================================================
        # FIX 3: Crop output to remove padding - use stored padding offsets
        # =====================================================================
        # The output has dimensions (N, tH, tW, C) where tH/tW are padded
        # We need to crop to actual scaled dimensions (sH, sW)
        final_output = final_output_tensor[:, pad_top:pad_top + sH, pad_left:pad_left + sW, :]
        
        if enable_debug:
            log(f"Cropped output from ({final_output_tensor.shape[1]}, {final_output_tensor.shape[2]}) "
                f"to ({final_output.shape[1]}, {final_output.shape[2]}) removing padding", 
                message_type='info', icon="✂️")

        del video, LQ
        clean_vram()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_single_frame_input and frames.shape[0] == 1:
        # Temporal padding exists only to satisfy the video model. Preserve the
        # first aligned output frame instead of median-filtering generated detail.
        return final_output[:1].to("cpu").float()

    return final_output[:frames.shape[0], :, :, :]

def flashvsr(pipe, frames, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, force_offload, enable_debug=False, chunk_size=0, resize_factor=1.0, mode="full", fix_method="wavelet"):
    """
    =============================================================================
    FIX 9 & 10: Unified Processing Pipeline with Pre-Flight Check
    =============================================================================
    
    Main FlashVSR processing function.
    - FIX 4: Antialiased working-resolution resize and reference-aligned crop
    - FIX 5: VRAM Advisory Logging with 95% threshold
    - FIX 9: Pre-Flight Resource Check before processing
    - FIX 10: Unified processing logic applied across all modes
    """
    if frames.ndim != 4 or frames.shape[0] < 1:
        raise ValueError("FlashVSR requires a non-empty IMAGE batch with shape (N, H, W, C).")
    if scale not in (2, 4):
        raise ValueError(f"FlashVSR scale must be 2 or 4, got {scale}.")
    if not 0.0 < resize_factor <= 1.0:
        raise ValueError(f"resize_factor must be in (0, 1], got {resize_factor}.")

    # Aggressive garbage collection (FIX 5)
    clean_vram()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    if chunk_size > 0:
        log(
            "frame_chunk_size is ignored because independent chunks reset temporal "
            "attention and decoder state. Tiny-Long already streams internally.",
            message_type='warning', icon="⚠️",
        )
        chunk_size = 0

    if tiled_dit:
        log(
            "tiled_dit is ignored because running independent spatial DiT tiles "
            "changes sparse-attention context and introduces seams or repeated detail.",
            message_type='warning', icon="⚠️",
        )
        tiled_dit = False

    # ==========================================================================
    # FIX 9: Pre-Flight Resource Check (BEFORE loading heavy models/processing)
    # ==========================================================================
    preflight_result = log_preflight_check(
        frames.shape[2], frames.shape[1], frames.shape[0], scale, chunk_size, resize_factor, 
        tiled_vae, tiled_dit, mode=mode
    )
    
    # If pre-flight check suggests OOM, optionally apply recommended settings
    # (Currently just logs warnings - user can adjust settings manually)
    
    # ==========================================================================
    # Working-resolution resize. Any downscale is lossy, so use antialiased
    # bicubic consistently instead of treating nearest-neighbor as lossless.
    # ==========================================================================
    if resize_factor < 1.0 and resize_factor > 0:
        log(f"Resizing input by factor {resize_factor}...", message_type='info', icon="📉")
        orig_H, orig_W = frames.shape[1], frames.shape[2]
        new_H = max(1, int(orig_H * resize_factor))
        new_W = max(1, int(orig_W * resize_factor))
        
        frames_permuted = frames.permute(0, 3, 1, 2)
        frames_resized = F.interpolate(
            frames_permuted,
            size=(new_H, new_W),
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )
        log(
            "Using antialiased BICUBIC interpolation; reduced resolution is lossy.",
            message_type='info', icon="🔍",
        )
        
        frames = frames_resized.permute(0, 2, 3, 1)  # Back to NHWC
        del frames_permuted, frames_resized
        clean_vram()

    start_time = time.time()
    
    # Get current dimensions (after potential resize)
    N, H, W, C = frames.shape

    # ==========================================================================
    # FIX 5 & 10: Unified Debug Logging (same for all modes)
    # ==========================================================================
    if enable_debug:
        _device = pipe.device
        log(f"Debug Mode: Enabled", message_type='info', icon="🐞")
        log(f"Device: {_device}", message_type='info', icon="🖥️")
        log(f"Processing Mode: {mode}", message_type='info', icon="⚙️")
        if torch.cuda.is_available():
             log(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", message_type='info', icon="💾")
        log(f"Input Frames: {frames.shape}", message_type='info', icon="🎞️")
        log(f"Chunk Size: {chunk_size}", message_type='info', icon="📦")
        log(f"Tiled DiT: {tiled_dit}, Tiled VAE: {tiled_vae}", message_type='info', icon="🧩")
        log_resource_usage(prefix="Start")
    
    # VRAM Advisory (FIX 5) - Enhanced with mode
    if torch.cuda.is_available():
        log_vram_advisory(W, H, N, scale, tiled_vae, tiled_dit, mode=mode)

    # VRAM check and warning - FIX 5: Use 95% threshold (RTX 5070 Ti target)
    if torch.cuda.is_available():
        vram_free, vram_total = torch.cuda.mem_get_info()
        vram_used = vram_total - vram_free
        vram_usage_ratio = vram_used / vram_total

        # FIX 5: Only trigger OOM recovery at 95% threshold (not 90%)
        if vram_usage_ratio > VRAM_OOM_THRESHOLD:
            log(f"Warning: VRAM usage is very high ({vram_usage_ratio*100:.1f}% > {VRAM_OOM_THRESHOLD*100:.0f}%)! Enabling fallback options is recommended.", 
                message_type='warning', icon="⚠️")

    # Store input resolution for summary (FIX 8)
    input_resolution = f"{frames.shape[2]}x{frames.shape[1]}"
    
    is_single_frame_input = (frames.shape[0] == 1)
    current_tiled_vae = tiled_vae
    while True:
        try:
            final_output_tensor = process_chunk(
                pipe, frames, scale, color_fix, fix_method,
                current_tiled_vae, tiled_dit,
                tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio,
                local_range, seed, force_offload, enable_debug,
                is_single_frame_input=is_single_frame_input,
            )
            break
        except torch.OutOfMemoryError as exc:
            clean_vram()
            if mode == "full" and not current_tiled_vae:
                current_tiled_vae = True
                log(
                    "OOM detected; retrying with the Wan VAE's native tiler. "
                    "DiT spatial tiling remains disabled to preserve model context.",
                    message_type='warning', icon="🔄",
                )
                continue
            raise RuntimeError(
                "FlashVSR ran out of VRAM. Reduce resize_factor/input resolution or "
                "use Tiny-Long. Automatic DiT tiling and stateless frame chunking "
                "are disabled because they change output quality."
            ) from exc

    end_time = time.time()
    total_time = end_time - start_time
    fps = frames.shape[0] / total_time if total_time > 0 else 0
    output_resolution = f"{final_output_tensor.shape[2]}x{final_output_tensor.shape[1]}"
    
    # ==========================================================================
    # FIX 8: Summary logging at end of processing
    # ==========================================================================
    log("=" * 60, message_type='info')
    log("PROCESSING SUMMARY", message_type='finish', icon="📊")
    log(f"Total Processing Time: {total_time:.2f}s ({fps:.2f} FPS)", message_type='info', icon="⏱️")
    log(f"Input Resolution: {input_resolution} ({frames.shape[0]} frames)", message_type='info', icon="📥")
    log(f"Output Resolution: {output_resolution} ({final_output_tensor.shape[0]} frames)", message_type='info', icon="📤")
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3
        log(f"Peak VRAM Used: {peak_memory:.2f} GB", message_type='info', icon="📈")
        
    log_resource_usage(prefix="Final")
    log("=" * 60, message_type='info')
    
    return final_output_tensor


class FlashVSRNodeInitPipe:
    """
    =============================================================================
    FIX 1: Unified VAE Selection - Merged vae_type and alt_vae into vae_model
    =============================================================================
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["FlashVSR", "FlashVSR-v1.1"], {
                    "default": "FlashVSR-v1.1",
                    "tooltip": "Select the FlashVSR model version. V1.1 is recommended for better stability."
                }),
                "mode": (["tiny", "tiny-long", "full"], {
                    "default": "full",
                    "tooltip": 'Operation mode. Tiny modes use TCDecoder for speed. Full uses the selected Wan VAE and provides the highest reconstruction quality.'
                }),
                "vae_model": (VAE_MODEL_OPTIONS, {
                    "default": "Wan2.1",
                    "tooltip": 'Full decoder: Wan2.1 for fidelity or LightVAE_W2.1 for lower VRAM. Official Wan2.2 is incompatible with FlashVSR latent channels. Tiny modes use TCDecoder.'
                }),
                "force_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, forces offloading of models to CPU RAM after execution to free up VRAM for other nodes."
                }),
                "precision": (["fp16", "bf16", "auto"], {
                    "default": "auto",
                    "tooltip": "Inference precision. 'auto' selects bf16 if supported (RTX 30/40/50 series), otherwise fp16. bf16 is recommended."
                }),
                "device": (device_choices, {
                    "default": device_choices[0],
                    "tooltip": "Select the computation device (CUDA GPU, CPU, etc.). 'auto' picks the best available."
                }),
                "attention_mode": (ATTENTION_MODE_OPTIONS, {
                    "default": "auto",
                    "tooltip": 'Auto prefers a mask-capable backend. Dense Sage/FlashAttention choices accelerate compatible calls; masked self-attention uses masked SDPA when required.'
                }),
            }
        }
    
    RETURN_TYPES = ("PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    DESCRIPTION = 'Initializes FlashVSR with validated attention and decoder backends. Full mode uses the selected Wan VAE; Tiny modes use TCDecoder.'
    
    def main(self, model, mode, vae_model, force_offload, precision, device, attention_mode):
        _device = device
        if device == "auto":
            _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else device
        if _device == "auto" or _device not in device_choices:
            raise RuntimeError("No devices found to run FlashVSR!")
            
        if _device.startswith("cuda"):
            torch.cuda.set_device(_device)
            
        # Auto bfloat16 detection
        if precision == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                precision = "bf16"
                log("Auto-detected bf16 support.", message_type='info', icon="⚙️")
            else:
                precision = "fp16"
                log("Defaulting to fp16.", message_type='info', icon="⚙️")
            
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        try:
            dtype = dtype_map[precision]
        except:
            dtype = torch.bfloat16

        # Use unified vae_model parameter
        pipe = init_pipeline(
            model, mode, _device, dtype,
            vae_model=vae_model,
            attention_mode=attention_mode,
        )
        # FIX 10: Store mode with pipe for unified processing logic
        return((pipe, force_offload, mode),)

class FlashVSRNodeAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PIPE", {
                    "tooltip": "The initialized FlashVSR pipeline object from the Init node."
                }),
                "frames": ("IMAGE", {
                    "tooltip": "Input video frames to be upscaled. Batch of images (N, H, W, C)."
                }),
                "scale": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 4,
                    "tooltip": "Upscaling factor. 2x or 4x. Higher scale requires more VRAM and compute."
                }),
                "color_fix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Match decoded colors to the input using the selected correction method."
                }),
                "tiled_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use the Full Wan VAE's native tiler to reduce decode VRAM. Tiny modes ignore this option because TCDecoder spatial tiling changes output quality."
                }),
                "tiled_dit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Process the DiT in spatial tiles. This reduces VRAM but changes global attention context, so leave it disabled for best quality."
                }),
                "tile_size": ("INT", {
                    "default": 256,
                    "min": 32,
                    "max": 1024,
                    "step": 32,
                    "tooltip": "Size of the tiles for DiT processing. Smaller = less VRAM, more tiles, slower."
                }),
                "tile_overlap": ("INT", {
                    "default": 24,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Overlap pixels between tiles to blend seams. Higher overlap = smoother transitions but more computation."
                }),
                "unload_dit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the DiT model from VRAM before VAE decoding starts. Use this if VAE decode runs out of memory."
                }),
                "sparse_ratio": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Control for sparse attention. 1.5 is faster, 2.0 is more stable/quality. (For sparse backends only)"
                }),
                "kv_ratio": ("FLOAT", {
                    "default": 3.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Key/Value cache ratio. Higher values preserve more temporal context at the cost of VRAM; 3.5 matches the quality-oriented reference setting."
                }),
                "local_range": ("INT", {
                    "default": 11,
                    "min": 7,
                    "max": 11,
                    "step": 2,
                    "tooltip": "Local sparse-attention window. 11 matches the reference ComfyUI node and favors temporal stability; 9 is sharper."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624,
                    "tooltip": "Random seed for noise generation. Same seed + same settings = reproducible results."
                }),
                "frame_chunk_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Reserved for compatibility. Values above 0 are ignored because independent chunks reset temporal attention and decoder state."
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable verbose logging to console. Shows VRAM usage, step times, tile info, and detailed progress."
                }),
                "keep_models_on_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Move models to CPU RAM instead of keeping them in VRAM when not in use. Prevents VRAM fragmentation/OOM."
                }),
                "resize_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Resize input frames before processing. Set to 0.5x for large 1080p+ videos to save VRAM."
                }),
                "fix_method": (FIX_METHOD_OPTIONS, {
                    "default": "wavelet",
                    "tooltip": "Color correction method. Wavelet better preserves generated high-frequency detail; AdaIN is a simpler statistical match."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    
    def main(self, pipe, frames, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, frame_chunk_size, enable_debug, keep_models_on_cpu, resize_factor, fix_method="wavelet"):
        # FIX 10: Extract mode from pipe tuple for unified processing
        # Pipe tuple structure: (pipeline_object, force_offload, mode)
        # Backwards compatible with older 2-element tuples (pipeline, force_offload)
        if len(pipe) >= 3:
            _pipe, init_force_offload, mode = pipe
        else:
            _pipe = pipe[0]
            init_force_offload = bool(pipe[1]) if len(pipe) > 1 else False
            mode = "full"  # Default fallback for backwards compatibility
        effective_force_offload = bool(init_force_offload or keep_models_on_cpu)
        output = flashvsr(
            _pipe, frames, scale, color_fix, tiled_vae, tiled_dit,
            tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio,
            local_range, seed, effective_force_offload, enable_debug,
            frame_chunk_size, resize_factor, mode=mode, fix_method=fix_method,
        )
        return(output.cpu().float(),)

class FlashVSRNode:
    """
    =============================================================================
    FIX 1: Unified VAE Selection - Single vae_model dropdown
    =============================================================================
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Input video frames to be upscaled. Batch of images (N, H, W, C)."
                }),
                "model": (["FlashVSR", "FlashVSR-v1.1"], {
                    "default": "FlashVSR-v1.1",
                    "tooltip": "Select the FlashVSR model version. V1.1 is recommended for better stability."
                }),
                "mode": (["tiny", "tiny-long", "full"], {
                    "default": "full",
                    "tooltip": 'Operation mode. Tiny modes use TCDecoder for speed. Full uses the selected Wan VAE and provides the highest reconstruction quality.'
                }),
                "vae_model": (VAE_MODEL_OPTIONS, {
                    "default": "Wan2.1",
                    "tooltip": 'Full decoder: Wan2.1 for fidelity or LightVAE_W2.1 for lower VRAM. Official Wan2.2 is incompatible with FlashVSR latent channels. Tiny modes use TCDecoder.'
                }),
                "scale": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 4,
                    "tooltip": "Upscaling factor. 2x or 4x."
                }),
                "tiled_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use the Full Wan VAE's native tiler to reduce decode VRAM. Tiny modes ignore this option."
                }),
                "tiled_dit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Process the DiT in spatial tiles. Leave disabled for best quality because tiling changes global attention context."
                }),
                "unload_dit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload the DiT model from VRAM before VAE decoding starts to free up memory. Recommended for 16GB VRAM."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624,
                    "tooltip": "Random seed for noise generation."
                }),
                "frame_chunk_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Reserved for compatibility. Values above 0 are ignored to preserve temporal attention and decoder state."
                }),
                "attention_mode": (ATTENTION_MODE_OPTIONS, {
                    "default": "auto",
                    "tooltip": 'Auto prefers a mask-capable backend. Dense Sage/FlashAttention choices accelerate compatible calls; masked self-attention uses masked SDPA when required.'
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable extensive logging for debugging."
                }),
                "keep_models_on_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Move models to CPU RAM instead of keeping them in VRAM when not in use."
                }),
                "resize_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Resize input frames before processing. Set to 0.5x for large 1080p+ videos to save VRAM."
                }),
                "color_fix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Match output colors to the input after decoding."
                }),
                "fix_method": (FIX_METHOD_OPTIONS, {
                    "default": "wavelet",
                    "tooltip": "Color correction method. Wavelet better preserves generated high-frequency detail."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    DESCRIPTION = 'Single-node FlashVSR upscaling with quality-oriented BF16, attention, and Wavelet defaults.'
    
    def main(self, model, frames, mode, vae_model, scale, tiled_vae, tiled_dit, unload_dit, seed, frame_chunk_size, attention_mode, enable_debug, keep_models_on_cpu, resize_factor, color_fix=True, fix_method="wavelet"):
        _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "auto"
        if _device == "auto" or _device not in device_choices:
            raise RuntimeError("No devices found to run FlashVSR!")
            
        if _device.startswith("cuda"):
            torch.cuda.set_device(_device)
            
        dtype = (
            torch.bfloat16
            if _device.startswith("cuda") and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        # Use unified vae_model parameter
        pipe = init_pipeline(
            model, mode, _device, dtype,
            vae_model=vae_model,
            attention_mode=attention_mode,
        )
        # FIX 10: Pass mode for unified processing logic
        output = flashvsr(
            pipe, frames, scale, color_fix, tiled_vae, tiled_dit,
            256, 24, unload_dit, 2.0, 3.5, 11, seed,
            keep_models_on_cpu, enable_debug, frame_chunk_size, resize_factor,
            mode=mode, fix_method=fix_method,
        )
        return(output.cpu().float(),)

NODE_CLASS_MAPPINGS = {
    "FlashVSRNode": FlashVSRNode,
    "FlashVSRNodeAdv": FlashVSRNodeAdv,
    "FlashVSRInitPipe": FlashVSRNodeInitPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSRNode": "FlashVSR Ultra-Fast",
    "FlashVSRNodeAdv": "FlashVSR Ultra-Fast (Advanced)",
    "FlashVSRInitPipe": "FlashVSR Init Pipeline",
}
