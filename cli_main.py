#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlashVSR Command-Line Interface
===============================

A mirror-grade CLI that maps 1:1 with the ComfyUI node inputs.
All parameters from FlashVSRNode, FlashVSRNodeAdv, and FlashVSRNodeInitPipe
are exposed as command-line arguments.

Usage:
    python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2

For full help:
    python cli_main.py --help
"""

import argparse
import os
import shutil
import subprocess
import sys
import gc
import math

# =============================================================================
# CLI argument parsing - EXHAUSTIVE mapping from ComfyUI node INPUT_TYPES
# =============================================================================

def parse_args():
    """
    Parse command-line arguments.
    
    Every argument corresponds directly to a parameter in the ComfyUI node
    INPUT_TYPES (FlashVSRNode, FlashVSRNodeAdv, FlashVSRNodeInitPipe).
    """
    parser = argparse.ArgumentParser(
        description="FlashVSR CLI - Video Super Resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic 2x upscale with defaults
    python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2

    # 4x upscale with VAE tiling enabled for lower VRAM
    python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 4 \\
        --tiled_vae --unload_dit

    # Long video using the stateful streaming pipeline
    python cli_main.py --input long_video.mp4 --output upscaled.mp4 \\
        --mode tiny-long

    # Low VRAM mode
    python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2 \\
        --vae_model LightVAE_W2.1 --tiled_vae --unload_dit

For more information, visit: https://github.com/DNPMBHC/ComfyUI-FlashVSR
"""
    )

    # ==========================================================================
    # Required arguments
    # ==========================================================================
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input video file path (e.g., video.mp4)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output video file path (e.g., upscaled.mp4)'
    )

    # ==========================================================================
    # FlashVSRNodeInitPipe parameters (Pipeline Initialization)
    # ==========================================================================
    parser.add_argument(
        '--model',
        type=str,
        choices=['FlashVSR', 'FlashVSR-v1.1'],
        default='FlashVSR-v1.1',
        help='FlashVSR model version. V1.1 is recommended for better stability. (default: FlashVSR-v1.1)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['tiny', 'tiny-long', 'full'],
        default='full',
        help='Operation mode. "full" uses the selected Wan VAE for best reconstruction quality; tiny modes use TCDecoder for speed. (default: full)'
    )
    parser.add_argument(
        '--vae_model',
        type=str,
        choices=['Wan2.1', 'LightVAE_W2.1'],
        default='Wan2.1',
        help='VAE decoder: Wan2.1 (highest fidelity) or LightVAE_W2.1 (lower VRAM). The official Wan2.2 VAE uses incompatible 48-channel latents. (default: Wan2.1)'
    )
    parser.add_argument(
        '--force_offload',
        action='store_true',
        default=True,
        help='Force offloading of models to CPU RAM after execution to free up VRAM. (default: True)'
    )
    parser.add_argument(
        '--no_force_offload',
        action='store_true',
        help='Disable force offloading (keeps models in VRAM).'
    )
    parser.add_argument(
        '--precision',
        type=str,
        choices=['fp16', 'bf16', 'auto'],
        default='auto',
        help="Inference precision. 'auto' selects bf16 if supported (RTX 30/40/50 series), otherwise fp16. (default: auto)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Computation device (e.g., "cuda:0", "cuda:1", "cpu", "auto"). (default: auto)'
    )
    parser.add_argument(
        '--attention_mode',
        type=str,
        choices=['auto', 'sparse_sage_attention', 'sage_attention', 'block_sparse_attention', 'flash_attention_2', 'flash_attention_3', 'sdpa'],
        default='auto',
        help='Attention backend. auto prefers a mask-capable implementation; dense backends accelerate compatible calls while masked self-attention preserves the model topology through masked SDPA. (default: auto)'
    )

    # ==========================================================================
    # FlashVSRNodeAdv parameters (Processing)
    # ==========================================================================
    parser.add_argument(
        '--scale',
        type=int,
        choices=[2, 4],
        default=2,
        help='Upscaling factor. 2x or 4x. Higher scale requires more VRAM and compute. (default: 2)'
    )
    parser.add_argument(
        '--color_fix',
        action='store_true',
        default=True,
        help='Apply wavelet-based color correction to match output colors with input. (default: True)'
    )
    parser.add_argument(
        '--no_color_fix',
        action='store_true',
        help='Disable color correction.'
    )
    parser.add_argument(
        '--fix_method',
        type=str,
        choices=['wavelet', 'adain'],
        default='wavelet',
        help='Color correction method. wavelet preserves generated high-frequency detail; adain matches per-frame statistics. (default: wavelet)'
    )
    parser.add_argument(
        '--tiled_vae',
        action='store_true',
        default=False,
        help='Enable spatial tiling for the VAE decoder. Reduces VRAM usage significantly but is slower.'
    )
    parser.add_argument(
        '--tiled_dit',
        action='store_true',
        default=False,
        help='Deprecated compatibility flag. Spatial DiT tiling is ignored because it changes attention context and reduces quality.'
    )
    parser.add_argument(
        '--tile_size',
        type=int,
        default=256,
        help='Size of the tiles for DiT processing (32-1024). Smaller = less VRAM, more tiles, slower. (default: 256)'
    )
    parser.add_argument(
        '--tile_overlap',
        type=int,
        default=24,
        help='Overlap pixels between tiles to blend seams (8-512). Higher = smoother transitions. (default: 24)'
    )
    parser.add_argument(
        '--unload_dit',
        action='store_true',
        default=False,
        help='Unload the DiT model from VRAM before VAE decoding starts. Use if VAE decode runs out of memory.'
    )
    parser.add_argument(
        '--sparse_ratio',
        type=float,
        default=2.0,
        help='Control for sparse attention (1.5-2.0). 1.5 is faster, 2.0 is more stable/quality. (default: 2.0)'
    )
    parser.add_argument(
        '--kv_ratio',
        type=float,
        default=3.5,
        help='Key/value cache ratio (1.0-10.0). Higher values retain more temporal context at increased VRAM cost. (default: 3.5)'
    )
    parser.add_argument(
        '--local_range',
        type=int,
        choices=[7, 9, 11],
        default=11,
        help='Local attention range. 11 matches the reference ComfyUI node and favors temporal stability; 9 is sharper. (default: 11)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for noise generation. Same seed + same settings = reproducible results. (default: 0)'
    )
    parser.add_argument(
        '--frame_chunk_size',
        type=int,
        default=0,
        help='Deprecated quality-unsafe option. Only 0 is accepted because external chunks reset streaming attention and decoder state. (default: 0)'
    )
    parser.add_argument(
        '--enable_debug',
        action='store_true',
        default=False,
        help='Enable verbose logging to console. Shows VRAM usage, step times, tile info, and detailed progress.'
    )
    parser.add_argument(
        '--keep_models_on_cpu',
        action='store_true',
        default=True,
        help='Move models to CPU RAM instead of keeping them in VRAM when not in use. (default: True)'
    )
    parser.add_argument(
        '--no_keep_models_on_cpu',
        action='store_true',
        help='Keep models in VRAM (faster but uses more VRAM).'
    )
    parser.add_argument(
        '--resize_factor',
        type=float,
        default=1.0,
        help='Resize input frames before processing (0.1-1.0). Set to 0.5 for large 1080p+ videos. (default: 1.0)'
    )

    # ==========================================================================
    # Video I/O parameters
    # ==========================================================================
    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='Output video FPS. If not specified, uses input video FPS.'
    )
    parser.add_argument(
        '--codec',
        type=str,
        default='libx264',
        help='Video codec for output (e.g., libx264, libx265, h264_nvenc). (default: libx264)'
    )
    parser.add_argument(
        '--crf',
        type=int,
        default=18,
        help='Constant Rate Factor for quality (0-51, lower = better quality). (default: 18)'
    )
    parser.add_argument(
        '--start_frame',
        type=int,
        default=0,
        help='Start processing from this frame index (0-indexed). (default: 0)'
    )
    parser.add_argument(
        '--end_frame',
        type=int,
        default=-1,
        help='Stop processing at this frame index (-1 = process all). (default: -1)'
    )

    # ==========================================================================
    # Model paths (optional, for custom model locations)
    # ==========================================================================
    parser.add_argument(
        '--models_dir',
        type=str,
        default=None,
        help='Custom path to FlashVSR models directory. If not set, uses ComfyUI default or ./models'
    )

    args = parser.parse_args()
    if not 1.0 <= args.kv_ratio <= 10.0:
        parser.error('--kv_ratio must be between 1.0 and 10.0')
    if not 1.5 <= args.sparse_ratio <= 2.0:
        parser.error('--sparse_ratio must be between 1.5 and 2.0')
    if not 0.1 <= args.resize_factor <= 1.0:
        parser.error('--resize_factor must be between 0.1 and 1.0')
    if not 0 <= args.crf <= 51:
        parser.error('--crf must be between 0 and 51')
    if args.start_frame < 0:
        parser.error('--start_frame must be zero or greater')
    if args.end_frame != -1 and args.end_frame <= args.start_frame:
        parser.error('--end_frame must be greater than --start_frame, or -1')
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error('--fps must be a positive finite number')
    if args.frame_chunk_size != 0:
        parser.error(
            '--frame_chunk_size is disabled because stateless external chunks reset '
            'temporal attention and decoder state; use --mode tiny-long instead'
        )
    return args


# =============================================================================
# Video I/O utilities
# =============================================================================

# =============================================================================
# Video I/O utilities (Stream based)
# =============================================================================

class VideoReader:
    """Read the selected frame range as one stateful FlashVSR sequence."""
    def __init__(self, video_path, start_frame=0, end_frame=-1, chunk_size=0):
        import cv2
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.chunk_size = chunk_size
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0
        
        # Adjust end_frame
        if self.end_frame < 0 or self.end_frame > self.total_frames:
            self.end_frame = self.total_frames
            
        if self.start_frame >= self.total_frames:
            print(f"Warning: Start frame {self.start_frame} is beyond total frames {self.total_frames}.")
            self.end_frame = self.start_frame # Nothing to process

        self.current_frame = self.start_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_frame >= self.end_frame:
            self.cap.release()
            raise StopIteration

        import torch
        import numpy as np
        import cv2

        frames = []
        frames_to_read = self.chunk_size if self.chunk_size > 0 else (self.end_frame - self.current_frame)
        
        # Ensure we don't read past end_frame
        frames_to_read = min(frames_to_read, self.end_frame - self.current_frame)
        
        if frames_to_read <= 0:
            self.cap.release()
            raise StopIteration

        for _ in range(frames_to_read):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frames.append(frame_normalized)
            self.current_frame += 1

        if not frames:
            self.cap.release()
            raise StopIteration

        # Stack frames into tensor: (N, H, W, C)
        frames_tensor = torch.from_numpy(np.stack(frames, axis=0))
        return frames_tensor

    def get_info(self):
        return self.fps, self.total_frames

class VideoWriter:
    """
    Incremental FFmpeg writer that honors the requested codec and quality.
    """
    def __init__(self, output_path, fps, width, height, codec='libx264', crf=18):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.codec = {'h264': 'libx264', 'hevc': 'libx265'}.get(codec, codec)
        self.crf = crf

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except (ImportError, RuntimeError):
                ffmpeg = None
        if ffmpeg is None:
            raise RuntimeError(
                "FFmpeg is required for quality-controlled CLI output. Install FFmpeg "
                "or imageio-ffmpeg and ensure it is available on PATH."
            )

        quality_args = ['-crf', str(crf)]
        if self.codec.endswith('_nvenc'):
            quality_args = ['-rc:v', 'vbr', '-cq:v', str(crf), '-b:v', '0']
        elif self.codec in {'libvpx-vp9', 'libaom-av1'}:
            quality_args += ['-b:v', '0']

        command = [
            ffmpeg,
            '-hide_banner',
            '-loglevel', 'error',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-video_size', f'{width}x{height}',
            '-framerate', str(fps),
            '-i', 'pipe:0',
            '-an',
            '-c:v', self.codec,
            *quality_args,
            '-pix_fmt', 'yuv420p',
            output_path,
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )

    def write(self, frames_tensor):
        import numpy as np

        try:
            import torch
        except ImportError:
            torch = None
        
        # Convert tensor to numpy
        if torch is not None and isinstance(frames_tensor, torch.Tensor):
            frames_np = frames_tensor.cpu().numpy()
        else:
            frames_np = frames_tensor
        
        # Ensure values are in [0, 1] and convert to uint8
        frames_np = np.clip(frames_np, 0.0, 1.0)
        frames_np = (frames_np * 255).astype(np.uint8)
        
        n_frames = frames_np.shape[0]
        
        for i in range(n_frames):
            try:
                frame = np.ascontiguousarray(frames_np[i])
                self.process.stdin.write(frame.tobytes())
            except BrokenPipeError as exc:
                error = self.process.stderr.read().decode('utf-8', errors='replace').strip()
                raise RuntimeError(f"FFmpeg stopped while encoding: {error}") from exc
            
    def release(self):
        if self.process is None:
            return
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        error = self.process.stderr.read().decode('utf-8', errors='replace').strip()
        return_code = self.process.wait()
        self.process = None
        if return_code != 0:
            raise RuntimeError(f"FFmpeg failed to encode '{self.output_path}': {error}")

def format_time(seconds):
    """
    Format seconds into HH:MM:SS or MM:SS
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _resolve_force_offload(args):
    """Match the Init + Advanced node offload semantics used in ComfyUI."""
    init_force_offload = args.force_offload and not args.no_force_offload
    keep_models_on_cpu = args.keep_models_on_cpu and not args.no_keep_models_on_cpu
    return bool(init_force_offload or keep_models_on_cpu)


# =============================================================================
# Main CLI entry point
# =============================================================================

def main():
    args = parse_args()

    # Safety check: ensure output file does not already exist
    if os.path.exists(args.output):
        print(f"Error: Output file '{args.output}' already exists. Aborting to prevent overwrite.", file=sys.stderr)
        sys.exit(1)
    
    # Handle boolean flag pairs
    force_offload = _resolve_force_offload(args)
    color_fix = args.color_fix and not args.no_color_fix

    print("=" * 60)
    print("FlashVSR CLI - Video Super Resolution")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}, Mode: {args.mode}")
    print(f"VAE: {args.vae_model}, Scale: {args.scale}x")
    print("=" * 60)

    # ==========================================================================
    # Setup environment and imports
    # ==========================================================================
    
    # Mock ComfyUI modules for standalone CLI operation
    from unittest.mock import MagicMock
    
    # Create mock folder_paths module
    folder_paths_mock = MagicMock()
    if args.models_dir:
        folder_paths_mock.models_dir = args.models_dir
    else:
        # Default to ./models or ComfyUI default
        folder_paths_mock.models_dir = os.path.join(os.path.dirname(__file__), "models")
    folder_paths_mock.get_filename_list = MagicMock(return_value=[])
    sys.modules['folder_paths'] = folder_paths_mock
    
    # Create mock comfy modules
    comfy_mock = MagicMock()
    comfy_utils_mock = MagicMock()
    comfy_utils_mock.ProgressBar = MagicMock()
    sys.modules['comfy'] = comfy_mock
    sys.modules['comfy.utils'] = comfy_utils_mock
    
    # Now import FlashVSR modules
    import torch
    
    # Set device
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Device: {device}")
    
    # Import FlashVSR modules after mocking
    from nodes import (
        init_pipeline, flashvsr, log,
        VAE_MODEL_OPTIONS, VAE_MODEL_MAP
    )
    
    # ==========================================================================
    # Load input video (Lazily)
    # ==========================================================================
    print("\nInitializing Video Reader...")
    reader = VideoReader(
        args.input, 
        start_frame=args.start_frame, 
        end_frame=args.end_frame,
        chunk_size=0
    )
    
    input_fps, file_total_frames = reader.get_info()
    
    # Calculate actual frames to process based on reader's resolved range
    total_frames_to_process = reader.end_frame - reader.start_frame
    if total_frames_to_process <= 0:
        reader.cap.release()
        print("Error: The selected frame range contains no frames.", file=sys.stderr)
        raise SystemExit(1)
    
    if args.end_frame > 0 or args.start_frame > 0:
        print(f"Input: {args.input} ({input_fps:.2f} FPS)")
        print(f"Processing frames: {reader.start_frame} to {reader.end_frame} (Total: {total_frames_to_process})")
    else:
        print(f"Input: {args.input} ({input_fps:.2f} FPS, {total_frames_to_process} frames)")
        
    # Use output FPS if specified, otherwise use input FPS
    output_fps = args.fps if args.fps is not None else input_fps
    
    # ==========================================================================
    # Initialize pipeline
    # ==========================================================================
    print("\nInitializing FlashVSR pipeline...")
    
    # Determine dtype
    if args.precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Auto-detected bf16 support.")
        else:
            dtype = torch.float16
            print("Defaulting to fp16.")
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    
    # Set CUDA device if using CUDA
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    
    # Initialize the pipeline
    pipe = init_pipeline(
        model=args.model,
        mode=args.mode,
        device=device,
        dtype=dtype,
        vae_model=args.vae_model,
        attention_mode=args.attention_mode,
    )
    
    # ==========================================================================
    # Process the selected frame range as one stateful FlashVSR sequence
    # ==========================================================================
    print("\nProcessing video with FlashVSR...")
    
    writer = None
    processing_error = None
    total_processed = 0
    start_time_glob = 0
    
    try:
        import time
        start_time_glob = time.time()
        
        for chunk_idx, frames in enumerate(reader):
            # Calculate progress metrics
            elapsed = time.time() - start_time_glob
            
            # Speed (fps) - avoid division by zero
            if total_processed > 0 and elapsed > 0:
                speed_fps = total_processed / elapsed
                remaining_frames = total_frames_to_process - total_processed
                eta_seconds = remaining_frames / speed_fps
            else:
                speed_fps = 0.0
                eta_seconds = 0
            
            formatted_elapsed = format_time(elapsed)
            formatted_eta = format_time(eta_seconds)
            
            # Print status for the *current state* (before processing this chunk)
            # format: Progress:   8.34% | Processed: 6464/77514 | Elapsed: 1:34:31 | ETA: 0:12:10 | Speed: 1.25 fps
            progress_pct = (total_processed / total_frames_to_process) * 100 if total_frames_to_process > 0 else 0
            print(f"Progress: {progress_pct:6.2f}% | Processed: {total_processed}/{total_frames_to_process} | "
                  f"Elapsed: {formatted_elapsed} | ETA: {formatted_eta} | Speed: {speed_fps:.2f} fps")
            
            # Process the complete selected frame range as one stateful sequence.
            output_frames = flashvsr(
                pipe=pipe,
                frames=frames,
                scale=args.scale,
                color_fix=color_fix,
                fix_method=args.fix_method,
                tiled_vae=args.tiled_vae,
                tiled_dit=args.tiled_dit,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                unload_dit=args.unload_dit,
                sparse_ratio=args.sparse_ratio,
                kv_ratio=args.kv_ratio,
                local_range=args.local_range,
                seed=args.seed,
                force_offload=force_offload,
                enable_debug=args.enable_debug,
                chunk_size=0,
                resize_factor=args.resize_factor,
                mode=args.mode,
            )
            
            # Initialize Writer on first chunk
            if writer is None:
                h, w = output_frames.shape[1], output_frames.shape[2]
                print(f"Output dimensions: {w}x{h}")
                print(f"Saving output video to: {args.output}")
                writer = VideoWriter(
                    output_path=args.output,
                    fps=output_fps,
                    width=w,
                    height=h,
                    codec=args.codec,
                    crf=args.crf
                )
            
            # Write frames
            writer.write(output_frames)
            total_processed += frames.shape[0]
            
            # Cleanup
            del frames, output_frames
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except Exception as exc:
        processing_error = exc
        print(f"\nError during processing: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if writer:
            try:
                writer.release()
            except Exception as exc:
                if processing_error is None:
                    processing_error = exc
                    print(f"\nError while finalizing output: {exc}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
                else:
                    print(f"\nAdditional error while finalizing output: {exc}", file=sys.stderr)

    if processing_error is None and total_processed == 0:
        processing_error = RuntimeError("No frames could be decoded from the selected input range.")
    
    # ==========================================================================
    # Cleanup
    # ==========================================================================
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if processing_error is not None:
        if writer is not None and os.path.exists(args.output):
            try:
                os.remove(args.output)
            except OSError as exc:
                print(
                    f"Warning: could not remove incomplete output '{args.output}': {exc}",
                    file=sys.stderr,
                )
        raise SystemExit(1) from processing_error
    
    end_time_glob = time.time()
    total_duration = end_time_glob - start_time_glob
    avg_fps = total_processed / total_duration if total_duration > 0 else 0
    
    print("\n" + "=" * 60)
    print("FlashVSR processing complete!")
    print(f"Total Frames Processed: {total_processed}/{total_frames_to_process}")
    print(f"Total Time: {format_time(total_duration)} ({avg_fps:.2f} FPS)")
    print("=" * 60)


if __name__ == "__main__":
    main()
