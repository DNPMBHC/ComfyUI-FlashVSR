
import sys
import os
import inspect
import unittest
from unittest.mock import MagicMock, patch

# Mock ComfyUI modules
sys.modules['folder_paths'] = MagicMock()
sys.modules['folder_paths'].get_filename_list = MagicMock(return_value=[])
sys.modules['folder_paths'].models_dir = "/tmp/models"
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.utils'].ProgressBar = MagicMock()

# Mock dependencies that might be missing or require heavy setup
sys.modules['sageattention'] = MagicMock()
sys.modules['flash_attn'] = MagicMock()

import torch
# We need to ensure torch.cuda.is_available is mocked if no GPU
if not torch.cuda.is_available():
    torch.cuda.is_available = MagicMock(return_value=False)

from nodes import (
    flashvsr,
    prepare_input_tensor,
    FlashVSRNodeInitPipe,
    FlashVSRNode,
    FlashVSRNodeAdv,
    VAE_MODEL_OPTIONS,
    VAE_MODEL_MAP,
    FIX_METHOD_OPTIONS,
)
from nodes import estimate_vram_usage, get_optimal_settings, check_resources
from src.pipelines.flashvsr_full import FlashVSRFullPipeline
from src.pipelines.flashvsr_tiny import FlashVSRTinyPipeline
from src.pipelines.flashvsr_tiny_long import FlashVSRTinyLongPipeline
from src.models.wan_video_vae import WanVideoVAE, LightX2VVAE, create_video_vae
from src.models.utils import normalize_checkpoint_state_dict

class TestFlashVSRNodes(unittest.TestCase):
    def test_pipeline_instantiation(self):
        # We can't easily instantiate the full pipeline without models,
        # but we can check if the class loads and methods exist.
        self.assertTrue(hasattr(FlashVSRFullPipeline, '__call__'))

    def test_nodes_import(self):
        self.assertTrue(hasattr(FlashVSRNode, 'INPUT_TYPES'))
        self.assertTrue(hasattr(FlashVSRNodeAdv, 'INPUT_TYPES'))
        self.assertTrue(hasattr(FlashVSRNodeInitPipe, 'INPUT_TYPES'))

    def test_vae_model_options_available(self):
        """The UI exposes only VAE checkpoints compatible with 16-channel FlashVSR latents."""
        self.assertEqual(VAE_MODEL_OPTIONS, ["Wan2.1", "LightVAE_W2.1"])
        self.assertNotIn("Wan2.2", VAE_MODEL_OPTIONS)
    
    def test_vae_model_map_configured(self):
        """VAE_MODEL_MAP matches the selectable, validated implementations."""
        self.assertEqual(set(VAE_MODEL_MAP), set(VAE_MODEL_OPTIONS))
        
        # Test each entry has required keys (updated for new schema)
        for key, value in VAE_MODEL_MAP.items():
            self.assertIn("class", value)
            self.assertIn("file", value)
            self.assertIn("internal_name", value)
            self.assertIn("url", value)  # New: Direct URL for auto-download
            self.assertIn("dim", value)  # New: VAE dimension
            self.assertIn("z_dim", value)  # New: z dimension
        
        # CRITICAL: Verify DISTINCT file paths (no reuse)
        files = [VAE_MODEL_MAP[k]["file"] for k in VAE_MODEL_MAP]
        self.assertEqual(len(files), len(set(files)), "VAE files must be DISTINCT - no reuse!")
        
        # Verify specific file and class mappings
        self.assertEqual(VAE_MODEL_MAP["Wan2.1"]["file"], "Wan2.1_VAE.pth")
        self.assertEqual(VAE_MODEL_MAP["LightVAE_W2.1"]["file"], "lightvaew2_1.pth")
        self.assertIs(VAE_MODEL_MAP["Wan2.1"]["class"], WanVideoVAE)
        self.assertIs(VAE_MODEL_MAP["LightVAE_W2.1"]["class"], LightX2VVAE)

    def test_checkpoint_state_dict_normalization(self):
        target = {
            "model.layer.weight": torch.empty(2, 3),
            "model.layer.bias": torch.empty(2),
        }
        raw = {
            "state_dict": {
                "module.layer.weight": torch.ones(2, 3),
                "module.layer.bias": torch.ones(2),
            },
            "epoch": 12,
        }
        normalized = normalize_checkpoint_state_dict(raw, target)
        self.assertEqual(set(normalized), set(target))
        self.assertEqual(tuple(normalized["model.layer.weight"].shape), (2, 3))

    def test_vae_model_in_node_input_types(self):
        """Test that vae_model parameter is present in node INPUT_TYPES."""
        init_types = FlashVSRNodeInitPipe.INPUT_TYPES()
        self.assertIn('vae_model', init_types['required'])
        # Ensure old parameters are removed
        self.assertNotIn('vae_type', init_types['required'])
        self.assertNotIn('alt_vae', init_types['required'])
        
        node_types = FlashVSRNode.INPUT_TYPES()
        self.assertIn('vae_model', node_types['required'])
        self.assertNotIn('vae_type', node_types['required'])

    def test_quality_defaults_and_decoder_contract(self):
        """Keep quality-sensitive defaults and pipeline contracts from regressing."""
        self.assertEqual(FIX_METHOD_OPTIONS, ["wavelet", "adain"])

        tiny_signature = inspect.signature(FlashVSRTinyPipeline.__call__)
        self.assertEqual(tiny_signature.parameters["fix_method"].default, "wavelet")
        self.assertEqual(tiny_signature.parameters["color_fix_chunk_size"].default, 16)

        full_signature = inspect.signature(FlashVSRFullPipeline.__call__)
        self.assertIsNone(full_signature.parameters["decoder_mode"].default)
        self.assertEqual(full_signature.parameters["fix_method"].default, "wavelet")
        self.assertTrue(hasattr(FlashVSRFullPipeline, "set_decoder_mode"))
        full_pipe = FlashVSRFullPipeline(device="cpu")
        self.assertEqual(full_pipe.decoder_mode, "vae")
        self.assertEqual(full_pipe.set_decoder_mode("tcdecoder"), "tcd")

        init_required = FlashVSRNodeInitPipe.INPUT_TYPES()["required"]
        self.assertEqual(init_required["vae_model"][0], VAE_MODEL_OPTIONS)
        self.assertEqual(init_required["vae_model"][1]["default"], "Wan2.1")
        self.assertEqual(init_required["precision"][1]["default"], "auto")
        self.assertEqual(init_required["attention_mode"][1]["default"], "auto")

        node_required = FlashVSRNode.INPUT_TYPES()["required"]
        self.assertEqual(node_required["vae_model"][0], VAE_MODEL_OPTIONS)
        self.assertEqual(node_required["mode"][1]["default"], "full")
        self.assertEqual(node_required["frame_chunk_size"][1]["default"], 0)
        self.assertEqual(node_required["resize_factor"][1]["default"], 1.0)

        advanced_required = FlashVSRNodeAdv.INPUT_TYPES()["required"]
        self.assertEqual(advanced_required["local_range"][1]["default"], 11)

    def test_prepare_input_tensor_executes_reference_crop_path(self):
        """Exercise the real preprocessing call so keyword and range regressions fail."""
        frames = torch.rand(5, 70, 65, 3)
        prepared, height, width, frame_count, out_h, out_w, top, left = prepare_input_tensor(
            frames,
            device="cpu",
            scale=2,
            dtype=torch.float32,
        )

        self.assertEqual(prepared.shape, (1, 3, 9, 128, 128))
        self.assertEqual((height, width, frame_count), (128, 128, 9))
        self.assertEqual((out_h, out_w, top, left), (128, 128, 0, 0))
        self.assertGreaterEqual(float(prepared.min()), -1.0)
        self.assertLessEqual(float(prepared.max()), 1.0)

    def test_tiled_dit_is_ignored_to_preserve_full_frame_context(self):
        pipe = MagicMock(device="cpu", torch_dtype=torch.float32)
        frames = torch.rand(1, 32, 32, 3)
        expected = torch.rand(1, 64, 64, 3)

        with patch("nodes.clean_vram"), \
             patch("nodes.log_preflight_check", return_value={}), \
             patch("nodes.log_resource_usage"), \
             patch("nodes.process_chunk", return_value=expected) as process:
            output = flashvsr(
                pipe, frames, 2, True, False, True, 256, 24,
                False, 2.0, 3.5, 11, 0, False,
            )

        self.assertIs(output, expected)
        self.assertFalse(process.call_args.args[6])

    def test_advanced_node_honors_init_force_offload(self):
        pipeline = MagicMock()
        frames = torch.rand(1, 16, 16, 3)
        expected = torch.rand(1, 32, 32, 3)

        with patch("nodes.flashvsr", return_value=expected) as run:
            output, = FlashVSRNodeAdv().main(
                (pipeline, True, "full"), frames, 2, True, False, False,
                256, 24, False, 2.0, 3.5, 11, 0, 0, False, False, 1.0,
            )

        self.assertTrue(run.call_args.args[13])
        self.assertIs(output, expected)

    # ==========================================================================
    # FIX 9: Tests for Pre-Flight Resource Calculator
    # ==========================================================================
    def test_estimate_vram_usage_modes(self):
        """Test estimate_vram_usage returns different values for different modes."""
        vram_full = estimate_vram_usage(1280, 720, 100, 2, mode='full')
        vram_tiny = estimate_vram_usage(1280, 720, 100, 2, mode='tiny')
        vram_tiny_long = estimate_vram_usage(1280, 720, 100, 2, mode='tiny-long')
        
        # All should return positive values
        self.assertGreater(vram_full, 0)
        self.assertGreater(vram_tiny, 0)
        self.assertGreater(vram_tiny_long, 0)
        
        # Full mode should use most VRAM
        self.assertGreater(vram_full, vram_tiny_long)
    
    def test_estimate_vram_usage_tiling_reduces(self):
        """Test that tiling reduces estimated VRAM."""
        vram_no_tile = estimate_vram_usage(1280, 720, 100, 2, tiled_vae=False, tiled_dit=False)
        vram_tiled = estimate_vram_usage(1280, 720, 100, 2, tiled_vae=True, tiled_dit=True)
        
        self.assertLess(vram_tiled, vram_no_tile)
    
    def test_estimate_vram_usage_chunking_reduces(self):
        """Test that chunking reduces estimated VRAM."""
        vram_no_chunk = estimate_vram_usage(1280, 720, 100, 2, chunk_size=0)
        vram_chunked = estimate_vram_usage(1280, 720, 100, 2, chunk_size=25)
        
        self.assertLess(vram_chunked, vram_no_chunk)
    
    def test_get_optimal_settings_high_vram(self):
        """Test that high VRAM returns default settings."""
        settings = get_optimal_settings(640, 480, 50, 2, available_vram_gb=32.0, mode='full')
        
        # With 32GB VRAM, should be safe with defaults
        self.assertFalse(settings['tiled_vae'])
        self.assertFalse(settings['tiled_dit'])
        self.assertEqual(settings['chunk_size'], 0)
        self.assertEqual(settings['resize_factor'], 1.0)
    
    def test_get_optimal_settings_low_vram(self):
        """Low VRAM keeps quality-changing DiT/chunk fallbacks disabled."""
        settings = get_optimal_settings(1920, 1080, 200, 4, available_vram_gb=4.0, mode='full')
        
        # The native VAE tiler is allowed, but independent DiT tiles/chunks alter
        # attention and temporal state and must not be enabled automatically.
        self.assertTrue(settings['tiled_vae'])
        self.assertFalse(settings['tiled_dit'])
        self.assertEqual(settings['chunk_size'], 0)
        self.assertLess(settings['resize_factor'], 1.0)

    def test_vae_factory_function(self):
        """Test the create_video_vae factory function."""
        # Test wan2.1
        vae1 = create_video_vae('wan2.1')
        self.assertIsInstance(vae1, WanVideoVAE)
        
        # Official Wan2.2 is a 48-channel VAE and cannot decode FlashVSR latents.
        with self.assertRaisesRegex(ValueError, "48-channel"):
            create_video_vae('wan2.2')
        
        # Test lightx2v
        vae3 = create_video_vae('lightx2v', use_full_arch=True)
        self.assertIsInstance(vae3, LightX2VVAE)

    def test_lightx2v_vae_initialization(self):
        """Test LightX2VVAE initialization and basic attributes."""
        vae = LightX2VVAE(z_dim=16, dim=64, use_full_arch=True)
        self.assertEqual(vae.vae_type, "lightx2v")
        self.assertEqual(vae.upsampling_factor, 8)
        self.assertIsNotNone(vae.model)

    def test_vae_factory_invalid_type(self):
        """Test that factory raises error for invalid VAE type."""
        with self.assertRaises(ValueError):
            create_video_vae('invalid_vae_type')

    def test_full_pipeline_vram_optimization(self):
        pipe = FlashVSRFullPipeline(device="cpu")
        pipe.load_models_to_device = MagicMock()
        pipe.offload_model = MagicMock()
        pipe.decode_video = MagicMock(return_value=torch.zeros((1, 3, 21, 64, 64)))
        pipe.dit = MagicMock()
        pipe.dit.blocks = []
        pipe.vae = MagicMock()
        pipe.prompt_emb_posi = {'context': torch.zeros(1), 'stats': 'load'}
        pipe.generate_noise = MagicMock(return_value=torch.zeros((1, 16, 7, 8, 8)))
        pipe.timestep = torch.tensor([1000.0])
        pipe.t_mod = torch.zeros(1)
        pipe.t = torch.zeros(1)

        noise_pred = torch.zeros((1, 16, 6, 8, 8))
        with patch(
            "src.pipelines.flashvsr_full.model_fn_wan_video",
            return_value=(noise_pred, None, None),
        ) as model_fn, patch("src.pipelines.flashvsr_full.clean_vram"):
            output = pipe(
                prompt="test",
                num_frames=25,
                height=64,
                width=64,
                cfg_scale=1.0,
                unload_dit=True,
                force_offload=True,
                enable_debug_logging=True,
            )

        self.assertEqual(tuple(output.shape), (3, 21, 64, 64))
        pipe.load_models_to_device.assert_called_once_with(["dit"])
        pipe.dit.to.assert_called_once_with("cpu")
        pipe.offload_model.assert_called_once_with()
        model_fn.assert_called_once()
        decoded_latents = pipe.decode_video.call_args.args[0]
        self.assertEqual(tuple(decoded_latents.shape), (1, 16, 6, 8, 8))
        self.assertEqual(pipe.decode_video.call_args.kwargs["decoder_mode"], "vae")

    def test_tiny_pipeline_rejects_fewer_than_25_frames(self):
        pipe = FlashVSRTinyPipeline(device="cpu")
        with self.assertRaisesRegex(ValueError, "at least 25 padded frames"):
            pipe(num_frames=21, cfg_scale=1.0)

    def test_tiny_long_pipeline_rejects_fewer_than_25_frames(self):
        pipe = FlashVSRTinyLongPipeline(device="cpu")
        with self.assertRaisesRegex(ValueError, "at least 25 padded frames"):
            pipe(num_frames=21, cfg_scale=1.0)

    def test_full_pipeline_rejects_fewer_than_25_frames(self):
        pipe = FlashVSRFullPipeline(device="cpu")
        with self.assertRaisesRegex(ValueError, "at least 25 padded frames"):
            pipe(num_frames=21, cfg_scale=1.0)

    def test_tiny_decoders_keep_temporal_state_between_latent_segments(self):
        class StatefulDecoder:
            def __init__(self):
                self.clean_calls = 0
                self.decode_calls = 0
                self.has_temporal_state = False

            def to(self, device):
                return self

            def clean_mem(self):
                self.clean_calls += 1
                self.has_temporal_state = False

            def decode_video(self, latents, **kwargs):
                if self.decode_calls == 0:
                    self.assert_first_segment(latents)
                    frame_count = 21
                    self.has_temporal_state = True
                else:
                    if not self.has_temporal_state:
                        raise AssertionError("TCDecoder temporal state was cleared between segments")
                    if latents.shape[1] != 2:
                        raise AssertionError(f"Expected a 2-frame continuation latent, got {latents.shape[1]}")
                    frame_count = 8
                self.decode_calls += 1
                return torch.zeros((1, frame_count, 3, 64, 64))

            @staticmethod
            def assert_first_segment(latents):
                if latents.shape[1] != 6:
                    raise AssertionError(f"Expected a 6-frame initial latent, got {latents.shape[1]}")

        cases = (
            (FlashVSRTinyPipeline, "src.pipelines.flashvsr_tiny"),
            (FlashVSRTinyLongPipeline, "src.pipelines.flashvsr_tiny_long"),
        )
        for pipeline_cls, module_name in cases:
            with self.subTest(pipeline=pipeline_cls.__name__):
                pipe = pipeline_cls(device="cpu", torch_dtype=torch.float32)
                pipe.load_models_to_device = MagicMock()
                pipe.dit = MagicMock()
                pipe.dit.blocks = []
                pipe.prompt_emb_posi = {'context': torch.zeros(1), 'stats': 'load'}
                pipe.generate_noise = MagicMock(
                    return_value=torch.zeros((1, 16, 9, 8, 8))
                )
                pipe.timestep = torch.tensor([1000.0])
                pipe.t_mod = torch.zeros(1)
                pipe.t = torch.zeros(1)
                decoder = StatefulDecoder()
                pipe.TCDecoder = decoder

                def model_fn(_dit, x, **kwargs):
                    return torch.zeros_like(x), None, None

                with patch(f"{module_name}.model_fn_wan_video", side_effect=model_fn), \
                     patch(f"{module_name}.clean_vram"):
                    output = pipe(
                        prompt="test",
                        num_frames=33,
                        height=64,
                        width=64,
                        cfg_scale=1.0,
                        color_fix=False,
                        progress_bar_cmd=lambda values: values,
                    )

                self.assertEqual(tuple(output.shape), (3, 29, 64, 64))
                self.assertEqual(decoder.decode_calls, 2)
                self.assertEqual(decoder.clean_calls, 2)

if __name__ == '__main__':
    unittest.main()
