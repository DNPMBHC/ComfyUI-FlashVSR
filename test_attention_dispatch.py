import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from src.models import wan_video_dit


class TestAttentionDispatch(unittest.TestCase):
    def setUp(self):
        self.q = torch.randn(1, 4, 8)
        self.k = torch.randn(1, 4, 8)
        self.v = torch.randn(1, 4, 8)
        self.mask = torch.ones(1, 2, 1, 1, dtype=torch.bool)

    def test_sparse_sage_selection_is_not_overridden_by_flash_attention_2(self):
        sparse_result = torch.randn(1, 2, 4, 4)
        sparse = MagicMock(return_value=sparse_result)
        flash_func = MagicMock(side_effect=AssertionError("FlashAttention 2 must not run"))

        with patch.object(wan_video_dit, "SPARSE_SAGE_ATTN_AVAILABLE", True), \
             patch.object(wan_video_dit, "FLASH_ATTN_2_AVAILABLE", True), \
             patch.object(wan_video_dit, "_cuda_architecture", return_value="sm89"), \
             patch.object(wan_video_dit, "sparse_sageattn", sparse), \
             patch.object(wan_video_dit, "flash_attn", SimpleNamespace(flash_attn_func=flash_func)):
            output = wan_video_dit.flash_attention(
                self.q, self.k, self.v,
                num_heads=2,
                attention_mask=self.mask,
                attention_mode="sparse_sage_attention",
            )

        self.assertEqual(output.shape, self.q.shape)
        sparse.assert_called_once()
        flash_func.assert_not_called()

    def test_flash_attention_2_runs_only_when_selected(self):
        flash_result = torch.randn(1, 4, 2, 4)
        flash_func = MagicMock(return_value=flash_result)
        sparse = MagicMock(side_effect=AssertionError("Sparse Sage must not run"))

        with patch.object(wan_video_dit, "FLASH_ATTN_2_AVAILABLE", True), \
             patch.object(wan_video_dit, "_cuda_architecture", return_value="sm120"), \
             patch.object(wan_video_dit, "flash_attn", SimpleNamespace(flash_attn_func=flash_func)), \
             patch.object(wan_video_dit, "sparse_sageattn", sparse):
            output = wan_video_dit.flash_attention(
                self.q, self.k, self.v,
                num_heads=2,
                attention_mask=self.mask,
                attention_mode="flash_attention_2",
            )

        self.assertEqual(output.shape, self.q.shape)
        flash_func.assert_called_once()
        sparse.assert_not_called()

    def test_flash_attention_3_runs_only_when_selected(self):
        flash_result = torch.randn(1, 4, 2, 4)
        flash_func = MagicMock(return_value=flash_result)

        with patch.object(wan_video_dit, "FLASH_ATTN_3_AVAILABLE", True), \
             patch.object(wan_video_dit, "_cuda_architecture", return_value="sm90"), \
             patch.object(wan_video_dit, "_flash3_architecture_supported", return_value=True), \
             patch.object(wan_video_dit, "flash_attn_interface", SimpleNamespace(flash_attn_func=flash_func)):
            output = wan_video_dit.flash_attention(
                self.q, self.k, self.v,
                num_heads=2,
                attention_mode="flash_attention_3",
            )

        self.assertEqual(output.shape, self.q.shape)
        flash_func.assert_called_once()

    def test_sage_attention_uses_the_v22_backend(self):
        sage_result = torch.randn(1, 2, 4, 4)
        sage_func = MagicMock(return_value=sage_result)

        with patch.object(wan_video_dit, "SAGE_ATTN_AVAILABLE", True), \
             patch.object(wan_video_dit, "_cuda_architecture", return_value="sm120"), \
             patch.object(wan_video_dit, "sageattn", sage_func):
            output = wan_video_dit.flash_attention(
                self.q, self.k, self.v,
                num_heads=2,
                attention_mode="sage_attention",
            )

        self.assertEqual(output.shape, self.q.shape)
        sage_func.assert_called_once()

    def test_sdpa_ignores_installed_optional_backends(self):
        flash_func = MagicMock(side_effect=AssertionError("FlashAttention 2 must not run"))
        sparse = MagicMock(side_effect=AssertionError("Sparse Sage must not run"))

        with patch.object(wan_video_dit, "FLASH_ATTN_2_AVAILABLE", True), \
             patch.object(wan_video_dit, "_cuda_architecture", return_value="sm120"), \
             patch.object(wan_video_dit, "flash_attn", SimpleNamespace(flash_attn_func=flash_func)), \
             patch.object(wan_video_dit, "sparse_sageattn", sparse):
            output = wan_video_dit.flash_attention(
                self.q, self.k, self.v,
                num_heads=2,
                attention_mask=self.mask,
                attention_mode="sdpa",
            )

        self.assertEqual(output.shape, self.q.shape)
        flash_func.assert_not_called()
        sparse.assert_not_called()

    def test_unavailable_selected_backend_falls_back_to_sdpa(self):
        with patch.object(wan_video_dit, "FLASH_ATTN_2_AVAILABLE", False), \
             patch.object(wan_video_dit, "_cuda_architecture", return_value=None):
            output = wan_video_dit.flash_attention(
                self.q, self.k, self.v,
                num_heads=2,
                attention_mode="flash_attention_2",
            )

        self.assertEqual(output.shape, self.q.shape)

    def test_runtime_failure_of_selected_backend_falls_back_to_sdpa(self):
        flash_func = MagicMock(side_effect=RuntimeError("unsupported CUDA kernel"))

        with patch.object(wan_video_dit, "FLASH_ATTN_2_AVAILABLE", True), \
             patch.object(wan_video_dit, "_cuda_architecture", return_value="sm120"), \
             patch.object(wan_video_dit, "flash_attn", SimpleNamespace(flash_attn_func=flash_func)):
            output = wan_video_dit.flash_attention(
                self.q, self.k, self.v,
                num_heads=2,
                attention_mode="flash_attention_2",
            )

        self.assertEqual(output.shape, self.q.shape)
        flash_func.assert_called_once()


if __name__ == "__main__":
    unittest.main()
