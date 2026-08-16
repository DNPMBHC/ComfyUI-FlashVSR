import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from src.models import wan_video_dit


class TestAttentionDispatch(unittest.TestCase):
    def setUp(self):
        wan_video_dit._ATTENTION_WARNINGS.clear()
        self.q = torch.randn(1, 4, 8)
        self.k = torch.randn(1, 4, 8)
        self.v = torch.randn(1, 4, 8)
        self.mask = torch.ones(1, 2, 1, 1, dtype=torch.bool)

    def test_blackwell_auto_prefers_block_sparse_attention(self):
        with patch.object(wan_video_dit, "BLOCK_ATTN_AVAILABLE", True), \
             patch.object(wan_video_dit, "SAGE_ATTN_AVAILABLE", True), \
             patch.object(wan_video_dit, "FLASH_ATTN_2_AVAILABLE", True), \
             patch.object(wan_video_dit, "_cuda_architecture", return_value="sm120"):
            self.assertEqual(
                wan_video_dit.resolve_attention_mode("auto"),
                "block_sparse_attention",
            )

    def test_unavailable_explicit_dense_backend_is_not_replaced_by_another_wheel(self):
        availability = {
            "sage_attention": "SAGE_ATTN_AVAILABLE",
            "flash_attention_2": "FLASH_ATTN_2_AVAILABLE",
            "flash_attention_3": "FLASH_ATTN_3_AVAILABLE",
        }
        for requested_mode, availability_name in availability.items():
            with self.subTest(requested_mode=requested_mode), \
                 patch.object(wan_video_dit, "BLOCK_ATTN_AVAILABLE", True), \
                 patch.object(wan_video_dit, "SAGE_ATTN_AVAILABLE", True), \
                 patch.object(wan_video_dit, "FLASH_ATTN_2_AVAILABLE", True), \
                 patch.object(wan_video_dit, "FLASH_ATTN_3_AVAILABLE", True), \
                 patch.object(wan_video_dit, availability_name, False), \
                 patch.object(wan_video_dit, "_cuda_architecture", return_value="sm90"), \
                 patch.object(wan_video_dit, "_flash3_architecture_supported", return_value=True):
                self.assertEqual(
                    wan_video_dit.resolve_attention_mode(requested_mode),
                    "sdpa",
                )

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
                attention_mode="flash_attention_2",
            )

        self.assertEqual(output.shape, self.q.shape)
        flash_func.assert_called_once()
        sparse.assert_not_called()

    def test_dense_backends_with_block_mask_use_masked_sdpa(self):
        mask = torch.tensor(
            [[
                [[True, False], [False, True]],
                [[False, True], [True, False]],
            ]],
            dtype=torch.bool,
        )
        q = self.q.reshape(1, 4, 2, 4).permute(0, 2, 1, 3)
        k = self.k.reshape(1, 4, 2, 4).permute(0, 2, 1, 3)
        v = self.v.reshape(1, 4, 2, 4).permute(0, 2, 1, 3)
        token_mask = mask.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=token_mask)
        expected = expected.permute(0, 2, 1, 3).reshape_as(self.q)

        dense_backends = {
            "sage_attention": "_sage_attention",
            "flash_attention_2": "_flash_attention_2",
            "flash_attention_3": "_flash_attention_3",
            "sdpa": "_sdpa_attention",
        }
        for attention_mode, backend_name in dense_backends.items():
            wan_video_dit._ATTENTION_WARNINGS.clear()
            with self.subTest(attention_mode=attention_mode), \
                 patch.object(wan_video_dit, "resolve_attention_mode", return_value=attention_mode), \
                 patch.object(
                     wan_video_dit,
                     backend_name,
                     side_effect=AssertionError("Dense backend cannot consume the block mask"),
                 ) as dense_backend:
                output = wan_video_dit.flash_attention(
                    self.q, self.k, self.v,
                    num_heads=2,
                    attention_mask=mask,
                    attention_mode=attention_mode,
                )

            torch.testing.assert_close(output, expected)
            dense_backend.assert_not_called()

    def test_self_attention_always_builds_the_reference_draft_mask(self):
        x = torch.randn(1, 128, 8)
        draft_mask = torch.ones(1, 2, 1, 1, dtype=torch.bool)

        for attention_mode in wan_video_dit.ATTENTION_MODES:
            with self.subTest(attention_mode=attention_mode):
                attention = wan_video_dit.SelfAttention(dim=8, num_heads=2)
                attention.attn.attention_mode = attention_mode
                with patch.object(wan_video_dit, "rope_apply", side_effect=lambda tensor, _freqs, _heads: tensor), \
                     patch.object(
                         wan_video_dit,
                         "generate_draft_block_mask",
                         return_value=draft_mask,
                     ) as regular_mask, \
                     patch.object(
                         wan_video_dit,
                         "generate_draft_block_mask_sage",
                         return_value=draft_mask,
                     ) as sage_mask, \
                     patch.object(attention.attn, "forward", return_value=torch.zeros_like(x)) as backend:
                    output = attention(
                        x,
                        freqs=None,
                        f=2,
                        h=8,
                        w=8,
                        topk=0,
                        kv_len=1,
                        local_range=1,
                    )

                self.assertEqual(output.shape, x.shape)
                self.assertIs(backend.call_args.args[3], draft_mask)
                if attention_mode == "sparse_sage_attention":
                    sage_mask.assert_called_once()
                    regular_mask.assert_not_called()
                else:
                    regular_mask.assert_called_once()
                    sage_mask.assert_not_called()

    def test_blackwell_auto_initialization_failure_preserves_self_attention_mask(self):
        with patch.object(wan_video_dit, "BLOCK_ATTN_AVAILABLE", True), \
             patch.object(wan_video_dit, "SAGE_ATTN_AVAILABLE", True), \
             patch.object(wan_video_dit, "FLASH_ATTN_2_AVAILABLE", False), \
             patch.object(wan_video_dit, "FLASH_ATTN_3_AVAILABLE", False), \
             patch.object(wan_video_dit, "_cuda_architecture", return_value="sm120"), \
             patch.object(
                 wan_video_dit,
                 "_initialize_attention_backend",
                 side_effect=lambda mode: mode == "sage_attention",
             ) as initialize_backend:
            effective_mode = wan_video_dit.validate_attention_mode("auto")

        self.assertEqual(effective_mode, "sage_attention")
        initialize_backend.assert_has_calls([
            call("block_sparse_attention"),
            call("sage_attention"),
        ])

        attention = wan_video_dit.SelfAttention(dim=8, num_heads=2)
        attention.attn.attention_mode = effective_mode
        x = torch.randn(1, 128, 8)
        draft_mask = torch.ones(1, 2, 1, 1, dtype=torch.bool)
        with patch.object(wan_video_dit, "SAGE_ATTN_AVAILABLE", True), \
             patch.object(wan_video_dit, "_cuda_architecture", return_value="sm120"), \
             patch.object(wan_video_dit, "rope_apply", side_effect=lambda tensor, _freqs, _heads: tensor), \
             patch.object(
                 wan_video_dit,
                 "generate_draft_block_mask",
                 return_value=draft_mask,
             ) as generate_mask, \
             patch.object(
                 wan_video_dit,
                 "generate_draft_block_mask_sage",
                 side_effect=AssertionError("Dense fallback must use the reference draft mask"),
             ), \
             patch.object(
                 wan_video_dit,
                 "_block_masked_sdpa_attention",
                 return_value=torch.zeros_like(x),
             ) as masked_sdpa, \
             patch.object(
                 wan_video_dit,
                 "_sage_attention",
                 side_effect=AssertionError("SageAttention must not consume the block mask"),
             ) as sage_backend:
            output = attention(
                x,
                freqs=None,
                f=2,
                h=8,
                w=8,
                topk=0,
                kv_len=1,
                local_range=1,
            )

        self.assertEqual(output.shape, x.shape)
        generate_mask.assert_called_once()
        masked_sdpa.assert_called_once()
        self.assertIs(masked_sdpa.call_args.args[4], draft_mask)
        sage_backend.assert_not_called()

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
        sage_func = MagicMock(side_effect=AssertionError("SageAttention must not replace FlashAttention 2"))

        with patch.object(wan_video_dit, "FLASH_ATTN_2_AVAILABLE", True), \
             patch.object(wan_video_dit, "SAGE_ATTN_AVAILABLE", True), \
             patch.object(wan_video_dit, "_cuda_architecture", return_value="sm120"), \
             patch.object(wan_video_dit, "flash_attn", SimpleNamespace(flash_attn_func=flash_func)), \
             patch.object(wan_video_dit, "sageattn", sage_func):
            output = wan_video_dit.flash_attention(
                self.q, self.k, self.v,
                num_heads=2,
                attention_mode="flash_attention_2",
            )

        self.assertEqual(output.shape, self.q.shape)
        flash_func.assert_called_once()
        sage_func.assert_not_called()

    def test_backend_oom_is_re_raised_without_disabling_backend(self):
        flash_func = MagicMock(side_effect=torch.OutOfMemoryError("out of memory"))

        with patch.object(wan_video_dit, "FLASH_ATTN_2_AVAILABLE", True), \
             patch.object(wan_video_dit, "_cuda_architecture", return_value="sm120"), \
             patch.object(
                 wan_video_dit,
                 "flash_attn",
                 SimpleNamespace(flash_attn_func=flash_func),
             ):
            with self.assertRaises(torch.OutOfMemoryError):
                wan_video_dit.flash_attention(
                    self.q, self.k, self.v,
                    num_heads=2,
                    attention_mode="flash_attention_2",
                )
            self.assertTrue(wan_video_dit.FLASH_ATTN_2_AVAILABLE)

        flash_func.assert_called_once()

    def test_block_sparse_runtime_failure_preserves_block_mask(self):
        mask = torch.tensor(
            [[
                [[True, False], [False, True]],
                [[False, True], [True, False]],
            ]],
            dtype=torch.bool,
        )
        failing_backend = MagicMock(side_effect=RuntimeError("unsupported CUDA kernel"))

        q = self.q.reshape(1, 4, 2, 4).permute(0, 2, 1, 3)
        k = self.k.reshape(1, 4, 2, 4).permute(0, 2, 1, 3)
        v = self.v.reshape(1, 4, 2, 4).permute(0, 2, 1, 3)
        token_mask = mask.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=token_mask)
        expected = expected.permute(0, 2, 1, 3).reshape_as(self.q)

        with patch.object(wan_video_dit, "BLOCK_ATTN_AVAILABLE", True), \
             patch.object(wan_video_dit, "_cuda_architecture", return_value="sm120"), \
             patch.object(wan_video_dit, "block_sparse_attn_func", failing_backend), \
             self.assertWarnsRegex(RuntimeWarning, "preserving the attention mask"):
            output = wan_video_dit.flash_attention(
                self.q, self.k, self.v,
                num_heads=2,
                attention_mask=mask,
                attention_mode="block_sparse_attention",
            )

        torch.testing.assert_close(output, expected)
        failing_backend.assert_called_once()

    def test_sparse_sage_runtime_failure_preserves_block_mask(self):
        mask = torch.tensor(
            [[
                [[True, False], [False, True]],
                [[False, True], [True, False]],
            ]],
            dtype=torch.bool,
        )
        failing_backend = MagicMock(side_effect=RuntimeError("unsupported Triton kernel"))

        q = self.q.reshape(1, 4, 2, 4).permute(0, 2, 1, 3)
        k = self.k.reshape(1, 4, 2, 4).permute(0, 2, 1, 3)
        v = self.v.reshape(1, 4, 2, 4).permute(0, 2, 1, 3)
        token_mask = mask.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=token_mask)
        expected = expected.permute(0, 2, 1, 3).reshape_as(self.q)

        with patch.object(wan_video_dit, "SPARSE_SAGE_ATTN_AVAILABLE", True), \
             patch.object(wan_video_dit, "_cuda_architecture", return_value="sm89"), \
             patch.object(wan_video_dit, "sparse_sageattn", failing_backend), \
             self.assertWarnsRegex(RuntimeWarning, "preserving the attention mask"):
            output = wan_video_dit.flash_attention(
                self.q, self.k, self.v,
                num_heads=2,
                attention_mask=mask,
                attention_mode="sparse_sage_attention",
            )

        torch.testing.assert_close(output, expected)
        failing_backend.assert_called_once()


if __name__ == "__main__":
    unittest.main()
