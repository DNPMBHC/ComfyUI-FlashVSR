import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import time
import warnings
import importlib
import importlib.util
import importlib.metadata
from typing import Tuple, Optional, List
from einops import rearrange
from .utils import hash_state_dict_keys

ATTENTION_MODES = (
    "auto",
    "sparse_sage_attention",
    "sage_attention",
    "block_sparse_attention",
    "flash_attention_2",
    "flash_attention_3",
    "sdpa",
)
ATTENTION_MODE = "auto"

_ATTENTION_MODE_ALIASES = {
    "sparse_sage": "sparse_sage_attention",
    "sage": "sage_attention",
    "sageattention": "sage_attention",
    "block_sparse": "block_sparse_attention",
    "flash_attn_2": "flash_attention_2",
    "flash_attn_3": "flash_attention_3",
}
_ATTENTION_WARNINGS = set()

def _module_exists(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


flash_attn = None
flash_attn_interface = None
block_sparse_attn_func = None
sparse_sageattn = None
sageattn = None
FLASH_ATTN_2_AVAILABLE = _module_exists("flash_attn")
FLASH_ATTN_3_AVAILABLE = _module_exists("flash_attn_interface")
BLOCK_ATTN_AVAILABLE = _module_exists("block_sparse_attn")
SAGE_ATTN_AVAILABLE = _module_exists("sageattention")
SPARSE_SAGE_ATTN_AVAILABLE = _module_exists("triton")

from PIL import Image
import numpy as np


def _warn_attention_once(message: str):
    if message not in _ATTENTION_WARNINGS:
        _ATTENTION_WARNINGS.add(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _load_flash_attention_2():
    global flash_attn, FLASH_ATTN_2_AVAILABLE
    if flash_attn is None:
        try:
            flash_attn = importlib.import_module("flash_attn")
        except Exception:
            FLASH_ATTN_2_AVAILABLE = False
            raise
    return flash_attn


def _load_flash_attention_3():
    global flash_attn_interface, FLASH_ATTN_3_AVAILABLE
    if flash_attn_interface is None:
        try:
            flash_attn_interface = importlib.import_module("flash_attn_interface")
        except Exception:
            FLASH_ATTN_3_AVAILABLE = False
            raise
    return flash_attn_interface


def _load_block_sparse_attention():
    global block_sparse_attn_func, BLOCK_ATTN_AVAILABLE
    if block_sparse_attn_func is None:
        try:
            module = importlib.import_module("block_sparse_attn")
            block_sparse_attn_func = module.block_sparse_attn_func
        except Exception:
            BLOCK_ATTN_AVAILABLE = False
            raise
    return block_sparse_attn_func


def _load_sparse_sage_attention():
    global sparse_sageattn, SPARSE_SAGE_ATTN_AVAILABLE
    if sparse_sageattn is None:
        try:
            module = importlib.import_module(".sparse_sage.core", package=__package__)
            sparse_sageattn = module.sparse_sageattn
        except Exception:
            SPARSE_SAGE_ATTN_AVAILABLE = False
            raise
    return sparse_sageattn


def _load_sage_attention():
    global sageattn, SAGE_ATTN_AVAILABLE
    if sageattn is None:
        try:
            module = importlib.import_module("sageattention")
            sageattn = module.sageattn
        except Exception:
            SAGE_ATTN_AVAILABLE = False
            raise
    return sageattn


def _distribution_version(*names: str) -> Optional[str]:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _cuda_architecture() -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        return f"sm{major}{minor}"
    except Exception:
        return None


def _flash3_architecture_supported() -> bool:
    # The current FA3 Windows wheel is Hopper-only. Allow an explicit opt-in
    # for newer wheels that ship Blackwell kernels before they are recognized.
    if os.environ.get("FLASHVSR_ENABLE_EXPERIMENTAL_FA3", "").lower() in {"1", "true", "yes"}:
        return True
    return _cuda_architecture() in {"sm90"}


def attention_backend_status() -> dict:
    return {
        "cuda_arch": _cuda_architecture(),
        "flash_attention_2": {
            "available": FLASH_ATTN_2_AVAILABLE,
            "version": _distribution_version("flash-attn"),
        },
        "flash_attention_3": {
            "available": FLASH_ATTN_3_AVAILABLE and _flash3_architecture_supported(),
            "version": _distribution_version("flash-attn-3", "flash_attn_3"),
        },
        "sage_attention": {
            "available": SAGE_ATTN_AVAILABLE,
            "version": _distribution_version("sageattention"),
        },
        "block_sparse_attention": {
            "available": BLOCK_ATTN_AVAILABLE,
            "version": _distribution_version("block-sparse-attn", "block_sparse_attn"),
        },
        "sparse_sage_attention": {
            "available": SPARSE_SAGE_ATTN_AVAILABLE and _cuda_architecture() not in {"sm120", "sm121"},
            "version": None,
        },
    }


def normalize_attention_mode(attention_mode: Optional[str]) -> str:
    mode = attention_mode or ATTENTION_MODE
    mode = _ATTENTION_MODE_ALIASES.get(mode, mode)
    if mode not in ATTENTION_MODES:
        supported = ", ".join(ATTENTION_MODES)
        raise ValueError(f"Unknown attention mode '{mode}'. Supported modes: {supported}")
    return mode


def _backend_declared_available(mode: str) -> bool:
    if mode == "sdpa":
        return True
    arch = _cuda_architecture()
    if arch is None:
        return False
    if mode == "sparse_sage_attention":
        return SPARSE_SAGE_ATTN_AVAILABLE and arch not in {"sm120", "sm121"}
    if mode == "sage_attention":
        return SAGE_ATTN_AVAILABLE
    if mode == "block_sparse_attention":
        return BLOCK_ATTN_AVAILABLE
    if mode == "flash_attention_2":
        return FLASH_ATTN_2_AVAILABLE
    if mode == "flash_attention_3":
        return FLASH_ATTN_3_AVAILABLE and _flash3_architecture_supported()
    return False


def _auto_attention_candidates() -> List[str]:
    arch = _cuda_architecture()
    if arch in {"sm120", "sm121"}:
        # The reference FlashVSR path builds a draft/local block mask. Prefer
        # the Blackwell-capable block-sparse wheel so auto mode keeps those
        # attention semantics instead of silently switching to dense Sage.
        return ["block_sparse_attention", "sage_attention", "flash_attention_2", "flash_attention_3", "sdpa"]
    if arch == "sm90":
        return ["flash_attention_3", "flash_attention_2", "sage_attention", "sparse_sage_attention", "sdpa"]
    if arch is not None:
        return ["sparse_sage_attention", "block_sparse_attention", "flash_attention_2", "sage_attention", "sdpa"]
    return ["sdpa"]


def _fallback_attention_candidates(mode: str) -> List[str]:
    # Kept as a private compatibility helper for callers that imported it in
    # older forks. Explicit modes now have one safe fallback only: SDPA.
    return ["sdpa"]


def _initialize_attention_backend(mode: str) -> bool:
    try:
        if mode == "sparse_sage_attention":
            _load_sparse_sage_attention()
        elif mode == "sage_attention":
            _load_sage_attention()
        elif mode == "block_sparse_attention":
            _load_block_sparse_attention()
        elif mode == "flash_attention_2":
            _load_flash_attention_2()
        elif mode == "flash_attention_3":
            _load_flash_attention_3()
        return True
    except torch.OutOfMemoryError:
        # OOM is a resource failure, not evidence that the selected kernel is
        # incompatible. Let the pipeline's OOM recovery handle it without
        # poisoning backend availability for later inferences.
        raise
    except Exception as exc:
        _warn_attention_once(
            f"FlashVSR attention backend '{mode}' could not be initialized "
            f"({type(exc).__name__}: {exc})."
        )
        return False


def resolve_attention_mode(attention_mode: Optional[str]) -> str:
    requested = normalize_attention_mode(attention_mode)
    if requested != "auto":
        # An explicit selection is authoritative. Optional backends must not
        # replace one another merely because another wheel is installed.
        return requested if _backend_declared_available(requested) else "sdpa"

    for candidate in _auto_attention_candidates():
        if _backend_declared_available(candidate):
            return candidate
    return "sdpa"


def validate_attention_mode(attention_mode: Optional[str]) -> str:
    requested = normalize_attention_mode(attention_mode)
    candidates = _auto_attention_candidates() if requested == "auto" else [requested]
    for candidate in candidates:
        if not _backend_declared_available(candidate):
            continue
        if _initialize_attention_backend(candidate):
            return candidate

    if requested != "auto" and requested != "sdpa":
        _warn_attention_once(
            f"FlashVSR attention backend '{requested}' is unavailable; using PyTorch SDPA."
        )
    return "sdpa"

# ----------------------------
# Local / window masks
# ----------------------------
@torch.no_grad()
def build_local_block_mask_shifted_vec(block_h: int,
                                       block_w: int,
                                       win_h: int = 6,
                                       win_w: int = 6,
                                       include_self: bool = True,
                                       device=None) -> torch.Tensor:
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = torch.clamp(r_all - r_half, 0, H - win_h)
    end_r   = start_r + win_h - 1
    start_c = torch.clamp(c_all - c_half, 0, W - win_w)
    end_c   = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask

@torch.no_grad()
def build_local_block_mask_shifted_vec_normal_slide(block_h: int,
                                                   block_w: int,
                                                   win_h: int = 6,
                                                   win_w: int = 6,
                                                   include_self: bool = True,
                                                   device=None) -> torch.Tensor:
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = r_all - r_half
    end_r   = start_r + win_h - 1
    start_c = c_all - c_half
    end_c   = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask


class WindowPartition3D:
    """Partition / reverse-partition helpers for 5-D tensors (B,F,H,W,C)."""
    @staticmethod
    def partition(x: torch.Tensor, win: Tuple[int, int, int]):
        B, F, H, W, C = x.shape
        wf, wh, ww = win
        assert F % wf == 0 and H % wh == 0 and W % ww == 0, "Dims must divide by window size."
        x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, wf * wh * ww, C)

    @staticmethod
    def reverse(windows: torch.Tensor, win: Tuple[int, int, int], orig: Tuple[int, int, int]):
        F, H, W = orig
        wf, wh, ww = win
        nf, nh, nw = F // wf, H // wh, W // ww
        B = windows.size(0) // (nf * nh * nw)
        x = windows.view(B, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(B, F, H, W, -1)


@torch.no_grad()
def generate_draft_block_mask(batch_size, nheads, seqlen,
                              q_w, k_w, topk=10, local_attn_mask=None):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    avgpool_q = torch.mean(q_w, dim=1) 
    avgpool_k = torch.mean(k_w, dim=1)
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    avgpool_k = rearrange(avgpool_k, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    k_heads = avgpool_k.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = scores.shape[2] // local_attn_mask.shape[1]
    local_attn_mask = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    local_attn_mask = rearrange(local_attn_mask, 'x a y b -> (x a) (y b)')
    local_attn_mask = local_attn_mask.unsqueeze(0).repeat(repeat_head, 1, 1)
    local_attn_mask = local_attn_mask.to(torch.float32)
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == False, -float('inf'))
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == True, 0)
    scores = scores + local_attn_mask

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    n = flat.shape[1]
    apply_topk = min(flat.shape[1]-1, topk)
    thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
    thresholds = thresholds.unsqueeze(1)
    mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
    mask_new = rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)  # keep shape note
    # 修正：上行变量名统一
    # mask_new = rearrange(attn_map, 'h (it s1) s2 -> h (it s1) s2', it=seqlen) * 0 + mask_new
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask


@torch.no_grad()
def generate_draft_block_mask_sage(batch_size, nheads, seqlen,
                                      q_w, k_w, topk=10, local_attn_mask=None):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    
    avgpool_q = torch.mean(q_w, dim=1) 
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    
    k_w_split = k_w.view(k_w.shape[0], 2, 64, k_w.shape[2])
    avgpool_k_split = torch.mean(k_w_split, dim=2)
    avgpool_k_refined = rearrange(avgpool_k_split, 's two d -> (s two) d', two=2) # shape: (s*2, C)
    avgpool_k_refined = rearrange(avgpool_k_refined, 's (h d) -> s h d', h=nheads) # shape: (s*2, h, d)
    k_heads_doubled = avgpool_k_refined.permute(1, 0, 2) # shape: (h, s*2, d)
    
    k_heads_1, k_heads_2 = torch.chunk(k_heads_doubled, 2, dim=1)
    scores_1 = torch.einsum("hld,hmd->hlm", q_heads, k_heads_1) / math.sqrt(D)
    scores_2 = torch.einsum("hld,hmd->hlm", q_heads, k_heads_2) / math.sqrt(D)
    scores = torch.cat([scores_1, scores_2], dim=-1)

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = (scores.shape[2] // 2) // local_attn_mask.shape[1]
    
    local_attn_mask = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    local_attn_mask = rearrange(local_attn_mask, 'x a y b -> (x a) (y b)')
    local_attn_mask = local_attn_mask.repeat_interleave(2, dim=1)
    local_attn_mask = local_attn_mask.unsqueeze(0).repeat(repeat_head, 1, 1)
    
    assert scores.shape == local_attn_mask.shape, \
        f"Scores shape {scores.shape} != Mask shape {local_attn_mask.shape}"
    
    local_attn_mask = local_attn_mask.to(torch.float32)
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == False, -float('inf'))
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == True, 0)
    scores = scores + local_attn_mask

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    apply_topk = min(flat.shape[1]-1, topk)
    
    if apply_topk <= 0:
        mask_new = torch.zeros_like(flat, dtype=torch.bool).reshape(loop_num, s1, s2)
    else:
        thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
        thresholds = thresholds.unsqueeze(1)
        mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
        
    mask_new = rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask


# ----------------------------
# Attention kernels
# ----------------------------
def _sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int):
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


def _block_masked_sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 num_heads: int, attention_mask: torch.Tensor):
    """SDPA fallback for a block-level mask without materializing a token mask."""
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)

    if attention_mask.ndim == 3:
        attention_mask = attention_mask.unsqueeze(0)
    if attention_mask.ndim != 4:
        raise ValueError(
            "Block attention mask must have shape [batch, heads, query_blocks, key_blocks]."
        )

    batch_size, _, query_length, head_dim = q.shape
    key_length = k.shape[2]
    mask_batch, mask_heads, query_blocks, key_blocks = attention_mask.shape
    if mask_batch not in (1, batch_size):
        raise ValueError(f"Block attention mask batch dimension {mask_batch} cannot broadcast to {batch_size}.")
    if mask_heads not in (1, num_heads):
        raise ValueError(f"Block attention mask head dimension {mask_heads} cannot broadcast to {num_heads}.")
    if query_blocks == 0 or key_blocks == 0:
        raise ValueError("Block attention mask must contain at least one query and key block.")
    if query_length % query_blocks or key_length % key_blocks:
        raise ValueError(
            "Attention sequence lengths must be divisible by their block-mask dimensions "
            f"(query={query_length}/{query_blocks}, key={key_length}/{key_blocks})."
        )

    attention_mask = attention_mask.to(device=q.device, dtype=torch.bool)
    attention_mask = attention_mask.expand(batch_size, num_heads, query_blocks, key_blocks)
    query_block_size = query_length // query_blocks
    key_block_size = key_length // key_blocks

    q_blocks = q.reshape(batch_size, num_heads, query_blocks, query_block_size, head_dim)
    k_blocks = k.reshape(batch_size, num_heads, key_blocks, key_block_size, head_dim)
    v_blocks = v.reshape(batch_size, num_heads, key_blocks, key_block_size, head_dim)
    output_blocks = torch.empty_like(q_blocks)
    has_empty_rows = False

    # The emergency path launches SDPA per block/head and gathers only allowed
    # key blocks. Memory stays proportional to selected blocks instead of the
    # full query_length x key_length token mask.
    for batch_index in range(batch_size):
        for head_index in range(num_heads):
            head_k = k_blocks[batch_index, head_index]
            head_v = v_blocks[batch_index, head_index]
            for query_block_index in range(query_blocks):
                allowed_blocks = torch.nonzero(
                    attention_mask[batch_index, head_index, query_block_index],
                    as_tuple=False,
                ).flatten()
                if allowed_blocks.numel() == 0:
                    output_blocks[batch_index, head_index, query_block_index].zero_()
                    has_empty_rows = True
                    continue

                selected_k = head_k.index_select(0, allowed_blocks).reshape(1, 1, -1, head_dim)
                selected_v = head_v.index_select(0, allowed_blocks).reshape(1, 1, -1, head_dim)
                query_block = q_blocks[
                    batch_index, head_index, query_block_index
                ].unsqueeze(0).unsqueeze(0)
                output_blocks[batch_index, head_index, query_block_index] = (
                    F.scaled_dot_product_attention(query_block, selected_k, selected_v)[0, 0]
                )

    if has_empty_rows:
        _warn_attention_once(
            "FlashVSR block attention mask contains empty rows; their fallback output was set to zero."
        )
    x = output_blocks.reshape(batch_size, num_heads, query_length, head_dim)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


def _flash_attention_2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int):
    flash_attention_2_module = _load_flash_attention_2()
    q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
    x = flash_attention_2_module.flash_attn_func(q, k, v)
    return rearrange(x, "b s n d -> b s (n d)", n=num_heads)


def _flash_attention_3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int):
    flash_attention_3_module = _load_flash_attention_3()
    q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
    x = flash_attention_3_module.flash_attn_func(q, k, v)
    if isinstance(x, tuple):
        x = x[0]
    return rearrange(x, "b s n d -> b s (n d)", n=num_heads)


def _sage_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int):
    sage_attention_func = _load_sage_attention()
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = sage_attention_func(q, k, v, tensor_layout="HND", is_causal=False)
    if isinstance(x, tuple):
        x = x[0]
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


def _sparse_sage_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           num_heads: int, attention_mask: torch.Tensor):
    sparse_sage_attention_func = _load_sparse_sage_attention()
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = sparse_sage_attention_func(
        q, k, v,
        mask_id=attention_mask.to(torch.int8),
        is_causal=False,
        tensor_layout="HND",
    )
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


def _block_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            num_heads: int, attention_mask: torch.Tensor):
    block_sparse_attention_func = _load_block_sparse_attention()
    seqlen = q.shape[1]
    seqlen_kv = k.shape[1]
    q = rearrange(q, "b s (n d) -> (b s) n d", n=num_heads)
    k = rearrange(k, "b s (n d) -> (b s) n d", n=num_heads)
    v = rearrange(v, "b s (n d) -> (b s) n d", n=num_heads)
    cu_seqlens_q = torch.tensor([0, seqlen], device=q.device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, seqlen_kv], device=q.device, dtype=torch.int32)
    head_mask_type = torch.tensor([1] * num_heads, device=q.device, dtype=torch.int32)
    x = block_sparse_attention_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        head_mask_type,
        None,
        attention_mask,
        seqlen, seqlen_kv,
        0.0,
        deterministic=False,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    ).unsqueeze(0)
    return rearrange(x, "b s n d -> b s (n d)", n=num_heads)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int,
                    compatibility_mode=False, attention_mask=None, return_KV=False,
                    attention_mode=None):
    requested_mode = "sdpa" if compatibility_mode else normalize_attention_mode(attention_mode)
    mode = "sdpa" if compatibility_mode else resolve_attention_mode(requested_mode)
    sparse_modes = {"sparse_sage_attention", "block_sparse_attention"}

    if requested_mode in sparse_modes and mode == "sdpa" and attention_mask is not None:
        _warn_attention_once(
            f"FlashVSR attention backend '{requested_mode}' is unavailable; falling back to "
            "block-masked PyTorch SDPA and preserving the attention mask."
        )
        return _block_masked_sdpa_attention(q, k, v, num_heads, attention_mask)

    if requested_mode not in {"auto", "sdpa"} and mode != requested_mode:
        _warn_attention_once(
            f"FlashVSR attention backend '{requested_mode}' is unavailable; "
            "falling back to PyTorch SDPA."
        )

    if attention_mask is not None and mode not in sparse_modes:
        if mode != "sdpa":
            _warn_attention_once(
                f"FlashVSR attention backend '{mode}' cannot consume the model's block mask; "
                "using block-masked PyTorch SDPA for Self-Attention."
            )
        return _block_masked_sdpa_attention(q, k, v, num_heads, attention_mask)

    try:
        if mode == "sparse_sage_attention" and attention_mask is not None:
            return _sparse_sage_attention(q, k, v, num_heads, attention_mask)
        if mode == "block_sparse_attention" and attention_mask is not None:
            return _block_sparse_attention(q, k, v, num_heads, attention_mask)
        if mode in sparse_modes:
            _warn_attention_once(
                f"FlashVSR attention backend '{mode}' received no block mask; "
                "using PyTorch SDPA for this dense attention call."
            )
            return _sdpa_attention(q, k, v, num_heads)
        if mode == "sage_attention":
            return _sage_attention(q, k, v, num_heads)
        if mode == "flash_attention_2":
            return _flash_attention_2(q, k, v, num_heads)
        if mode == "flash_attention_3":
            return _flash_attention_3(q, k, v, num_heads)
    except torch.OutOfMemoryError:
        # Preserve the backend state so the caller can recover memory and
        # retry the same user-selected implementation.
        raise
    except Exception as exc:
        global FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
        global BLOCK_ATTN_AVAILABLE, SPARSE_SAGE_ATTN_AVAILABLE, SAGE_ATTN_AVAILABLE
        if mode == "flash_attention_2":
            FLASH_ATTN_2_AVAILABLE = False
        elif mode == "flash_attention_3":
            FLASH_ATTN_3_AVAILABLE = False
        elif mode == "sage_attention":
            SAGE_ATTN_AVAILABLE = False
        elif mode == "block_sparse_attention":
            BLOCK_ATTN_AVAILABLE = False
        elif mode == "sparse_sage_attention":
            SPARSE_SAGE_ATTN_AVAILABLE = False
        if mode in sparse_modes and attention_mask is not None:
            _warn_attention_once(
                f"FlashVSR attention backend '{mode}' failed ({type(exc).__name__}: {exc}); "
                "falling back to block-masked PyTorch SDPA and preserving the attention mask."
            )
            return _block_masked_sdpa_attention(q, k, v, num_heads, attention_mask)

        _warn_attention_once(
            f"FlashVSR attention backend '{mode}' failed ({type(exc).__name__}: {exc}); "
            "falling back to PyTorch SDPA."
        )

    return _sdpa_attention(q, k, v, num_heads)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


# ----------------------------
# Norms & Blocks
# ----------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads, attention_mode=None):
        super().__init__()
        self.num_heads = num_heads
        self.attention_mode = resolve_attention_mode(attention_mode)

    def set_attention_mode(self, attention_mode):
        self.attention_mode = resolve_attention_mode(attention_mode)
        return self.attention_mode
        
    def forward(self, q, k, v, attention_mask=None):
        x = flash_attention(
            q=q,
            k=k,
            v=v,
            num_heads=self.num_heads,
            attention_mask=attention_mask,
            attention_mode=self.attention_mode,
        )
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)
        self.local_attn_mask = None

    def clear_runtime_cache(self):
        self.local_attn_mask = None
        self.local_attn_mask_h = None
        self.local_attn_mask_w = None
        self.local_range = None

    def forward(self, x, freqs, f=None, h=None, w=None, local_num=None, topk=None,
                train_img=False, block_id=None, kv_len=None, is_full_block=False,
                is_stream=False, pre_cache_k=None, pre_cache_v=None, local_range = 9):
        B, L, D = x.shape
        if is_stream and pre_cache_k is not None and pre_cache_v is not None:
            assert f==2, "f must be 2"
        if is_stream and (pre_cache_k is None or pre_cache_v is None):
            assert f==6, " start f must be 6"
        assert L == f * h * w, "Sequence length mismatch with provided (f,h,w)."

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)

        win = (2, 8, 8)
        q = q.view(B, f, h, w, D)
        k = k.view(B, f, h, w, D)
        v = v.view(B, f, h, w, D)

        q_w = WindowPartition3D.partition(q, win)
        k_w = WindowPartition3D.partition(k, win)
        v_w = WindowPartition3D.partition(v, win)

        seqlen = f//win[0]
        one_len = k_w.shape[0] // B // seqlen
        if pre_cache_k is not None and pre_cache_v is not None:
            k_w = torch.cat([pre_cache_k, k_w], dim=0)
            v_w = torch.cat([pre_cache_v, v_w], dim=0)

        block_n = q_w.shape[0] // B
        block_s = q_w.shape[1]
        block_n_kv = k_w.shape[0] // B

        reorder_q = rearrange(q_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n, block_s=block_s)
        reorder_k = rearrange(k_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n_kv, block_s=block_s)
        reorder_v = rearrange(v_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n_kv, block_s=block_s)

        window_size = win[0]*h*w//128

        attention_mode = self.attn.attention_mode
        if (self.local_attn_mask is None
                or self.local_attn_mask_h != h // 8
                or self.local_attn_mask_w != w // 8
                or self.local_range != local_range
                or self.local_attn_mask.device != k_w.device):
            self.local_attn_mask = build_local_block_mask_shifted_vec_normal_slide(h//8, w//8, local_range, local_range, include_self=True, device=k_w.device)
            self.local_attn_mask_h = h//8
            self.local_attn_mask_w = w//8
            self.local_range = local_range

        if attention_mode == "sparse_sage_attention":
            attention_mask = generate_draft_block_mask_sage(
                B, self.num_heads, seqlen, q_w, k_w,
                topk=topk, local_attn_mask=self.local_attn_mask,
            )
        else:
            attention_mask = generate_draft_block_mask(
                B, self.num_heads, seqlen, q_w, k_w,
                topk=topk, local_attn_mask=self.local_attn_mask,
            )

        x = self.attn(reorder_q, reorder_k, reorder_v, attention_mask)

        cur_block_n, cur_block_s, _ = k_w.shape
        cache_num = cur_block_n // one_len
        if cache_num > kv_len:
            cache_k = k_w[one_len:, :, :]
            cache_v = v_w[one_len:, :, :]
        else:
            cache_k = k_w
            cache_v = v_w

        x = rearrange(x, 'b (block_n block_s) d -> (b block_n) (block_s) d', block_n=block_n, block_s=block_s)
        x = WindowPartition3D.reverse(x, win, (f, h, w))
        x = x.view(B, f*h*w, D)

        if is_stream:
            return self.o(x), cache_k, cache_v
        return self.o(x)


class CrossAttention(nn.Module):
    """
    仅考虑文本 context；提供持久 KV 缓存。
    """
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

        # 持久缓存
        self.cache_k = None
        self.cache_v = None

    @torch.no_grad()
    def init_cache(self, ctx: torch.Tensor):
        """ctx: [B, S_ctx, dim] —— 经过 text_embedding 之后的上下文"""
        self.cache_k = self.norm_k(self.k(ctx))
        self.cache_v = self.v(ctx)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor, y: torch.Tensor, is_stream: bool = False):
        """
        y 即文本上下文（未做其他分支）。
        """
        q = self.norm_q(self.q(x))
        assert self.cache_k is not None and self.cache_v is not None
        k = self.cache_k
        v = self.cache_v

        x = self.attn(q, k, v)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)

        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, f, h, w, local_num=None, topk=None,
                train_img=False, block_id=None, kv_len=None, is_full_block=False,
                is_stream=False, pre_cache_k=None, pre_cache_v=None, local_range = 9):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_output, self_attn_cache_k, self_attn_cache_v = self.self_attn(
            input_x, freqs, f, h, w, local_num, topk, train_img, block_id,
            kv_len=kv_len, is_full_block=is_full_block, is_stream=is_stream,
            pre_cache_k=pre_cache_k, pre_cache_v=pre_cache_v, local_range = local_range)

        x = self.gate(x, gate_msa, self_attn_output)
        x = x + self.cross_attn(self.norm3(x), context, is_stream=is_stream)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        if is_stream:
            return x, self_attn_cache_k, self_attn_cache_v
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


# ----------------------------
# WanModel (no image branch) — init 时即产生 KV 缓存
# ----------------------------
class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        # init_context: torch.Tensor,     # <<<< 必填：在 __init__ 里用它生成 cross-attn KV 缓存
        has_image_input: bool = False,
        attention_mode: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.attention_mode = resolve_attention_mode(attention_mode)

        # patch embed
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # text / time embed
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)

        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        self._cross_kv_initialized = False

    def set_attention_mode(self, attention_mode: str) -> str:
        requested_mode = normalize_attention_mode(attention_mode)
        effective_mode = validate_attention_mode(requested_mode)
        self.requested_attention_mode = requested_mode
        self.attention_mode = effective_mode
        for module in self.modules():
            if isinstance(module, AttentionModule):
                module.set_attention_mode(effective_mode)
        return effective_mode

    # 可选：手动清空 / 重新初始化
    def clear_cross_kv(self):
        for blk in self.blocks:
            blk.cross_attn.clear_cache()
        self._cross_kv_initialized = False

    def clear_runtime_caches(self):
        for blk in self.blocks:
            blk.cross_attn.clear_cache()
            blk.self_attn.clear_runtime_cache()
        self._cross_kv_initialized = False

    @torch.no_grad()
    def reinit_cross_kv(self, new_context: torch.Tensor):
        ctx_txt = self.text_embedding(new_context)
        for blk in self.blocks:
            blk.cross_attn.init_cache(ctx_txt)
        self._cross_kv_initialized = True

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                LQ_latents: Optional[List[torch.Tensor]] = None,
                train_img: bool = False,
                topk_ratio: Optional[float] = None,
                kv_ratio: Optional[float] = None,
                local_num: Optional[int] = None,
                is_full_block: bool = False,
                causal_idx: Optional[int] = None,
                **kwargs,
                ):
        # time / text embeds
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # 这里仍会嵌入 text（CrossAttention 若已有缓存会忽略它）
        # context = self.text_embedding(context)

        # 输入打补丁
        x, (f, h, w) = self.patchify(x)
        B = x.shape[0]

        # window / masks 超参
        win = (2, 8, 8)
        seqlen = f//win[0]
        if local_num is None:
            local_random = random.random()
            if local_random < 0.3:
                local_num = seqlen - 3
            elif local_random < 0.4:
                local_num = seqlen - 4
            elif local_random < 0.5:
                local_num = seqlen - 2
            else:
                local_num = seqlen

        window_size = win[0]*h*w//128
        square_num = window_size*window_size
        topk_ratio = 2.0
        topk = min(max(int(square_num*topk_ratio), 1), int(square_num*seqlen)-1)

        if kv_ratio is None:
            kv_ratio = (random.uniform(0., 1.0)**2)*(local_num-2-2)+2
        kv_len = min(max(int(window_size*kv_ratio), 1), int(window_size*seqlen)-1)

        decay_ratio = random.uniform(0.7, 1.0)

        # RoPE 3D
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        # blocks
        for block_id, block in enumerate(self.blocks):
            if LQ_latents is not None and block_id < len(LQ_latents):
                x += LQ_latents[block_id]

            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, f, h, w, local_num, topk,
                            train_img, block_id, kv_len, is_full_block, False,
                            None, None,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, f, h, w, local_num, topk,
                        train_img, block_id, kv_len, is_full_block, False,
                        None, None, 
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs, f, h, w, local_num, topk,
                          train_img, block_id, kv_len, is_full_block, False,
                          None, None)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    

# ----------------------------
# State dict converter（保持原映射；已忽略 has_image_input 使用）
# ----------------------------
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        # 保留原有哈希匹配返回的 config；实现本身不使用 has_image_input 分支
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 16,"dim": 1536,"ffn_dim": 8960,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 12,"num_layers": 30,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 16,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 36,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 36,"dim": 1536,"ffn_dim": 8960,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 12,"num_layers": 30,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 48,"dim": 1536,"ffn_dim": 8960,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 12,"num_layers": 30,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 48,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 36,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6,"has_image_pos_emb": False}
        else:
            config = {}
        return state_dict, config
