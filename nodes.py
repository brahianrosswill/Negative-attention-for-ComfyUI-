from comfy.ldm.modules.attention import optimized_attention
from comfy.model_management import interrupt_current_processing
from copy import deepcopy
import torch
import math

def scaled_dot_product_attention_with_negative(query, key, value, negative_key=None, negative_value=None, negative_strength=1, attn_mask=None, is_causal=False, scale=None, renorm=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf")).to(query.device)
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    negative_attn_weight = query @ negative_key.transpose(-2, -1) * scale_factor

    diff_val = (torch.softmax(attn_weight+attn_bias, dim=-1) @ value) - (torch.softmax(negative_attn_weight+attn_bias, dim=-1) @ negative_value)
    proj = torch.nn.Linear(diff_val.size(-1), S, bias=False, dtype=diff_val.dtype).to(diff_val.device)
    diff = proj(diff_val)

    attn_weight = (attn_weight + attn_bias + diff * negative_strength).softmax(dim=-1)

    return attn_weight @ value

class attention_patch():
    def __init__(self, negative_strength):
        self.negative_strength = negative_strength

    def attention_with_negative(self, q, k, v, extra_options, mask=None, attn_precision=None):
        heads = extra_options if isinstance(extra_options, int) else extra_options['n_heads']

        if k.shape[-2] // 77 <= 1:
            return optimized_attention(q, k, v, heads, mask, attn_precision)

        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

        negative_k = k[:,:, k.size(-2)//2:,:]
        negative_v = v[:,:, k.size(-2)//2:,:]

        scale = 1 / math.sqrt(q.size(-1))
        out = scaled_dot_product_attention_with_negative(q, k[:,:,:k.size(-2)//2 ,:], v[:,:,:v.size(-2)//2 ,:], negative_key=negative_k, negative_value=negative_v, negative_strength=self.negative_strength, attn_mask=mask, scale=scale)

        out = (
            out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        )
        return out

def nan_interrupt_patch(model):
    def interrupt_on_nan(args):
        denoised = args["denoised"]
        if torch.isnan(denoised).any() or torch.isinf(denoised).any():
            print(" NaN values detected. Interrupting.")
            interrupt_current_processing()
        return denoised
    m = model.clone()
    m.set_model_sampler_post_cfg_function(interrupt_on_nan)
    return m

class NegativeAttentionPatchNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 1/4, "round":1/1000}),
                    }
                }

    TOGGLES = {}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("Model",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/negative attention"

    def patch(self, model, strength):
        m = model.clone()
        m = nan_interrupt_patch(m)

        levels = ["input","middle","output"]
        layer_names = [[l, n, True] for l in levels for n in range(12)]

        patch = attention_patch(negative_strength=strength)

        for current_level, b_number, toggle in layer_names:
            m.set_model_attn2_replace(patch.attention_with_negative, current_level, b_number)

        return (m,)

class ConcatSneakyConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "concat_mode":  (["crop_to_shortest","prolongate_to_longest_by_loop","prolongate_to_longest_with_empty_or_0"],),
                    "negative_out": (["empty_or_0","invert","crop_to_77_tokens"],),
                    },
                "optional":{
                    "empty": ("CONDITIONING",),
                }
                }

    TOGGLES = {}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("Positive", "Negative",)
    FUNCTION = "exec"

    CATEGORY = "model_patches/negative attention"

    def combine_conds(self, cond_pos, cond_neg, cond_empty, mode, neg_mode=""):
        if cond_pos.shape[-2] == cond_neg.shape[-2]:
            return torch.cat((cond_pos, cond_neg), dim=-2)

        if mode == "crop_to_shortest":
            shortest = min(cond_pos.shape[-2], cond_neg.shape[-2])
            cond_pos = cond_pos[:,:shortest,:]
            cond_neg = cond_neg[:,:shortest,:]

        elif mode == "prolongate_to_longest_by_loop":
            longest = max(cond_pos.shape[-2], cond_neg.shape[-2])
            if cond_pos.shape[-2] < longest:
                cond_pos = self.loop_tensor(cond_pos, longest)
            if cond_neg.shape[-2] < longest:
                cond_neg = self.loop_tensor(cond_neg, longest)

        elif mode == "prolongate_to_longest_with_empty_or_0":
            longest = max(cond_pos.shape[-2], cond_neg.shape[-2])
            if cond_empty is None:
                if cond_pos.shape[-2] < longest:
                    cond_pos = self.pad_zero_tensor(cond_pos, longest)
                if cond_neg.shape[-2] < longest:
                    cond_neg = self.pad_zero_tensor(cond_neg, longest)
            else:
                if cond_pos.shape[-2] < longest:
                    cond_pos = self.pad_empty_tensor(cond_pos, cond_empty, longest)
                if cond_neg.shape[-2] < longest:
                    cond_neg = self.pad_empty_tensor(cond_neg, cond_empty, longest)

        return torch.cat((cond_pos, cond_neg), dim=-2)

    def loop_tensor(self, tensor, target_length):
        repeat_times = (target_length + tensor.shape[-2] - 1) // tensor.shape[-2]
        repeated = tensor.repeat(1, repeat_times, 1)
        return repeated[:,:target_length,:]

    def pad_zero_tensor(self, tensor, target_length):
        pad_size = target_length - tensor.shape[-2]
        padding = torch.zeros(tensor.shape[0], pad_size, tensor.shape[-1], device=tensor.device)
        return torch.cat((tensor, padding), dim=-2)

    def pad_empty_tensor(self, tensor, empty, target_length):
        pad = empty.repeat(1, 20, 1)
        return torch.cat((tensor, pad), dim=-2)[:,:target_length,:]

    def swap_halves(self, tensor):
        mid = tensor.shape[-2] // 2
        first_half = tensor[...,:mid,:]
        second_half = tensor[...,mid:,:]
        return torch.cat((second_half, first_half), dim=-2)

    def exec(self, positive, negative, concat_mode, negative_out, empty=None):
        cond_empty = None
        if empty is not None:
            cond_empty = empty[0][0].clone()

        pos_out = self.combine_conds(positive[0][0].clone(), negative[0][0].clone(), cond_empty, concat_mode)

        if negative_out == "invert":
            neg_out = self.swap_halves(pos_out.clone())
    
        elif negative_out == "empty_or_0":
            if empty is None:
                neg_out = torch.zeros_like(pos_out)[...,:77,:]
            else:
                neg_out = empty[0][0].clone()[...,:77,:]

        elif negative_out == "crop_to_77_tokens":
            neg_out = negative[0][0].clone()[...,:77,:]

        positive_cond_out = deepcopy(positive)
        negative_cond_out = deepcopy(negative)
        positive_cond_out[0][0] = pos_out
        negative_cond_out[0][0] = neg_out

        return (positive_cond_out,negative_cond_out,)