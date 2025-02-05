from comfy.ldm.modules.attention import optimized_attention
from comfy.model_management import interrupt_current_processing
from copy import deepcopy
import torch

class attention_patch():
    def __init__(self, renorm, negative_strength):
        self.negative_strength = negative_strength
        self.renorm = renorm

    def attention_with_negative(self, q, k, v, extra_options, mask=None, attn_precision=None):
        heads = extra_options if isinstance(extra_options, int) else extra_options['n_heads']

        if k.shape[-2] // 77 <= 1:
            return optimized_attention(q, k, v, heads, mask, attn_precision)

        negative_k = k[:, k.size(-2)//2:,:]
        negative_v = v[:, v.size(-2)//2:,:]

        out_pos = optimized_attention(q, k[:,:k.size(-2)//2 ,:], v[:,:v.size(-2)//2 ,:],heads,mask,attn_precision)

        if self.negative_strength == 0:
            return out_pos

        out_neg = optimized_attention(q, negative_k, negative_v, heads, mask, attn_precision)

        if self.renorm:
            out_norm = out_pos.norm()

        out = out_pos + (out_pos - out_neg) * self.negative_strength

        if self.renorm:
            out = out * out_norm / out.norm()

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
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 1/10, "round":1/1000}),
                    "rescale_after": ("BOOLEAN", {"default": False, "tooltip": "Ensures that the scale of the output is the same as before taking the difference.\nThis can fix over bright/dark results and help to raise the scale higher."}),
                    }
                }

    TOGGLES = {}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("Model",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/negative attention"

    def patch(self, model, strength, rescale_after):
        m = model.clone()
        m = nan_interrupt_patch(m) # high scales can cause black images, let's not sample this.

        levels = ["input","middle","output"]
        layer_names = [[l, n, True] for l in levels for n in range(12)]

        patch = attention_patch(negative_strength=strength, renorm=rescale_after)

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