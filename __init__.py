from .nodes import NegativeAttentionPatchNode, ConcatSneakyConditioning

NODE_CLASS_MAPPINGS = {
    "Negative cross attention": NegativeAttentionPatchNode,
    "Negative cross attention concatenate": ConcatSneakyConditioning,
}