import torch
import torch.nn as nn
import math
from transformers.models.phi3.modeling_phi3 import Phi3Attention 
# Note: Phi-4 likely shares architecture with Phi-3 or similar in transformers library. 
# Depending on the specific transformers version, the class name might be PhiAttention or Phi3Attention.
# We will inspect the model structure dynamically to be safe, but import for typing if needed.

def monarch_forward_patch(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
    """
    Monkey-patched forward pass for Phi-4 Attention using Monarch Attention approximation.
    """
    bsz, q_len, _ = hidden_states.size()

    # 1. Projection
    qkv = self.qkv_proj(hidden_states)
    query_pos = self.num_heads * self.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
    value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

    # Reshape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    # RoPE
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    # Dynamic import of apply_rotary_pos_emb based on module content or fallback
    # self.rotary_emb is usually an nn.Module. 
    # The helper function is often in the same module as the Attention class.
    
    # Try to get the function from the module of the class
    try:
        module_utils = sys.modules[self.__module__]
        apply_rotary_pos_emb = getattr(module_utils, "apply_rotary_pos_emb")
    except (KeyError, AttributeError):
        # Fallback for Phi-3 / Llama
        try:
            from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb
        except ImportError:
            try: 
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            except ImportError:
                raise ImportError("Could not find apply_rotary_pos_emb function.")

    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # KV Cache
    if past_key_value is not None:
        # Phi-3 uses cache_kwargs with sin/cos
        cache_kwargs = {"sin": sin, "cos": cos} 
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # 2. GQA -> Repeat KV
    key_states = torch.repeat_interleave(key_states, dim=1, repeats=self.num_heads // self.num_key_value_heads)
    value_states = torch.repeat_interleave(value_states, dim=1, repeats=self.num_heads // self.num_key_value_heads)

    # 3. Monarch Attention Logic
    Q = query_states / math.sqrt(self.head_dim)
    K = key_states
    V = value_states
    
    attn_output = monarch_attention_impl(Q, K, V)
    
    # 4. Final Projection
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

import sys
import types

def apply_monarch_monkey_patch(model):
    """
    Applies the surgical Monarch patch to the model.
    Dynamically finds the attention layer class to patch.
    """
    print("Applying Monarch Attention Monkey Patch...")
    
    # Find the attention class from the first layer
    # Usually model.model.layers[0].self_attn
    try:
        if hasattr(model, "model"):
             # Common for Phi/Llama
             example_layer = model.model.layers[0]
        elif hasattr(model, "layers"):
             example_layer = model.layers[0]
        else:
             # Fallback search
             example_layer = None
             for m in model.modules():
                 if hasattr(m, "self_attn"):
                     example_layer = m
                     break
        
        if example_layer and hasattr(example_layer, "self_attn"):
            attn_class = example_layer.self_attn.__class__
            print(f"Detected Attention Class: {attn_class.__name__}")
        else:
            print("Could not detect attention class structure automatically.")
            attn_class = None
            
    except Exception as e:
        print(f"Error during autodetection: {e}")
        attn_class = None
    
    count = 0
    # Patch instances
    for name, module in model.named_modules():
        if attn_class and isinstance(module, attn_class):
            print(f"Patching layer: {name}")
            module.forward = types.MethodType(monarch_forward_patch, module)
            count += 1
        elif not attn_class and "Attention" in module.__class__.__name__:
             # Semantic fallback
             print(f"Patching likely candidate: {name} ({module.__class__.__name__})")
             module.forward = types.MethodType(monarch_forward_patch, module)
             count += 1
            
    if count == 0:
        print("WARNING: No layers patched!")
    else:
        print(f"Patched {count} layers.")
        
    return model

if __name__ == "__main__":
    # Test stub
    pass
