# This code is partially adapted from https://github.com/lucidrains/flamingo-pytorch/blob/main/flamingo_pytorch/flamingo_pytorch.py

import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch.utils.checkpoint import checkpoint

def exists(val):
    return val is not None

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False)
    )

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_attention_head=64, heads=8):
        super().__init__()
        self.scale = dim_attention_head ** -0.5
        self.heads = heads
        inner_dim = dim_attention_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads
        q = self.to_q(latents)

        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h=h)
        q = q * self.scale

        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h=h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(self, *, dim, depth, dim_attention_head=64, heads=8, num_latents=64, perceiver_output_dim=4, ff_mult=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(perceiver_output_dim, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_attention_head=dim_attention_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        times = x.shape[1]
        x = x + self.media_pos_emb[:times]

        latents = repeat(self.latents, 'n d -> b m n d', b=x.shape[0], m=x.shape[1])

        for attn, ff in self.layers:
            # Apply gradient checkpointing to attention and feedforward layers
            latents = checkpoint(attn, x, latents, use_reentrant=False) + latents
            latents = checkpoint(ff, latents, use_reentrant=False) + latents

        return self.norm(latents)

class MaskedCrossAttention(nn.Module):
    def __init__(self, *, dim, dim_attention_head=64, heads=8, only_attend_immediate_media=True):
        super().__init__()
        self.scale = dim_attention_head ** -0.5
        self.heads = heads
        inner_dim = dim_attention_head * heads

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, x, media, media_locations=None):
        b, t, m = media.shape[:3]
        h = self.heads
        x = self.norm(x)
        q = self.to_q(x)
        media = rearrange(media, 'b t n d -> b (t n) d')
        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)

        q = q * self.scale
        sim = einsum('... i d, ... j d -> ... i j', q, k)

        if exists(media_locations):
            text_time = media_locations.cumsum(dim=-1)
            media_time = torch.arange(t, device=x.device) + 1
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge
            text_to_media_mask = mask_op(rearrange(text_time, 'b i -> b 1 i 1'), repeat(media_time, 'j -> 1 1 1 (j m)', m=m))
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if exists(media_locations) and self.only_attend_immediate_media:
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
            attn = attn.masked_fill(text_without_media_mask, 0.)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, *, dim_text, dim_media, dim_attention_head=64, heads=8, ff_mult=4, only_attend_immediate_media=False):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim_text,
            dim_attention_head=dim_attention_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.]))
        self.ff = FeedForward(dim_text, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))
        if dim_text != dim_media:
            self.media_projection = nn.Linear(dim_media, dim_text)
        else:
            self.media_projection = nn.Identity()

    def forward(self, x, media):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        if media.ndim == 4:
            media = self.media_projection(media).squeeze(1)
        else:
            media = self.media_projection(media)

        # Apply gradient checkpointing to cross-attention and feedforward layers
        media = checkpoint(self.attn, media, x , use_reentrant=False ) * self.attn_gate.tanh() + media
        media = checkpoint(self.ff, media , use_reentrant=False) * self.ff_gate.tanh() + media

        return media



def build_gca_components(config, requires_grad=True):
    """
    Builder function that instantiates the PerceiverResampler and GatedCrossAttentionBlock
    based on a configuration dictionary, and sets requires_grad for their parameters.
    
    Args:
        config (dict): Configuration dictionary containing the required parameters.
        requires_grad (bool): Whether the parameters of the components should require gradients.
        
    Returns:
        perceiver_resampler (nn.Module): The PerceiverResampler instance.
        gated_cross_attention_block (nn.Module): The GatedCrossAttentionBlock instance.
    """
    #config.mm_hidden_size, config.hidden_size

    # print("config", config)

    # Extract parameters for PerceiverResampler from config
    dim_media = getattr(config, 'mm_hidden_size', None)
    dim_text = getattr(config, 'llm_hidden_size', None)
    perceiver_heads = getattr(config, 'perceiver_heads', 8)
    gated_attention_heads = getattr(config, 'gated_attention_heads', 8)
    perceiver_depth = getattr(config, 'perceiver_depth', 2)
    dim_attention_head = getattr(config, 'dim_attention_head', 64)
    perceiver_num_latents = getattr(config, 'num_media_tokens', 128)
    perceiver_output_dim = getattr(config, 'llm_hidden_size', None)
    ff_mult = getattr(config, 'ff_mult', 2)


    # Instantiate the PerceiverResampler
    perceiver_resampler = PerceiverResampler(
        dim=dim_media,
        depth=perceiver_depth,
        dim_attention_head=dim_attention_head,
        heads=perceiver_heads,
        num_latents=perceiver_num_latents,
        perceiver_output_dim=perceiver_output_dim,
        ff_mult=ff_mult
    )
    
    # Instantiate the GatedCrossAttentionBlock
    gated_cross_attention = GatedCrossAttentionBlock(
        dim_text=dim_text,
        dim_media=perceiver_output_dim,
        dim_attention_head=dim_attention_head,
        heads=gated_attention_heads,
        ff_mult=ff_mult
    )

    # Set requires_grad for PerceiverResampler and GatedCrossAttentionBlock parameters
    for param in perceiver_resampler.parameters():
        param.requires_grad = requires_grad
    
    for param in gated_cross_attention.parameters():
        param.requires_grad = requires_grad        

    return perceiver_resampler, gated_cross_attention
