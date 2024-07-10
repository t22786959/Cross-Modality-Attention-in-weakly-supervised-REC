import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch import einsum


def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        only_attend_immediate_media = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(
        self,
        x,
        media,
        media_locations = None
    ):
        b, t, m = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        k, v = self.to_kv(media).chunk(2, dim = -1)
        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        attn = sim.softmax(dim = -1)
        out = einsum('... i j, ... j d -> ... j d', attn, v)
        return self.to_out(out)

class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        only_attend_immediate_media = True
    ):
        super().__init__()

        self.inpu = nn.Linear(1024, 512, bias = False)
        self.attn = CrossAttention(dim = dim, dim_head = dim_head, heads = heads, only_attend_immediate_media = only_attend_immediate_media)
        self.attn2 = CrossAttention(dim = dim, dim_head = dim_head, heads = heads, only_attend_immediate_media = only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))
        self.attn_gate2 = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult = ff_mult)
        self.ff2 = FeedForward(dim, mult = ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))
        self.ff_gate2 = nn.Parameter(torch.tensor([0.]))

        self.coefficient = nn.Parameter(torch.tensor(0.1))
        self.mapping_lang = self._make_mlp(512, 512, 0.1)
        self.gamma = nn.ModuleList(nn.Linear(512, 512) for _ in [512])
        self.beta = nn.ModuleList(nn.Linear(512, 512) for _ in [512])
        self.norms = nn.ModuleList([nn.InstanceNorm2d(512) for _ in [ 512]])


    def forward(
        self,
        x,
        y,                  # media tensor, encoded by perceiver resample - (batch, time, latents, dim)
        media_locations = None  # boolean tensor indicating positions of media - (batch, sequence)
    ):
        batch_size, channels, height, width = x.size()  
        x = x.view(batch_size,channels,-1).permute(0,2,1)
        
        x = self.inpu(x)
        x = self.attn(x , x, media_locations = media_locations) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh()  + x
        x_t = x
        y_feature  = self.mapping_lang(y)
        gamma = [F.tanh(gamma(y_feature)) for gamma in self.gamma]
        beta = [F.tanh(beta(y_feature)) for beta in self.beta]
        
        x = self.norms[0](x).permute(0,2,1)
        b = beta[0].view(batch_size, -1, 1).expand_as(x)
        g = gamma[0].view(batch_size, -1, 1).expand_as(x)
        x = F.relu(g * x + b)
        
        x_t = x_t.permute(0,2,1)
        x = x*self.coefficient + x_t
        
        x = x.permute(0,2,1)
        x_t = x_t.permute(0,2,1)

        x = self.attn2(x_t , x, media_locations = media_locations) * self.attn_gate2.tanh() + x_t
        x = self.ff2(x) * self.ff_gate2.tanh()  + x_t

        x = x.permute(0,2,1).view(batch_size,512,height,width)

        return x
    
    def _make_mlp(self, input_dim, output_dim, drop):
        return nn.Sequential(nn.Linear(input_dim, output_dim), 
                nn.BatchNorm1d(output_dim), 
                nn.ReLU(inplace=True), 
                nn.Dropout(drop),
                nn.Linear(output_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True))
    

