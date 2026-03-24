from collections import OrderedDict
from typing import Tuple, Union

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pearl import pa_attn

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.width = width
        self.heads = heads
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.arch, self.attn_strategy, self.gaussian_std = None, None, 0
        self.addition_cache = dict()

        self.use_kk=True
        self.alpha_kk=0.08
        self.align_solver='polar'
        self.polar_iters=5

    def set_attn_options(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def set_params(self, arch, attn_strategy):
        assert arch in ['reduced', 'vanilla']
        assert attn_strategy in ['pearl', 'vanilla']

        self.arch, self.attn_strategy = arch, attn_strategy

    def forward(self, x: torch.Tensor, return_all=False):
        B, nc, w, h = x.shape
        n_patches = (w // self.patch_size, h // self.patch_size)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        for blk in self.transformer.resblocks[:-1]:
            x = blk(x)
        blk = self.transformer.resblocks[-1]
        if self.arch == 'vanilla':
            x = x + self.custom_attn(blk.attn, blk.ln_1(x), n_patches)
            x = x + blk.mlp(blk.ln_2(x))
        elif self.arch == 'reduced':
            x = self.custom_attn(blk.attn, blk.ln_1(x), n_patches)
        else:
            raise NotImplemented

        x = x.permute(1, 0, 2)  # LND -> NLD

        if return_all:
            return self.ln_post(x) @ self.proj

        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.positional_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.positional_embedding
        class_pos_embed = self.positional_embedding[[0]]
        patch_pos_embed = self.positional_embedding[1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), mode='bicubic',
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), align_corners=False, recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    @staticmethod
    def gaussian_window(dim1, dim2, std=1.):
        constant = 1 / (std * math.sqrt(2))
        ks = list()
        for dim in [dim1, dim2]:
            start = -(dim - 1) / 2.0
            k = torch.linspace(start=start * constant,
                               end=(start + (dim - 1)) * constant,
                               steps=dim,
                               dtype=torch.float)
            ks.append(k)
        dist_square_to_mu = (torch.stack(torch.meshgrid(*ks, indexing='ij')) ** 2).sum(0)
        return torch.exp(-dist_square_to_mu)

    @staticmethod
    def get_attention_addition(dim1, dim2, window, adjust_for_cls=True):
        m = torch.einsum('ij,kl->ijkl', torch.eye(dim1), torch.eye(dim2))
        m = m.permute((0, 3, 1, 2)).contiguous()  # m[ijkl] = 1 iff (i, j) == (k, l)
        out = F.conv2d(m.view(-1, dim1, dim2).unsqueeze(1), window.unsqueeze(0).unsqueeze(1), padding='same').squeeze(1)
        out = out.view(dim1 * dim2, dim1 * dim2)
        if adjust_for_cls:
            v_adjusted = torch.vstack([torch.zeros((1, dim1 * dim2)), out])
            out = torch.hstack([torch.zeros((dim1 * dim2 + 1, 1)), v_adjusted])
        return out

    @torch.no_grad()
    def get_last_layer_attn(self, img: torch.Tensor,
                            attn_strategy: str = 'pearl',
                            gaussian_std: float = 1.0,
                            use_kk: bool = True,
                            alpha_kk: float = 0.3):
        self.eval()
        B, _, H, W = img.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        x = self.conv1(img)                       # [B,C,Hp,Wp]
        x = x.reshape(B, self.width, -1).permute(0, 2, 1)   # [B,HpWp,C]
        cls = (self.class_embedding.to(x.dtype)
            + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device))
        x = torch.cat([cls, x], dim=1)

        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, H, W).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x).transpose(0, 1)       # [T,B,C]

        for blk in self.transformer.resblocks[:-1]:
            x = blk(x)
        last_blk = self.transformer.resblocks[-1]
        z = last_blk.ln_1(x)                     # [T,B,C]

        self.attn_strategy = attn_strategy
        self.gaussian_std = gaussian_std
        self.use_kk = use_kk
        self.alpha_kk = alpha_kk
        A = self.custom_attn(last_blk.attn, z, (Hp, Wp), return_attn=True)  # [B*H,T,T]
        num_heads = last_blk.attn.num_heads
        T, B_, C = z.shape
        A = A.view(B, num_heads, T, T)
        return A, (Hp, Wp)

    def custom_attn(self, attn_layer, x, n_patches, return_attn: bool = False, with_attn: bool = False):
        num_heads = attn_layer.num_heads
        num_tokens, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)

        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # ------------------------------------------------------------------
        # PEARL  (Procrustes Alignment in Self-Attention)
        # ------------------------------------------------------------------

        if self.attn_strategy == 'pearl':
            if return_attn:
                attn_weights = pa_attn(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    detach_cls=True,
                    use_kk=self.use_kk,
                    alpha_kk=self.alpha_kk,
                    align_solver=self.align_solver,
                    polar_iters=self.polar_iters,
                    return_attn=True
                )
                return attn_weights  # [B*H,T,T]

            if with_attn:
                attn_output, attn_weights = pa_attn(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    detach_cls=True,
                    use_kk=self.use_kk,
                    alpha_kk=self.alpha_kk,
                    align_solver=self.align_solver,
                    polar_iters=self.polar_iters,
                    with_attn=True
                )
            else:
                attn_output = pa_attn(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    detach_cls=True,
                    use_kk=self.use_kk,
                    alpha_kk=self.alpha_kk,
                    align_solver=self.align_solver,
                    polar_iters=self.polar_iters
                )

            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
            )
            attn_output = attn_layer.out_proj(attn_output)
            if with_attn:
                return attn_output, attn_weights
            return attn_output

        # ------------------------------------------------------------------
        # VANILLA  (CLIP Attention)
        # ------------------------------------------------------------------

        elif self.attn_strategy == 'vanilla':
            attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)

            if return_attn:
                return attn_weights

            attn_output = torch.bmm(attn_weights, v)
            attn_output = (
                attn_output.transpose(0, 1)
                .contiguous()
                .view(-1, bsz, embed_dim)
            )
            attn_output = attn_layer.out_proj(attn_output)

            if with_attn:
                return attn_output, attn_weights
            return attn_output

        else:
            raise NotImplementedError(f'attn_strategy {self.attn_strategy} is not implemented.')


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,  # 512
                 # vision
                 image_resolution: int,  # 224
                 vision_layers: Union[Tuple[int, int, int, int], int],  # 12
                 vision_width: int,  # 768
                 vision_patch_size: int,  # 16
                 # text
                 context_length: int,  # 77
                 vocab_size: int,  # 49408
                 transformer_width: int,  # 512
                 transformer_heads: int,  # 8
                 transformer_layers: int  # 12
                 ):
        super().__init__()
        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_all=False):
        return self.visual(image.type(self.dtype), return_all=return_all)

    def get_image_last_attn(self, image: torch.Tensor,
                            attn_strategy: str = 'vanilla',
                            gaussian_std: float = 1.0,
                            use_kk: bool = True,
                            alpha_kk: float = 0.3):
        return self.visual.get_last_layer_attn(
            image.type(self.dtype),
            attn_strategy=attn_strategy,
            gaussian_std=gaussian_std,
            use_kk=use_kk,
            alpha_kk=alpha_kk
        )

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
