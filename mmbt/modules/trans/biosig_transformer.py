from functools import partial
import torch
import torch.nn as nn
from mmbt.modules.position_embedding import CosinePositionEncoding, LearnablePositionalEncoding
from timm.models.layers import DropPath, trunc_normal_
from mmbt.config import ex
from timm.models.registry import register_model
from mmbt.modules.cnn.resnet_1d import resnet18
from torchsummary import summary
import copy


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        _x, attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(
        self,
        sig_size=1250,
        patch_size=50,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()

        self.num_patches = sig_size//patch_size
        self.patch_size = patch_size
        self.sig_size = sig_size
        self.in_chans = in_chans

        self.proj = nn.Conv1d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x):
        # x = self.patchify(x)
        x = x[:, :, :self.num_patches*self.patch_size]
        x = self.proj(x)
        return x


class BioSigTransformer(nn.Module):
    def __init__(
            self,
            sig_size=1250,
            patch_size=50,
            in_chans=1,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=None,
            add_norm_before_transformer=False,
            hybrid_backbone=None,
            config=None,
    ):
        """
        Args:
            sig_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of trans
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        drop_rate = drop_rate if config is None else config["drop_rate"]


        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.add_norm_before_transformer = add_norm_before_transformer

        self.patch_embed = PatchEmbed(
            sig_size=sig_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        ) if hybrid_backbone is None else hybrid_backbone

        num_patches = self.patch_embed.num_patches


        self.patch_size = patch_size
        self.patch_dim = sig_size // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if config["pos_embed"] == "CS":
            self.pos_embed = CosinePositionEncoding(embed_dim=embed_dim, seq_len=num_patches+1)
        if config["pos_embed"] == "LP":
            self.pos_embed = LearnablePositionalEncoding(embed_dim=embed_dim, seq_len=num_patches+1)
        if config["pos_embed"] == "":
            self.pos_embed = nn.Identity()
        self.pos_drop = nn.Dropout(p=drop_rate)


        if add_norm_before_transformer:
            self.pre_norm = norm_layer(embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}


    def signal_embed(self, _sig):

        B, _, _ = _sig.shape

        sig_embed = self.patch_embed(_sig).permute(0, 2, 1)

        sig_embed = torch.cat(
            (self.cls_token.expand(B, -1, -1),
             sig_embed)
            , dim=1)

        sig_embed = self.pos_drop(
            self.pos_embed(sig_embed)
        )

        if self.add_norm_before_transformer:
            sig_embed = self.pre_norm(sig_embed)

        return sig_embed


    def forward_features(self, _sig):
        sig_embed = self.signal_embed(_sig)
        x = sig_embed

        for blk in self.blocks:
            x, _ = blk(x)

        x = self.norm(x)
        return x

    def forward(self, _sig):
        x = self.forward_features(_sig)
        x = x[:, 0]
        x = self.head(x)
        return x


@register_model
def sit_base_patch50(pretrained=False, **kwargs):
    model = BioSigTransformer(
        patch_size=50,
        embed_dim=512,
        depth=4,
        num_heads=8,
        **kwargs
    )
    return model

@register_model
def bist_custom_patch50(pretrained=False, **kwargs):
    model = BioSigTransformer(
        sig_size=kwargs["config"]["ppg_size"],
        patch_size=50,
        embed_dim=kwargs["config"]["hidden_size"],
        depth=kwargs["config"]["num_layers"],
        num_heads=kwargs["config"]["num_heads"],
        add_norm_before_transformer=True,
        **kwargs
    )
    return model

@register_model
def bist_base_patch50(pretrained=False, **kwargs):
    model = BioSigTransformer(
        patch_size=50,
        embed_dim=768,
        depth=6,
        num_heads=12,
        **kwargs
    )
    return model

@register_model
def sit_large_patch50(pretrained=False, **kwargs):
    model = BioSigTransformer(
        patch_size=50,
        embed_dim=512,
        depth=6,
        num_heads=12,
        **kwargs
    )
    return model
