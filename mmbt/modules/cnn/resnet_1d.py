from torch import nn
import math
import torch.utils.model_zoo as model_zoo
from timm.models.registry import register_model
import torch.nn.functional as F
import torch
from collections import OrderedDict
import numpy as np
from torchsummary import summary

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed1d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.stride = stride
        self.dropout = nn.Dropout(.2)
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool1d(stride, ceil_mode=True)),
                ("0", nn.Conv1d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm1d(planes * self.expansion))
            ]))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv1d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool1d(stride, ceil_mode=True) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv1d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool1d(stride, ceil_mode=True)),
                ("0", nn.Conv1d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm1d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool1d(nn.Module):
    def __init__(self, temporal_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(temporal_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class Stem(nn.Module):
    def __init__(self, in_chan=3, width=64):
        super(Stem, self).__init__()
        self.conv1 = nn.Conv1d(in_chan, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x


class ModifiedResNet(nn.Module):

    def __init__(self, block, layers, output_dim, heads=8, in_chan=3, input_len=1250, width=64):
        super(ModifiedResNet, self).__init__()
        self.input_len = input_len

        # the 3-layer stem
        self.stem = Stem(in_chan, width)

        # residual layers
        self._inplanes = width
        # self.layers = nn.ModuleList(
        #     [
        #         self._make_layer(block, width, layers[0], stride=1),
        #         self._make_layer(block, width * 2, layers[1], stride=2),
        #         self._make_layer(block, width * 4, layers[2], stride=2),
        #         self._make_layer(block, width * 8, layers[3], stride=2)
        #     ]
        # )
        self.layer1 = self._make_layer(block, width, layers[0])
        self.layer2 = self._make_layer(block, width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, width * 8, layers[3], stride=2)
        # self.layer4 = self._make_layer(block, output_dim, layers[3], stride=2)

        embed_dim = width * 32 if block.expansion == 4 else width * 8
        self.attnpool = AttentionPool1d(math.ceil(input_len / 32), embed_dim, heads, output_dim)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        layers = [block(self._inplanes, planes, stride)]

        self._inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        # x = self.avgpool(x).squeeze(-1)


        return x

@register_model
def resnet18(output_dim, in_chan=3, **kwargs):
    model = ModifiedResNet(BasicBlock, [2, 2, 2, 2], output_dim=output_dim, in_chan=in_chan, **kwargs)
    return model

@register_model
def resnet34(output_dim, in_chan=3, **kwargs):
    model = ModifiedResNet(BasicBlock, [3, 4, 6, 3], output_dim=output_dim, in_chan=in_chan, **kwargs)
    return model

@register_model
def resnet50(output_dim, in_chan=3, **kwargs):
    model = ModifiedResNet(Bottleneck, [3, 4, 6, 3], output_dim=output_dim, in_chan=in_chan, **kwargs)
    return model

@register_model
def resnet101(output_dim, in_chan=3, **kwargs):
    model = ModifiedResNet(Bottleneck, [3, 4, 23, 3], output_dim=output_dim, in_chan=in_chan, **kwargs)
    return model

@register_model
def resnet152(output_dim, in_chan=3, **kwargs):
    model = ModifiedResNet(Bottleneck, [3, 8, 36, 3], output_dim=output_dim, in_chan=in_chan, **kwargs)
    return model


if __name__ == '__main__':
    import torch

    x = torch.randn(64, 3, 1250)
    m = resnet18(output_dim=768)
    # summary(m, (3, 1250))
    output = m(x)
    # m_child = nn.Sequential(*list(m.children())[:-2])
    # output = m_child(x)
    print(output.shape)

