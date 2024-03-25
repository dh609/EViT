import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from functools import partial
from thop import profile
from thop import clever_format

import torch
from torch.nn import Module


class FeatureMap(Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.
        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner


class ActivationFunctionFeatureMap(FeatureMap):
    """Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device):
        return

    def forward(self, x):
        return self.activation_function(x)


elu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.elu(x) + 1
)



class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding = 1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, bias=False, inplace=False):
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, stride=stride, padding= padding),
            norm_layer(out_channels),
            act_layer(inplace=inplace)
        )
class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding = 1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,  stride=stride, padding = padding),
            norm_layer(out_channels)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,stride=stride)
        )

class basicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 1, stride = 1, bias = False):
        super(basicBlock, self).__init__()
        self.conv1 = ConvBNAct(in_channel, out_channel, kernel_size=3, stride=1, padding=1, inplace=True)
        self.conv2 = ConvBN(out_channel, out_channel, kernel_size=3, stride=1,padding=1)
        self.down = ConvBNAct(in_channel, out_channel, kernel_size=1, stride=1, padding=0, inplace=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        down = self.down(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + down
        x = self.relu(x)
        return x

class Mlp(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNAct(in_features, hidden_features, kernel_size=1,padding=0)
        self.fc2 = nn.Sequential(nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features),
                                 norm_layer(hidden_features),
                                 act_layer())
        self.fc3 = ConvBN(hidden_features, out_features, kernel_size=1,padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.drop(x)

        return x

class RPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rpe_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.rpe_norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        return x + self.rpe_norm(self.rpe_conv(x))


class Stem(nn.Module):
    def __init__(self, inchannel, out_channel):
        super(Stem, self).__init__()
        self.conv1 = ConvBNAct(inchannel, out_channel, kernel_size=3, stride=2, padding=1, inplace= True)
        self.conv2 = ConvBNAct(out_channel, out_channel, kernel_size=3, stride=2, padding=1,inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LWMSA(nn.Module):
    def __init__(self,
                 dim=16,
                 num_heads=8,
                 window_size=16,
                 qkv_bias=False
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.eps = 1e-6
        self.ws = window_size

        self.qkv = Conv(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.proj = ConvBN(dim, dim, kernel_size=1)
        self.featuremap = elu_feature_map(dim)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps))
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps))
        return x


    def forward(self, x):
        _, _, H, W = x.shape
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        hh, ww = Hp//self.ws, Wp//self.ws
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h d (ws1 ws2)',
                            b=B, h=self.num_heads, d=C//self.num_heads, qkv=3, ws1=self.ws, ws2=self.ws)
        qs = q.chunk(self.num_heads,dim=1)
        ks = k.chunk(self.num_heads,dim=1)
        vs = v.chunk(self.num_heads,dim=1)
        feat = 0
        feats_out = []
        for i in range(self.num_heads):
            q = qs[i]
            k = ks[i]
            v = vs[i]
            if i > 0:
                q = q + feat
                k = k + feat
                v = v + feat
            self.featuremap.new_feature_map(q.device)
            q = self.featuremap.forward_queries(q).permute(0, 1, 3,2)
            k = self.featuremap.forward_keys(k)
            z = 1/(torch.einsum("bhnm, bhc->bhn", q, torch.sum(k, dim=-1) + self.eps))
            kv = torch.einsum('bhmn, bhcn->bhmc', k, v)
            attn = torch.einsum("bhnm, bhmc, bhn ->bhcn", q, kv, z)
            feat = attn
            feats_out.append(feat)
        attn = torch.cat(feats_out, dim=1)
        attn = rearrange(attn, '(b hh ww) h d (ws1 ws2) -> b (h d) (hh ws1) (ww ws2)',
                         b=B, h=self.num_heads, d=C // self.num_heads, ws1=self.ws, ws2=self.ws,
                         hh=Hp // self.ws, ww=Wp // self.ws)
        attn = attn[:, :, :H, :W]

        return attn.contiguous()


class Block(nn.Module):
    def __init__(self, dim=16, num_heads=8,  mlp_ratio=0.25, qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, window_size=16):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.ws = window_size
        self.attn = LWMSA(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                          window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.BatchNorm2d, rpe=True):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = norm_layer(dim)
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1, bias=False)
        self.rpe = rpe
        if self.rpe:
            self.proj_rpe = RPE(out_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        if self.rpe:
            x = self.proj_rpe(x)
        return x


class StageModule(nn.Module):
    def __init__(self, num_layers=2, in_dim=96, out_dim=96, num_heads=8, mlp_ratio=4., qkv_bias=False, use_pm=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=-1, shuffle=False):
        super().__init__()
        self.use_pm = use_pm
        if self.use_pm:
            self.patch_partition = PatchMerging(in_dim, out_dim)

        self.layers = nn.ModuleList([])
        for idx in range(num_layers):
            self.layers.append(Block(dim=out_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, drop=drop,
                                     drop_path=drop_path, act_layer=act_layer, window_size=window_size,
                                     norm_layer=norm_layer))

    def forward(self, x):
        if self.use_pm:
            x = self.patch_partition(x)
        for block in self.layers:
            x = block(x)
        return x


class Glb(nn.Module):
    def __init__(self,  mlp_ratio=4., window_sizes=[16, 16, 16, 16],
                 layers=[2, 2, 6, 2], num_heads=[4, 8, 16, 32], dims=[96, 192, 384, 768],
                 qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3):
        super().__init__()


        self.encoder_channels = dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]  # stochastic depth decay rule
        self.stage1 = StageModule(layers[0], dims[0], dims[0], num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  use_pm=False, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr[0], window_size=window_sizes[0])
        self.stage2 = StageModule(layers[1], dims[0], dims[1], num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  use_pm=True, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr[1], window_size=window_sizes[1])
        self.stage3 = StageModule(layers[2], dims[1], dims[2], num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  use_pm=True, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr[2], window_size=window_sizes[2])
        self.stage4 = StageModule(layers[3], dims[2], dims[3], num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  use_pm=True, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr[3], window_size=window_sizes[3])

    def forward(self, x):
        features = []
        x = self.stage1(x)
        features.append(x)
        x = self.stage2(x)
        features.append(x)
        x = self.stage3(x)
        features.append(x)
        x = self.stage4(x)
        features.append(x)

        return features


class DetailPath(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        dim1 = embed_dim // 2
        dim2 = embed_dim
        self.dp1 = basicBlock(96, dim1)
        self.dp2 = basicBlock(dim1, dim2)

    def forward(self, x):
        x = self.dp1(x)
        x = self.dp2(x)
        return x


class FPN(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 256, 512), decoder_channels=256):
        super().__init__()
        self.pre_conv0 = Conv(encoder_channels[0], decoder_channels, kernel_size=1)
        self.pre_conv1 = Conv(encoder_channels[1], decoder_channels, kernel_size=1)
        self.pre_conv2 = Conv(encoder_channels[2], decoder_channels, kernel_size=1)
        self.pre_conv3 = Conv(encoder_channels[3], decoder_channels, kernel_size=1)

        self.post_conv3 = nn.Sequential(ConvBNAct(decoder_channels, decoder_channels),
                                        nn.UpsamplingBilinear2d(scale_factor=2),
                                        ConvBNAct(decoder_channels, decoder_channels),
                                        nn.UpsamplingBilinear2d(scale_factor=2),
                                        ConvBNAct(decoder_channels, decoder_channels))

        self.post_conv2 = nn.Sequential(ConvBNAct(decoder_channels, decoder_channels),
                                        nn.UpsamplingBilinear2d(scale_factor=2),
                                        ConvBNAct(decoder_channels, decoder_channels))

        self.post_conv1 = ConvBNAct(decoder_channels, decoder_channels)
        self.post_conv0 = ConvBNAct(decoder_channels, decoder_channels)

    def upsample_add(self, up, x):
        up = F.interpolate(up, x.size()[-2:], mode='nearest')
        up = torch.mul(up, x) + x
        return up

    def forward(self, x0, x1, x2, x3):
        x3 = self.pre_conv3(x3)
        x2 = self.pre_conv2(x2)
        x1 = self.pre_conv1(x1)
        x0 = self.pre_conv0(x0)
        # print(x0.shape, 'x0')
        # print(x1.shape, 'x1')
        # print(x2.shape, 'x2')
        # print(x3.shape, 'x3')

        x2 = self.upsample_add(x3, x2)
        x1 = self.upsample_add(x2, x1)
        x0 = self.upsample_add(x1, x0)

        x3 = self.post_conv3(x3)
        x3 = F.interpolate(x3, x0.size()[-2:], mode='bilinear', align_corners=False)

        x2 = self.post_conv2(x2)
        x2 = F.interpolate(x2, x0.size()[-2:], mode='bilinear', align_corners=False)

        x1 = self.post_conv1(x1)
        x1 = F.interpolate(x1, x0.size()[-2:], mode='bilinear', align_corners=False)

        x0 = self.post_conv0(x0)

        x0 = x3 + x2 + x1 + x0
        print(x0.shape)
        out = []
        out.append(x0)
        out.append(x3)
        print(x3.shape)
        return out

class SPP(nn.Module):
    def __init__(self, channel):
        super(SPP, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool3 = nn.AdaptiveAvgPool2d(4)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(8)
        self.conv = nn.Conv2d(channel,channel,1)
    def forward(self, x):
        x1 = self.conv(self.avg_pool1(x))
        x2 = self.conv(self.avg_pool2(x))
        x2 = F.interpolate(x2, x.size()[-2:], mode='bilinear', align_corners=False)
        x3 = self.conv(self.avg_pool3(x))
        x3 = F.interpolate(x3, x.size()[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv(self.avg_pool4(x))
        x4 = F.interpolate(x4, x.size()[-2:], mode='bilinear', align_corners=False)
        out = x1 + x2 + x3 + x4 + x
        return out

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # b, c, h, w -> b, c, 1, 1 -> b, c
        avg = self.avg_pool(x).view(b, c)
        # b, c -> b, c // ratio -> b, c -> b, c, 1, 1
        fc = self.fc(avg).view(b, c, 1, 1)
        return x * fc

class GLBViT(nn.Module):
    def __init__(self,
                 decoder_channels=384,
                 dims=[96, 192, 384, 768],
                 window_sizes=[16, 16, 16, 16],
                 num_classes=2):
        super().__init__()
        self.stem = Stem(inchannel=3, out_channel=dims[0])
        self.backbone = Glb(layers=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],
                                    dims=dims, window_sizes=window_sizes)

        encoder_channels = self.backbone.encoder_channels
        self.dp = DetailPath(embed_dim=decoder_channels)
        self.rpe = RPE(dims[0])
        self.se = se_block(384)
        self.fpn = FPN(encoder_channels, decoder_channels)
        self.spp = SPP(384)
        self.conv = ConvBNAct(384,384,3)
        self.head = nn.Sequential(ConvBNAct(decoder_channels, encoder_channels[0]),
                                  nn.Dropout(0.1),
                                  nn.UpsamplingBilinear2d(scale_factor=2),
                                  Conv(encoder_channels[0], num_classes, kernel_size=1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        sz = x.size()[-2:]
        x = self.stem(x)
        dp = self.dp(x)
        x = self.rpe(x)
        x, x2, x3, x4 = self.backbone(x)
        x = self.fpn(x, x2, x3, x4)
        #[x, xmin]
        xm = x[1]
        xm = torch.ones_like(xm) - torch.sigmoid(xm)
        x = x[0] + dp
        add = x
        x = self.se(x)
        x = self.spp(x)
        x = x + add

        x2 = torch.mul(x, xm)
        x2 = self.conv(x2)
        x = x + x2
        x = self.head(x)

        x = F.interpolate(x, sz, mode='bilinear', align_corners=False)
        return x

if __name__ == '__main__':
    input = torch.randn(3, 3, 512, 512)
    net = GLBViT()
    out = net(input)
    print(out.shape)

