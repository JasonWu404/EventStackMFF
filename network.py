# -*- coding: utf-8 -*-
# @Author  : Juntao Wu, XinZhe Xie
# @University  : University of Science and Technology of China, ZheJiang University

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import cv2
from utils import gray_to_colormap


class PatchEmbed2D(nn.Module):
    """
    Turn B,C,H,W -> B,NumP,Dim  via nn.Unfold
    """
    def __init__(self, patch_size: int = 4):
        super().__init__()
        self.ps = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        pad_h = (self.ps - H % self.ps) % self.ps
        pad_w = (self.ps - W % self.ps) % self.ps
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        Hp, Wp = x.shape[-2] // self.ps, x.shape[-1] // self.ps
        tokens = self.unfold(x)                        # [B, C*ps*ps, Hp*Wp]
        tokens = tokens.transpose(1, 2).contiguous()   # [B, N, Dim]
        return tokens, (H, W, Hp, Wp, pad_h, pad_w, C)


class Unpatchify2D(nn.Module):
    """
    Turn tokens back to B,C,H,W using nn.Fold
    NOTE: do NOT cache a single Fold; output_size varies across pyramid levels.
    """
    def __init__(self, patch_size: int = 4):
        super().__init__()
        self.ps = patch_size

    def forward(self, tokens, meta):
        # tokens: [B, N, Dim], meta from PatchEmbed2D
        H, W, Hp, Wp, pad_h, pad_w, C = meta
        B, N, Dim = tokens.shape
        assert N == Hp * Wp and Dim == C * self.ps * self.ps, \
            f"token shape mismatch: N={N} vs {Hp*Wp}, Dim={Dim} vs {C*self.ps*self.ps}"
        fold = nn.Fold(output_size=(Hp * self.ps, Wp * self.ps),
                       kernel_size=self.ps, stride=self.ps)
        x = tokens.transpose(1, 2)      # [B, Dim, N]
        x = fold(x)                     # [B, C, Hp*ps, Wp*ps]
        if pad_h or pad_w:
            x = x[:, :, :H, :W]
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, p=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class MultiHeadSelfAttn(nn.Module):
    def __init__(self, dim, num_heads=8, attn_p=0.0, proj_p=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):  # x: [B,N,D]
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                          # [B,h,N,dh]
        attn = (q @ k.transpose(-2, -1)) * self.scale             # [B,h,N,N]
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))


class MultiHeadCrossAttn(nn.Module):
    """
    Cross-attn: out = SA( query=x_q , key=x_k , value=x_v )
    """
    def __init__(self, dim, num_heads=8, attn_p=0.0, proj_p=0.0, invert_softmax=False):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        self.invert_softmax = invert_softmax

    def forward(self, q_in, kv_in):  # [B,N,D],[B,N,D]
        B, N, D = q_in.shape
        q = self.q(q_in).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(kv_in).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(kv_in).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        dp = (q @ k.transpose(-2, -1)) * self.scale
        if self.invert_softmax:
            dp = -dp
        attn = self.attn_drop(dp.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))


class CrossModalPatchXAttnBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, attn_p=0.0, drop=0.0):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_kv1 = nn.LayerNorm(dim)
        self.self_img = MultiHeadSelfAttn(dim, num_heads, attn_p, drop)
        self.self_evt = MultiHeadSelfAttn(dim, num_heads, attn_p, drop)

        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_kv2 = nn.LayerNorm(dim)
        self.xattn_evt2img = MultiHeadCrossAttn(dim, num_heads, attn_p, drop, invert_softmax=True)
        self.xattn_img2evt = MultiHeadCrossAttn(dim, num_heads, attn_p, drop, invert_softmax=True)

        self.norm_mlp_i = nn.LayerNorm(dim)
        self.mlp_i = MLP(dim, mlp_ratio, drop)
        self.norm_mlp_e = nn.LayerNorm(dim)
        self.mlp_e = MLP(dim, mlp_ratio, drop)

    def forward(self, img_tok, evt_tok):

        i0, e0 = img_tok, evt_tok
        img_tok = i0 + self.self_img(self.norm_q1(img_tok))
        evt_tok = e0 + self.self_evt(self.norm_kv1(evt_tok))

        i1, e1 = img_tok, evt_tok
        img_tok = i1 + self.xattn_evt2img(self.norm_q2(img_tok), self.norm_kv2(evt_tok))
        evt_tok = e1 + self.xattn_img2evt(self.norm_q2(evt_tok), self.norm_kv2(img_tok))

        img_tok = img_tok + self.mlp_i(self.norm_mlp_i(img_tok))
        evt_tok = evt_tok + self.mlp_e(self.norm_mlp_e(evt_tok))
        return img_tok, evt_tok


class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels=None, square_kernel_size=3,
                 band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.channel_adj = (
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
            if self.out_channels != in_channels else nn.Identity()
        )
        gc = max(1, int(self.out_channels * branch_ratio))
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size,
                                   padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size),
                                  padding=(0, band_kernel_size // 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1),
                                  padding=(band_kernel_size // 2, 0), groups=gc)
        self.split_indexes = (max(0, self.out_channels - 3 * gc), gc, gc, gc)

    def forward(self, x):
        x = self.channel_adj(x)
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((x_id, self.dwconv_hw(x_hw),
                          self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att = nn.ModuleList(
            [nn.Linear(c_list_sum, c) if split_att == 'fc'
             else nn.Conv1d(c_list_sum, c, 1)
             for c in c_list[:-1]]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat([self.avgpool(t) for t in (t1, t2, t3, t4, t5)], dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        outs = []
        for i, c in enumerate([t1, t2, t3, t4, t5]):
            if i < 4:
                a = self.sigmoid(self.att[i](att))
            else:
                a = self.sigmoid(self.att[-1](att))
            if self.split_att == 'fc':
                a = a.transpose(-1, -2).unsqueeze(-1).expand_as(c)
            else:
                a = a.unsqueeze(-1).expand_as(c)
            outs.append(a * c)
        return tuple(outs)


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(
            nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
            nn.Sigmoid()
        )

    def forward(self, t1, t2, t3, t4, t5):
        out = []
        for t in (t1, t2, t3, t4, t5):
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = self.shared_conv2d(torch.cat([avg_out, max_out], dim=1))
            out.append(att * t)
        return tuple(out)


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5
        s1, s2, s3, s4, s5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = s1, s2, s3, s4, s5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5
        c1, c2, c3, c4, c5 = self.catt(t1, t2, t3, t4, t5)
        return c1 + s1, c2 + s2, c3 + s3, c4 + s4, c5 + s5

class EdgeGate(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        mid = max(8, c // r)
        self.g = nn.Sequential(
            nn.Conv2d(c, mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, e):            # e: [B,C,H,W]
        return self.g(e)             # [B,1,H,W]

class NUnimodalRefiner(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.conv = nn.Conv3d(1, 1, kernel_size=(kernel_size, 1, 1),
                              padding=(kernel_size // 2, 0, 0), bias=False)
        t = torch.arange(kernel_size).float() - kernel_size // 2
        g = torch.exp(-0.5 * (t / sigma) ** 2)
        g = (g / g.sum()).view(1, 1, kernel_size, 1, 1)
        with torch.no_grad():
            self.conv.weight.copy_(g)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, logits_bn_hw):   # [B,N,H,W]
        x = logits_bn_hw.unsqueeze(1)  # [B,1,N,H,W]
        x = self.conv(x).squeeze(1)    # [B,N,H,W]
        return x

class TextureAwareTemperature(nn.Module):
    def __init__(self, tau_min=0.35, tau_max=1.6):
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.edge = nn.Conv2d(1, 8, 3, padding=1)
        self.body = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1), nn.Sigmoid()
        )

    @staticmethod
    def _grad_mag(x):  # x: [B,1,H,W]
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, guide_gray):  # [B,1,H,W]
        tex = self._grad_mag(guide_gray)
        h = self.body(self.edge(tex))
        
        tau_map = self.tau_min + (self.tau_max - self.tau_min) * (1.0 - h)
        return tau_map

class FastGuidedFilter(nn.Module):
    def __init__(self, radius=5, eps=1e-3):
        super().__init__()
        self.radius = radius
        self.eps = eps

    def _box_filter(self, x):
        r = self.radius
        ch = x.size(1)
        w = torch.ones(ch, 1, 2 * r + 1, 2 * r + 1,
                       device=x.device, dtype=x.dtype) / ((2 * r + 1) ** 2)
        return F.conv2d(x, w, padding=r, groups=ch)

    def forward(self, I, p):  # I: guide [B,1,H,W], p: input depth [B,1,H,W]
        mean_I  = self._box_filter(I)
        mean_p  = self._box_filter(p)
        mean_Ip = self._box_filter(I * p)
        cov_Ip  = mean_Ip - mean_I * mean_p

        mean_II = self._box_filter(I * I)
        var_I   = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I

        mean_a = self._box_filter(a)
        mean_b = self._box_filter(b)
        q = mean_a * I + mean_b
        return q

class ConfidenceFromProbs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs):  # [B,N,H,W]
        B, N, H, W = probs.shape
        eps = 1e-12
        entropy = -torch.sum(probs * torch.log(probs + eps),
                             dim=1, keepdim=True)  # [B,1,H,W]
        entropy = entropy / (torch.log(torch.tensor(float(N),
                                    device=probs.device)) + eps)
        peak = probs.max(dim=1, keepdim=True).values            # [B,1,H,W]
        conf = (1.0 - entropy) * peak
        return conf.clamp(0.0, 1.0)


class EdgeAwareDiffusion(nn.Module):
    def __init__(self, iters=12, lam=0.6, sigma=0.1):
        super().__init__()
        self.iters = iters
        self.lam = lam
        self.sigma = sigma

    @staticmethod
    def _shift(x, direction):
        if direction == 'left':
            return F.pad(x[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')
        if direction == 'right':
            return F.pad(x[:, :, :, :-1], (1, 0, 0, 0), mode='replicate')
        if direction == 'up':
            return F.pad(x[:, :, 1:, :], (0, 0, 1, 0), mode='replicate')
        if direction == 'down':
            return F.pad(x[:, :, :-1, :], (0, 0, 0, 1), mode='replicate')
        raise ValueError

    def forward(self, guide, depth_init, confidence):

        I = guide
        p = depth_init
        w = confidence

        gx = torch.abs(I - self._shift(I, 'left'))
        gy = torch.abs(I - self._shift(I, 'up'))
        wx = torch.exp(-(gx / self.sigma) ** 2) 
        wy = torch.exp(-(gy / self.sigma) ** 2)  

        q = p.clone()
        for _ in range(self.iters):
            ql = self._shift(q, 'left')
            qr = self._shift(q, 'right')
            qu = self._shift(q, 'up')
            qd = self._shift(q, 'down')

            wxr = self._shift(wx, 'right')
            wyl = self._shift(wy, 'down')

            num = w * p + self.lam * (wx * ql + wxr * qr +
                                      wy * qu + wyl * qd)
            den = w + self.lam * (wx + wxr + wy + wyl) + 1e-6
            q = num / den
        return q.clamp(0.0, 1.0)

class FeatureExtractionCMCA(nn.Module):
    def __init__(self, c_list=[8, 16, 24, 32, 48, 64],
                 split_att='fc', bridge=True,
                 patch_size=4, heads=4,
                 cmca_levels=(False, False, True, True, True),
                 ps_list=(8, 8, 4, 4, 4)):
        super().__init__()

        self.gain = [0.0, 0.0, 1.0, 1.0, 1.0]
        self.bridge = bridge
        self.ps = patch_size

        self.cmca_levels = cmca_levels
        assert len(ps_list) == 5 and len(cmca_levels) == 5
        self.ps_list = ps_list

        self.patchers = nn.ModuleList([PatchEmbed2D(ps) for ps in ps_list])
        self.unpatchers = nn.ModuleList([Unpatchify2D(ps) for ps in ps_list])

        self.im_enc1 = nn.Conv2d(1, c_list[0], 3, padding=1)
        self.ev_enc1 = nn.Conv2d(1, c_list[0], 3, padding=1)

        self.enc2 = InceptionDWConv2d(c_list[0], c_list[1])
        self.enc3 = InceptionDWConv2d(c_list[1], c_list[2])
        self.enc4 = InceptionDWConv2d(c_list[2], c_list[3])
        self.enc5 = InceptionDWConv2d(c_list[3], c_list[4])
        self.enc6 = InceptionDWConv2d(c_list[4], c_list[5])

        self.ebn = nn.ModuleList([
            nn.GroupNorm(4, c_list[0]),
            nn.GroupNorm(4, c_list[1]),
            nn.GroupNorm(4, c_list[2]),
            nn.GroupNorm(4, c_list[3]),
            nn.GroupNorm(4, c_list[4]),
        ])

        dims_c = c_list[:5]  # t1~t5
        dims_tok = [c * (ps_list[i] ** 2) for i, c in enumerate(dims_c)]
        self.cmca_blocks = nn.ModuleList([
            CrossModalPatchXAttnBlock(dim=d, num_heads=heads)
            for d in dims_tok
        ])

        self.egates = nn.ModuleList([EdgeGate(c) for c in c_list[:5]])

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            self.scab_evt = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used for both image and event')

        # decoder
        self.dec1 = InceptionDWConv2d(c_list[5], c_list[4])
        self.dec2 = InceptionDWConv2d(c_list[4], c_list[3])
        self.dec3 = InceptionDWConv2d(c_list[3], c_list[2])
        self.dec4 = InceptionDWConv2d(c_list[2], c_list[1])
        self.dec5 = InceptionDWConv2d(c_list[1], c_list[0])

        self.dbn = nn.ModuleList([
            nn.GroupNorm(4, c_list[4]),
            nn.GroupNorm(4, c_list[3]),
            nn.GroupNorm(4, c_list[2]),
            nn.GroupNorm(4, c_list[1]),
            nn.GroupNorm(4, c_list[0]),
        ])

        self.embbed_dim = c_list[0]

        self.evt_dim = 4
        self.evt_depth_head = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[0], 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_list[0], self.evt_dim, 1)
        )

        self.final = nn.Conv2d(c_list[0], self.embbed_dim, 1)

        self.gamma = nn.Parameter(torch.tensor(0.10))

    def _cmca_fuse(self, img_f, evt_f, level_idx, g=None):
        patcher   = self.patchers[level_idx]
        unpatcher = self.unpatchers[level_idx]

        itok, meta = patcher(img_f)   # [B, N, C*ps^2]
        etok, _    = patcher(evt_f)

        itok_f, etok_f = self.cmca_blocks[level_idx](itok, etok)
        img_refined = unpatcher(itok_f, meta)

        if g is None:
            g = 1.0
        if isinstance(g, torch.Tensor):
            g = g.detach()

        img_out = img_f + g * (img_refined - img_f)
        evt_out = evt_f
        return img_out, evt_out

    def forward(self, x):
        B, N, C, H, W = x.shape
        assert C == 2
        img = x[:, :, 0].reshape(B * N, 1, H, W)
        evt = x[:, :, 1].reshape(B * N, 1, H, W)

        g5_l = None

        # L1
        i1 = F.gelu(F.max_pool2d(self.ebn[0](self.im_enc1(img)), 2, 2))
        e1 = F.gelu(F.max_pool2d(self.ebn[0](self.ev_enc1(evt)), 2, 2))
        if self.cmca_levels[0]:
            g1 = self.gain[0] * self.egates[0](e1)
            i1, e1 = self._cmca_fuse(i1, e1, level_idx=0, g=g1)

        # L2
        i2 = F.gelu(F.max_pool2d(self.ebn[1](self.enc2(i1)), 2, 2))
        e2 = F.gelu(F.max_pool2d(self.ebn[1](self.enc2(e1)), 2, 2))
        if self.cmca_levels[1]:
            g2 = self.gain[1] * self.egates[1](e2)
            i2, e2 = self._cmca_fuse(i2, e2, level_idx=1, g=g2)

        # L3
        i3 = F.gelu(F.max_pool2d(self.ebn[2](self.enc3(i2)), 2, 2))
        e3 = F.gelu(F.max_pool2d(self.ebn[2](self.enc3(e2)), 2, 2))
        if self.cmca_levels[2]:
            g3 = self.gain[2] * self.egates[2](e3)
            i3, e3 = self._cmca_fuse(i3, e3, level_idx=2, g=g3)

        # L4
        i4 = F.gelu(F.max_pool2d(self.ebn[3](self.enc4(i3)), 2, 2))
        e4 = F.gelu(F.max_pool2d(self.ebn[3](self.enc4(e3)), 2, 2))
        if self.cmca_levels[3]:
            g4 = self.gain[3] * self.egates[3](e4)
            i4, e4 = self._cmca_fuse(i4, e4, level_idx=3, g=g4)

        # L5
        i5 = F.gelu(F.max_pool2d(self.ebn[4](self.enc5(i4)), 2, 2))
        e5 = F.gelu(F.max_pool2d(self.ebn[4](self.enc5(e4)), 2, 2))
        e5_raw = e5
        if self.cmca_levels[4]:
            g5_l = self.gain[4] * self.egates[4](e5)
            i5, e5 = self._cmca_fuse(i5, e5, level_idx=4, g=g5_l)

        # bridge
        if self.bridge:
            i1, i2, i3, i4, i5 = self.scab(i1, i2, i3, i4, i5)
            e1, e2, e3, e4, e5 = self.scab_evt(e1, e2, e3, e4, e5)
        i6 = F.gelu(self.enc6(i5))
        e6 = F.gelu(self.enc6(e5))
        g5 = (self.gain[4] * self.egates[4](e5)) if g5_l is None else g5_l
        g5 = g5.detach()
        z = i6 + torch.clamp(self.gamma, 0, 0.3) * (g5 * e6)

        # decoder
        d5 = F.gelu(self.dbn[0](self.dec1(z)));      d5 = d5 + i5
        d4 = F.gelu(F.interpolate(self.dbn[1](self.dec2(d5)), scale_factor=2,
                                  mode='bilinear', align_corners=True)); d4 = d4 + i4
        d3 = F.gelu(F.interpolate(self.dbn[2](self.dec3(d4)), scale_factor=2,
                                  mode='bilinear', align_corners=True)); d3 = d3 + i3
        d2 = F.gelu(F.interpolate(self.dbn[3](self.dec4(d3)), scale_factor=2,
                                  mode='bilinear', align_corners=True)); d2 = d2 + i2
        d1 = F.gelu(F.interpolate(self.dbn[4](self.dec5(d2)), scale_factor=2,
                                  mode='bilinear', align_corners=True)); d1 = d1 + i1

        out0 = F.interpolate(self.final(d1), scale_factor=2,
                             mode='bilinear', align_corners=True)
        focus_maps_logits = out0              # [B*N,Cemb,H,W]


        evt_depth_feats = self.evt_depth_head(e5_raw)  # [B*N, Ce, H/32, W/32]
        evt_depth_feats = F.interpolate(
            evt_depth_feats, size=(H, W),
            mode='bilinear', align_corners=True
        )  # [B*N, Ce, H, W]
        evt_depth_feats = evt_depth_feats.view(B, N, self.evt_dim, H, W)

        focus_maps_logits = focus_maps_logits.view(B, N, -1, H, W)
        return focus_maps_logits, evt_depth_feats

def apply_rotary_pos_emb(x, sin, cos):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([x1 * cos - x2 * sin,
                         x1 * sin + x2 * cos], dim=-1)
    x_rot = x_rot.flatten(-2)
    return x_rot


def get_rotary_emb(dim, seq_len, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(seq_len, device=device).float()
    sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)  # [seq_len, dim//2]
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    return sin, cos


class DepthTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim=None, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ff_dim = ff_dim or embed_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        sin, cos = get_rotary_emb(self.embed_dim, x.size(1), x.device)
        x_rot = apply_rotary_pos_emb(x, sin, cos)

        x_attn = self.attn(x_rot, x_rot, x_rot)[0]
        x_attn = self.dropout(x_attn)

        x = self.norm1(x_rot + x_attn)
        x_ffn = self.ffn(x)
        x_ffn = self.dropout(x_ffn)
        x = self.norm2(x + x_ffn)
        return x


class DepthTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, ff_dim=None, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            DepthTransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        B, N, C, H, W = x.shape
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, N, C)
        for layer in self.layers:
            x_flat = layer(x_flat)
        x = x_flat.reshape(B, H, W, N, C).permute(0, 3, 4, 1, 2)
        return x


class LayerInteraction(nn.Module):
    def __init__(self, embed_dim_f, embed_dim_e, num_transformer_layers=1):
        super(LayerInteraction, self).__init__()
        self.emb_f = embed_dim_f
        self.emb_e = embed_dim_e
        self.embed_dim = embed_dim_f + embed_dim_e

        self.layer_interaction_depth = DepthTransformer(
            embed_dim=self.embed_dim,
            num_heads=4,
            num_layers=num_transformer_layers
        )

    def forward(self, focus_maps, evt_depth_feats):
        x = torch.cat([focus_maps, evt_depth_feats], dim=2)  # [B,N,Cf+Ce,H,W]

        att_out = self.layer_interaction_depth(x)   # [B,N,Cf+Ce,H,W]

        depth_logits = att_out.max(dim=2, keepdim=False).values  # [B,N,H,W]
        return depth_logits

class DepthMapCreation(nn.Module):
    def __init__(self, tau=0.7, learnable=True):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(tau))) if learnable \
                   else torch.tensor(float(tau))

    def forward(self, depth_logits, num_images, tau_map=None):
        # depth_logits: [B,N,H,W]
        if tau_map is None:
            tau = torch.clamp(self.tau, 0.3, 2.0).view(1, 1, 1, 1)
        else:
            tau = tau_map.clamp(0.3, 2.0)  # [B,1,H,W]

        focus_probs = F.softmax(depth_logits / tau, dim=1)  # [B,N,H,W]

        depth_indices = torch.linspace(
            1, num_images, num_images, device=depth_logits.device
        ).view(1, -1, 1, 1)
        depth_map = torch.sum(focus_probs * depth_indices, dim=1, keepdim=True)
        depth_map = (depth_map - depth_map.min()) / \
                    (depth_map.max() - depth_map.min() + 1e-6)
        return depth_map, focus_probs

class EventStackMFF(nn.Module):
    def __init__(self, patch_size=4, heads=4,
                 return_probs=False, vis_root="./feature_maps"):
        super().__init__()
        self.return_probs = return_probs
        self.vis_root = vis_root

        self.feature_extraction = FeatureExtractionCMCA(
            patch_size=patch_size, heads=heads
        )
        self.layer_interaction = LayerInteraction(
            embed_dim_f=self.feature_extraction.embbed_dim,
            embed_dim_e=self.feature_extraction.evt_dim,
            num_transformer_layers=1
        )

        self.n_refine = NUnimodalRefiner(kernel_size=5, sigma=1.0)
        self.tau_predictor = TextureAwareTemperature(
            tau_min=0.35, tau_max=1.6
        )
        self.depth_map_creation = DepthMapCreation(
            tau=0.7, learnable=True
        )

        self.guided = FastGuidedFilter(radius=5, eps=1e-3)
        self.conf_est = ConfidenceFromProbs()
        self.diffuse = EdgeAwareDiffusion(
            iters=12, lam=0.6, sigma=0.08
        )

        self.apply(self._init_weights_wrapper)

    def _init_weights_wrapper(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def generate_fused_image(x, depth_map):
        """
        x: [B,N,H,W]
        depth_map: [B,1,H,W] in [0,1]
        """
        B, N, H, W = x.shape
        depth_map = depth_map.squeeze(1)  # [B,H,W]
        depth_map_continuous = (depth_map * (N - 1)).clamp(0, N - 1)
        depth_map_index = torch.round(depth_map_continuous).long()
        depth_map_index = torch.clamp(depth_map_index, 0, N - 1)
        depth_map_index_expanded = depth_map_index.unsqueeze(1)  # [B,1,H,W]
        fused_image = torch.gather(x, 1, depth_map_index_expanded)
        return fused_image, depth_map_index

    @staticmethod
    def generate_soft_fused(x, probs):
        # x: [B,N,H,W], probs: [B,N,H,W]
        return torch.sum(x * probs, dim=1, keepdim=True)  # [B,1,H,W]

    def forward(self, image_stack, event_stack,
                dataset_name=None, scene_name=None):
        if image_stack.shape != event_stack.shape:
            raise ValueError("image_stack and event_stack must have same shape [B,N,H,W]")
        img = image_stack.float()
        evt = event_stack.float()
        B, N, H, W = img.shape

        stacked = torch.stack([img, evt], dim=2)  # [B,N,2,H,W]
        focus_maps, evt_depth_feats = self.feature_extraction(stacked)  # [B,N,Cf,H,W], [B,N,Ce,H,W]

        if (dataset_name is not None) and (scene_name is not None):
            self.visualize_feature_maps_ini(
                focus_maps,
                root_dir=self.vis_root,
                dataset_name=dataset_name,
                scene_name=scene_name
            )

        focus_maps_depth = self.layer_interaction(focus_maps, evt_depth_feats)  # [B,N,H,W]

        if (dataset_name is not None) and (scene_name is not None):
            self.visualize_feature_maps_final(
                focus_maps_depth,
                root_dir=self.vis_root,
                dataset_name=dataset_name,
                scene_name=scene_name
            )
        depth_logits = self.n_refine(focus_maps_depth)  # [B,N,H,W]
        guide0 = img.mean(dim=1, keepdim=True)  # [B,1,H,W]
        tau_map = self.tau_predictor(guide0)    # [B,1,H,W]
        depth_map, probs = self.depth_map_creation(
            depth_logits, N, tau_map=tau_map
        )  # [B,1,H,W], [B,N,H,W]
        soft_fused = self.generate_soft_fused(img, probs)  # [B,1,H,W]
        depth_map_refined = self.guided(soft_fused, depth_map).clamp(0.0, 1.0)
        conf = self.conf_est(probs)  # [B,1,H,W]
        depth_refined = self.diffuse(soft_fused, depth_map_refined, conf)
        fused_image, depth_map_index = self.generate_fused_image(img, depth_refined)

        if self.return_probs:
            return fused_image, depth_refined, depth_map_index, probs
        else:
            return fused_image, depth_refined, depth_map_index

    def visualize_feature_maps_ini(self, feature_maps,
                                   root_dir, dataset_name, scene_name):
        out_dir = os.path.join(root_dir, str(dataset_name), str(scene_name))
        os.makedirs(out_dir, exist_ok=True)

        B, N, C, H, W = feature_maps.shape
        for n in range(N):
            layer = feature_maps[0, n, 0].detach().cpu().numpy()
            layer = cv2.normalize(layer, None, 0, 255, cv2.NORM_MINMAX)
            layer_color = gray_to_colormap(layer)
            out_name = os.path.join(out_dir, f"layer_{n:02d}_color_ini.png")
            cv2.imwrite(out_name, cv2.cvtColor(layer_color, cv2.COLOR_RGB2BGR))

        print(f"[INI] Colorized feature maps for {N} layers saved to {out_dir}")

    def visualize_feature_maps_final(self, feature_maps,
                                     root_dir, dataset_name, scene_name):
        """
        Stage-2: feature_maps: [B, N, H, W]
        """
        out_dir = os.path.join(root_dir, str(dataset_name), str(scene_name))
        os.makedirs(out_dir, exist_ok=True)

        B, N, H, W = feature_maps.shape
        for n in range(N):
            layer = feature_maps[0, n].detach().cpu().numpy()
            layer = cv2.normalize(layer, None, 0, 255, cv2.NORM_MINMAX)
            layer_color = gray_to_colormap(layer)
            out_name = os.path.join(out_dir, f"layer_{n:02d}_color_final.png")
            cv2.imwrite(out_name, cv2.cvtColor(layer_color, cv2.COLOR_RGB2BGR))

        print(f"[FINAL] Colorized focus maps for {N} layers saved to {out_dir}")


if __name__ == "__main__":
    B, N, H, W = 1, 6, 128, 128
    img = torch.rand(B, N, H, W)
    evt = torch.rand(B, N, H, W)
    model = EventStackMFF(return_probs=True)
    out = model(img, evt)
    print("Output len:", len(out))
    for i, o in enumerate(out):
        print(f"out[{i}].shape =", o.shape)
