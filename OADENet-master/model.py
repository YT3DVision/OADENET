import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import time

class Net(nn.Module):
    def __init__(self, angRes):
        super(Net, self).__init__()
        self.num_cascade = 2
        mindisp = -8
        maxdisp = 8
        self.angRes = angRes
        self.maxdisp = maxdisp
        self.init_feature1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0), bias=False),
            nn.BatchNorm3d(16),
            ResB1(16), ResB1(16), ResB1(16), ResB1(16),
            nn.Conv3d(16, 8, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0), bias=False),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.init_feature2 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            ResB2(16), ResB2(16), ResB2(16), ResB2(16),
            nn.Conv3d(16, 8, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.init_feature3 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(16),
            ResB3(16), ResB3(16), ResB3(16), ResB3(16),
            nn.Conv3d(16, 8, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.eaf1 = nn.Sequential(
            nn.Conv2d(3*8, 6, kernel_size=1, stride=1, groups=3),
            nn.BatchNorm2d(6),

        )
        self.eaf2 = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=1, stride=1, groups=3),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.lastconv = nn.Conv3d(8, 8, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)
        self.build_cost = BuildCost(8, 512, angRes, mindisp, maxdisp)
        self.aggregate = Aggregate(512, 64, mindisp, maxdisp) #modify 160->64
        # self.avg_pool = nn.AdaptiveAvgPool3d((1, 32, 32))
        self.mask_refine = nn.Sequential(
            nn.Conv2d(81, 25, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(25),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(25, 81, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(81),
        )


    def forward(self, x, dispGT=None):
        lf = rearrange(x, 'b c (a1 h) (a2 w) -> b c a1 a2 h w', a1=self.angRes, a2=self.angRes)
        x = rearrange(x, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=self.angRes, a2=self.angRes)
        b, c, _, h, w = x.shape #x(1,1,81,32,32)
        # feat1 = self.init_feature1(x)  #feat(1,8,81,32,32)
        # feat2 = self.init_feature2(x)  #feat(1,8,81,32,32)
        # feat3 = self.init_feature3(x)  #feat(1,8,81,32,32)

        feat1 = rearrange(self.init_feature1(x), 'b c (u v) h w -> b c (h u) (w v)', u=self.angRes, v=self.angRes)
        feat2 = rearrange(self.init_feature2(x), 'b c (u v) h w -> b c (h u) (w v)', u=self.angRes, v=self.angRes)
        feat3 = rearrange(self.init_feature3(x), 'b c (u v) h w -> b c (h u) (w v)', u=self.angRes, v=self.angRes)

        feat = [feat1, feat2, feat3]

        # Spatial Attention Fusion ours
        attn = self.eaf1(torch.cat(feat, 1))  # feature(b,c,hu,wv)input MPI attn shape(b,3,hu,wv)
        attn = self.eaf2(attn)
        attn = attn.softmax(dim=1)
        final_feature = (attn.unsqueeze(2) * torch.stack(feat, 1)).sum(
            dim=1)  # attn.unsqueeze(2) shape(b,3,1,ha,wa) torch.stack(buffer, 1) shape(b,3,c,ha,wa) final_feature(b,c,ha,wa)

        final_feature = rearrange(final_feature, 'b c (h u) (w v)->b c (u v) h w', u=self.angRes, v=self.angRes)
        if dispGT is not None:
            mask_gt = Generate_mask(lf, dispGT) #(1,81,32,32) (b,a*a,h,w)  lf shape(b,c,u,v,h,w) dispGT((1,1,32,32))
            # mask_gt = mask_gt[:, 40, :, :].unsqueeze(1) #中心视点的maskgt全是1，也就是默认无遮挡，其他视点多少有些遮挡，所以不是1,大概是0.9或者0.8左右
        mask = torch.ones(1, self.angRes ** 2, h, w).to(x.device) #(1,81,32,32) (b,a*a,h,w) #mask的值全为1，即无遮挡
        cost = self.build_cost(final_feature, mask) #cost(1,512,9,32,32) (b,c,depthrange,h,w)，build_cost可用mask信息，也可以不用，此处不用
        disp = self.aggregate(cost) #disp(1,1,32,32) (b,c,h,w)
        mask = Generate_mask(lf, disp) #(1,81,32,32) (b,a*a,h,w)
        # cost = self.build_cost(final_feature, mask) #cost(1,512,9,32,32) (b,c,depthrange,h,w)
        # disp = self.aggregate(cost) #disp(1,1,32,32) (b,c,h,w)
        mask_ref = self.mask_refine(mask) #ours

        if dispGT is not None:
            return disp, mask_gt, mask_ref
        else:
            return disp, mask_ref

#
class BuildCost(nn.Module):
    def __init__(self, channel_in, channel_out, angRes, mindisp, maxdisp):
        super(BuildCost, self).__init__()
        self.oacc = ModulateConv2d(channel_in, channel_out, kernel_size=angRes, stride=1, bias=False)
        self.angRes = angRes
        self.mindisp = mindisp
        self.maxdisp = maxdisp
        self.channel_att = channel_out
        self.channel_in = channel_in
        self.attn=DynamicWeightTransformerAttention(512)
    def forward(self, x, mask): #x shape(1,8,81,32,32) ,mask shape(1,81,32,32)
        b, c, aa, h, w = x.shape
        x = rearrange(x, 'b c (a1 a2) h w -> (b a1 a2) c h w', a1=self.angRes, a2=self.angRes)
        bdr = (self.angRes // 2) * self.maxdisp
        pad = nn.ZeroPad2d((bdr, bdr, bdr, bdr)) #depend on left,right,top,bottom padding zero value
        x_pad = pad(x)
        x_pad = rearrange(x_pad, '(b a1 a2) c h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        h_pad, w_pad = h + 2 * bdr, w + 2 * bdr
        mask_avg = torch.mean(mask, dim=1) #we can learning it
        cost = []
        for d in range(self.mindisp, self.maxdisp + 1):
            dila = [h_pad - d, w_pad - d]
            self.oacc.dilation = dila
            crop = (self.angRes // 2) * (d - self.mindisp)
            if d == self.mindisp:
                feat = x_pad
            else:
                feat = x_pad[:, :, crop: -crop, crop: -crop]
                #输出通道改小一点，current_cost的输出为(1,512,32,32),可以跟个局部attention注意力，输出仍需要是（1,512,32,32）
            current_cost = self.oacc(feat, mask) #feat shape may be not same,because of crop operation
            current_cost=self.attn(current_cost)
            cost.append(current_cost / mask_avg.unsqueeze(1).repeat(1, current_cost.shape[1], 1, 1))
        cost = torch.stack(cost, dim=2)

        return cost



class DynamicWeightTransformerAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, reduction=4):
        """
        基于 Transformer 的自注意力模块，带有动态权重调整
        Args:
            in_channels (int): 输入通道数
            num_heads (int): 注意力头的数量
            reduction (int): 通道压缩比，用于降低计算复杂度
        """
        super(DynamicWeightTransformerAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.reduced_channels = in_channels // reduction
        self.head_dim = self.reduced_channels // num_heads
        assert self.head_dim * num_heads == self.reduced_channels, "通道数必须能被头数整除"

        # 通道降维
        self.qkv_proj = nn.Linear(in_channels, 3 * self.reduced_channels, bias=False)
        self.output_proj = nn.Linear(self.reduced_channels, in_channels, bias=False)

        # 注意力归一化
        self.softmax = nn.Softmax(dim=-1)
        self.scale = self.head_dim ** -0.5  # 缩放因子

        # LayerNorm 和激活函数
        self.norm = nn.LayerNorm(in_channels)
        self.activation = nn.Sigmoid()

        # 动态权重生成网络
        self.dynamic_weight_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # 预处理：Flatten H, W -> N
        x_flatten = x.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        x_flatten = self.norm(x_flatten)  # LayerNorm

        # 计算 Q, K, V
        qkv = self.qkv_proj(x_flatten)  # (B, N, 3 * reduced_channels)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)  # (B, N, reduced_channels)

        # 拆分多头
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 注意力计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn_weights = self.softmax(attn_weights)

        # 动态权重生成：通过输入特征生成动态缩放因子
        dynamic_weights = self.dynamic_weight_net(x)  # (B, 1, H, W)
        dynamic_weights = dynamic_weights.view(B, 1, N).expand(-1, self.num_heads, N)  # (B, num_heads, N)

        # 加权注意力
        attn_weights = attn_weights * dynamic_weights.unsqueeze(-1)  # (B, num_heads, N, N)

        # 加权值
        attention = torch.matmul(attn_weights, v)  # (B, num_heads, N, head_dim)

        # 合并头
        attention = attention.permute(0, 2, 1, 3).reshape(B, N, self.reduced_channels)  # (B, N, reduced_channels)

        # 恢复通道数
        out = self.output_proj(attention)  # (B, N, C)

        # 恢复形状：Unflatten N -> H, W
        out = out.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)

        # 残差连接
        return self.activation(self.gamma*out + x)


class Aggregate(nn.Module):
    def __init__(self, inC, channel, mindisp, maxdisp): #inC 512,channel 64
        super(Aggregate, self).__init__()
        self.sq = nn.Sequential(
            nn.Conv3d(inC, channel, 1, 1, 0, bias=False), nn.BatchNorm3d(channel), nn.LeakyReLU(0.1, inplace=True))
        self.Conv1 = nn.Sequential(
            nn.Conv3d(channel, channel, 3, 1, 1, bias=False), nn.BatchNorm3d(channel), nn.LeakyReLU(0.1, inplace=True))
        self.Conv2 = nn.Sequential(
            nn.Conv3d(channel, channel, 3, 1, 1, bias=False), nn.BatchNorm3d(channel), nn.LeakyReLU(0.1, inplace=True))
        self.Resb1 = ResB3D(channel)
        self.Resb2 = ResB3D(channel)
        self.Conv3 = nn.Sequential(
            nn.Conv3d(channel, channel, 3, 1, 1, bias=False), nn.BatchNorm3d(channel), nn.LeakyReLU(0.1, inplace=True))
        self.Conv4 = nn.Conv3d(channel, 1, 3, 1, 1, bias=False)
        self.softmax = nn.Softmax(1)
        self.mindisp = mindisp
        self.maxdisp = maxdisp
        # self.refine = Refine_Block(dim=9) #ours


    def forward(self, psv):
        buffer = self.sq(psv)
        buffer = self.Conv1(buffer)
        buffer = self.Conv2(buffer)
        buffer = self.Resb1(buffer)
        buffer = self.Resb2(buffer)
        buffer = self.Conv3(buffer)
        score = self.Conv4(buffer)
        attmap = self.softmax(score.squeeze(1))
        # attmap = self.refine(attmap)
        temp = torch.zeros(attmap.shape).to(attmap.device)
        for d in range(self.maxdisp - self.mindisp + 1):
            temp[:, d, :, :] = attmap[:, d, :, :] * (self.mindisp + d)
        disp = torch.sum(temp, dim=1, keepdim=True)
        return disp

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# class Refine_Block(nn.Module):
#     def __init__(self, dim, kernel_size = 1):
#         super(Refine_Block, self).__init__()
#
#
#         self.norm = LayerNorm2d(dim)
#         # self.input_conv = nn.Conv2d(dim , dim, kernel_size=1, bias=True)
#         self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, padding= 0, groups = 1, bias=True)
#         self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding= 1, groups = dim, bias=True)
#         self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding= 2, groups = dim, bias=True)
#         self.gelu = nn.GELU()
#         self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
#
#
#     def forward(self , x):#(b,d,h,w)
#         fea = self.norm(x)
#         # fea = self.input_conv(fea)
#         fea1 = self.conv1x1(fea)
#         fea2 = self.conv3x3(fea)
#         fea3 = self.conv5x5(fea)
#
#         fea = fea1 * (fea2 + fea3)
#         fea = self.gelu(fea)
#         out = self.out_conv(fea)
#         out += x
#         return out


class ResB1(nn.Module):
    def __init__(self, feaC):
        super(ResB1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(feaC, feaC, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0)),
            nn.BatchNorm3d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(feaC, feaC, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0)),
            nn.BatchNorm3d(feaC))

    def forward(self, x):
        out = self.conv(x)
        return x + out

class ResB2(nn.Module):
    def __init__(self, feaC):
        super(ResB2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(feaC, feaC, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(feaC, feaC, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(feaC))

    def forward(self, x):
        out = self.conv(x)
        return x + out

class ResB3(nn.Module):
    def __init__(self, feaC):
        super(ResB3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(feaC, feaC, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(feaC, feaC, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(feaC))

    def forward(self, x):
        out = self.conv(x)
        return x + out

class ResB3D(nn.Module):
    def __init__(self, channels):
        super(ResB3D, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False), #using D,H,W to conv 3D
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(channels))
        self.calayer = CALayer(channels, 17)

    def forward(self, x):
        buffer = self.body(x) #just conv,not important
        return self.calayer(buffer) + x

class CALayer(nn.Module):
    def __init__(self, channel, num_views):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((num_views, 1, 1))
        self.conv_du = nn.Sequential(  #MLP operation
                nn.Conv3d(channel, channel // 16, 1, 1, 0, bias=True),
                nn.BatchNorm3d(channel // 16),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv3d(channel // 16, channel, 1, 1, 0, bias=True),
                nn.BatchNorm3d(channel),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def Generate_mask(lf, disp):
    b, c, angRes, _, h, w = lf.shape
    x_base = torch.linspace(0, 1, w).repeat(1, h, 1).to(lf.device)
    y_base = torch.linspace(0, 1, h).repeat(1, w, 1).transpose(1, 2).to(lf.device)
    center = (angRes - 1) // 2
    img_ref = lf[:, :, center, center, :, :]
    img_res = []
    img_depth = []

    for u in range(angRes):
        for v in range(angRes):
            img = lf[:, :, u, v, :, :]
            if (u == center) & (v == center):
                img_warped = img
            else:
                du, dv = u - center, v - center
                img_warped = warp(img, -disp, du, dv, x_base, y_base)
            img_res.append(abs((img_warped - img_ref)))
            img_depth.append(abs((img_warped)))
    mask = torch.cat(img_res, dim=1)
    out = (1 - mask)**2
    return out


def warp(img, disp, du, dv, x_base, y_base):

    b, _, h, w = img.size()
    x_shifts = dv * disp[:, 0, :, :] / w
    y_shifts = du * disp[:, 0, :, :] / h
    flow_field = torch.stack((x_base + x_shifts, y_base + y_shifts), dim=3)
    img_warped = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')
    return img_warped


class ModulateConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, dilation=1, bias=False):
        super(ModulateConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.flatten = nn.Unfold(kernel_size=1, stride=1, dilation=1, padding=0)
        self.fuse = nn.Conv2d(channel_in * kernel_size * kernel_size, channel_out,
                              kernel_size=1, stride=1, padding=0, bias=bias, groups=channel_in)

    def forward(self, x, mask): #(1,8,576,576) mask shape(1,81,32,32)
        mask_flatten = self.flatten(mask) #(1,81,1024)
        Unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation) #an conv2d
        x_unfold = Unfold(x) #(1,648,1024)
        x_unfold_modulated = x_unfold * mask_flatten.repeat(1, x.shape[1], 1) #(1,648,1024)
        Fold = nn.Fold(output_size=(mask.shape[2], mask.shape[3]), kernel_size=1, stride=1) #output_size have same shape of mask
        x_modulated = Fold(x_unfold_modulated) #(1,648,32,32)
        out = self.fuse(x_modulated) #(1,512,32,32)
        return out


if __name__ == "__main__":
    angRes = 9
    net = Net(angRes).cuda()
    from thop import profile
    input = torch.randn(1, 1, 32 * angRes, 32 * angRes).cuda()
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))