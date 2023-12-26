import math
import torch
import torchvision
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch import nn
import torch.nn.functional as F
from custom_models import *
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
from timm.models.cait import Cait
from timm.models.vision_transformer import VisionTransformer
from math import sqrt

        

def positionalencoding2d(D_, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D_ % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D_))
    P = torch.zeros(D_, H, W)
    # Each dimension use half of D
    D = D_ // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))


def freia_flow_head(c, n_feat):
    coder = Ff.SequenceINN(n_feat)
    print('NF coder:', n_feat)
    for k in range(c.coupling_blocks):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def freia_cflow_head(c, n_feat):
    n_cond = c.condition_vec
    # n_cond = n_feat//2
    coder = Ff.SequenceINN(n_feat)
    # add = n_feat//512
    print('CNF coder:', n_feat)
    # for k in range(c.coupling_blocks + add*2 - 2):
    for k in range(c.coupling_blocks):
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=False)
    return coder


def load_decoder_arch(c, dim_in):
    if   c.dec_arch == 'sanflow':
        decoder = freia_cflow_head(c, dim_in)
    else:
        raise NotImplementedError('{} is not supported NF!'.format(c.dec_arch))
    return decoder


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_activation_2(name):
    def hook(model, input, output):
        activation[name] = output
    return hook


def load_encoder_arch(c, L):
    # encoder pretrained on natural images:
    pool_cnt = 0
    pool_dims = list()
    pool_layers = ['layer'+str(i) for i in range(L)]
    if 'resnet' in c.enc_arch:
        if   c.enc_arch == 'resnet18':
            encoder = resnet18(pretrained=True, progress=True)
        elif c.enc_arch == 'resnet34':
            encoder = resnet34(pretrained=True, progress=True)
        elif c.enc_arch == 'resnet50':
            encoder = resnet50(pretrained=True, progress=True)
        elif c.enc_arch == 'resnext50_32x4d':
            encoder = resnext50_32x4d(pretrained=True, progress=True)
        elif c.enc_arch == 'wide_resnet50_2':
            encoder = wide_resnet50_2(pretrained=True, progress=True)
        elif c.enc_arch == 'wide_resnet101_2':
            encoder = wide_resnet101_2(pretrained=True, progress=True)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        #
        if L >= 4:
            encoder.layer1.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer1[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer1[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 3:
            encoder.layer2.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer2[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer2[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.layer3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer3[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer3[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.layer4.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer4[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer4[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
    elif 'vit' in c.enc_arch:
        if  c.enc_arch == 'vit_base_patch16_224':
            encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        elif  c.enc_arch == 'vit_base_patch16_384':
            encoder = timm.create_model('vit_base_patch16_384', pretrained=True)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        #
        if L >= 3:
            encoder.blocks[10].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.blocks[9].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.blocks[8].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
    elif 'cait' in c.enc_arch:
        encoder = timm.create_model("cait_m48_448", pretrained=True)
        # channels = [768]
        # scales = [16]
        #
        if L >= 3:
            encoder.blocks[47].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[47].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.blocks[31].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[15].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.blocks[15].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[31].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
    elif 'deit' in c.enc_arch:
        encoder = timm.create_model("deit_base_distilled_patch16_384", pretrained=True)
        # channels = [768]
        # scales = [16]
        #
        if L >= 3:
            encoder.blocks[10].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            # pool_dims.append(encoder.blocks[11].norm2.LayerNorm.out_features)
            pool_dims.append(encoder.blocks[10].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.blocks[6].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            # pool_dims.append(encoder.blocks[3].norm2.LayerNorm.out_features)
            pool_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.blocks[2].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            # pool_dims.append(encoder.blocks[7].norm2.LayerNorm.out_features)
            pool_dims.append(encoder.blocks[2].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
    elif 'efficient' in c.enc_arch:
        if 'b5' in c.enc_arch:
            encoder = timm.create_model(c.enc_arch, pretrained=True)
            blocks = [-2, -3, -5]
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        #
        if L >= 3:
            encoder.blocks[blocks[2]][-1].bn3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[blocks[2]][-1].bn3.num_features)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.blocks[blocks[1]][-1].bn3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[blocks[1]][-1].bn3.num_features)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.blocks[blocks[0]][-1].bn3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[blocks[0]][-1].bn3.num_features)
            pool_cnt = pool_cnt + 1
    elif 'mobile' in c.enc_arch:
        if  c.enc_arch == 'mobilenet_v3_small':
            encoder = mobilenet_v3_small(pretrained=True, progress=True).features
            blocks = [-2, -5, -10]
        elif  c.enc_arch == 'mobilenet_v3_large':
            encoder = mobilenet_v3_large(pretrained=True, progress=True).features
            blocks = [-2, -5, -11]
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        #
        if L >= 3:
            encoder[blocks[2]].block[-1][-3].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder[blocks[2]].block[-1][-3].out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder[blocks[1]].block[-1][-3].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder[blocks[1]].block[-1][-3].out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder[blocks[0]].block[-1][-3].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder[blocks[0]].block[-1][-3].out_channels)
            pool_cnt = pool_cnt + 1
    else:
        raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
    #
    return encoder, pool_layers, pool_dims


###############################################################################################
# # class DnCNN(nn.Module):
class SNet(nn.Module):
    # def __init__(self, channels=3, num_of_layers=17):
    def __init__(self, channels=3, num_of_layers=5):
        super(SNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        self.layer1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        self.last_layer = nn.Conv2d(in_channels=features, out_channels=2, kernel_size=kernel_size, padding=padding, bias=False)
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.layer1(x))
        residual = x
        out = self.dncnn(x)
        out = self.last_layer(out + residual)
        return out


class SNet_mean(nn.Module):
    # def __init__(self, channels=3, num_of_layers=17):
    def __init__(self, channels=3, num_of_layers=5):
        super(SNet_mean, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        # layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        # layers.append(nn.ReLU(inplace=True))
        self.layer1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.last_layer = nn.Conv2d(in_channels=features, out_channels=3, kernel_size=kernel_size, padding=padding, bias=False)
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.layer1(x))
        residual = x
        out = self.dncnn(x)
        out = self.last_layer(out + residual)
        return out
