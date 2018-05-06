from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

__all__ = ['SattentionNet', 'sattention']


def _make_conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1,
               bias=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    bn = nn.BatchNorm2d(out_planes)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(conv, bn, relu)



class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)

class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    pass

class SattentionNet(nn.Module):
    def __init__(self, num_features=128, seqlen=6, norm=True, spanum=3, pretrained=1):
        super(SattentionNet, self).__init__()

        self.atn_height = 8
        self.atn_width = 4
        self.pretrained = pretrained
        self.spanum = spanum
        self.seqlen = seqlen
        self.num_features = num_features
        self.norm = norm

        self.conv1 = nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, self.spanum, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.spanum)

        self.softmax = BottleSoftmax()

        self.feat = nn.Linear(2048, self.num_features)

        if not self.pretrained:
            self.reset_params()


    def forward(self, x):
        x = x.view( (-1,2048)+x.size()[-2:] )

        atn = x
        atn = self.conv1(atn)
        atn = self.bn1(atn)
        atn = self.relu1(atn)
        atn = self.conv2(atn)
        atn = self.bn2(atn)
        atn = attn.view(-1,self.spanum, self.atn_height*self.atn_width)
        atn = self.softmax(atn)

        # Diversity Regularization
        reg = atn

        # Multiple Spatial Attention
        atn = atn.view(atn.size(0), self.spanum, 1, self.atn_height, self.atn_width)
        atn = atn.expand(atn.size(0), self.spanum, 2048, self.atn_height, self.atn_width)
        x = x.view(x.size(0), 1, 2048, self.atn_height, self.atn_width)
        x = x.expand(x.size(0), self.spanum, 2048, self.atn_height, self.atn_width)

        x = x * atn
        x = x.view(-1, 2048, self.atn_height, self.atn_width)
        x = F.avg_pool2d(x, x.size()[2:])*x.size(2)*x.size(3)
        x = x.view(-1, 2048)

        x = self.feat(x)
        x = x.view(-1, self.spanum, self.num_features)

        if self.norm:
            x = x / x.norm(2, 2).view(-1,self.spanum,1).expand_as(x)
        x = x.view(-1, self.seqlen, self.spanum, self.num_features)

        return x, reg

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def sattention(**kwargs):
    return SattentionNet(**kwargs)
