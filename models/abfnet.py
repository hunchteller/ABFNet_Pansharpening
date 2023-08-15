import torch
import torch.nn.functional as F
from torch import nn

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None, pad_model=None):
        super(ConvBlock, self).__init__()

        self.pad_model = pad_model
        self.norm = norm
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(self.output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(self.output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

        if self.pad_model == None:
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding,
                                        bias=self.bias)
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0,
                                        bias=self.bias)

    def forward(self, x):
        out = x
        if self.pad_model is not None:
            out = self.padding(out)

        if self.norm is not None:
            out = self.bn(self.conv(out))
        else:
            out = self.conv(out)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class SRCModule(nn.Module):
    def __init__(self, channels, kernel=-1):
        super(SRCModule, self).__init__()

        mid_channels = channels // 4
        self.reduction_pan = nn.Conv2d(channels, mid_channels*mid_channels, 1)
        self.reduction_ms = nn.Conv2d(channels, mid_channels, 1)


        self.expand_ms = nn.Conv2d(mid_channels, channels, 1)
        self.mid_channels = mid_channels

    def forward(self, xpan, xms):
        """

        Args:
            xpan: bn, dim, h, w
            xms: bn,  dim
        Returns:

        """
        bn, c, h, w = xpan.shape
        kernel = self.reduction_pan(xpan).view(bn, self.mid_channels, self.mid_channels, h, w) # b c h w
        xms = self.reduction_ms(xms)

        d = torch.rsqrt((kernel ** 2).sum(dim=(2, 3, 4), keepdim=True) + 1e-10)
        kernel = kernel * d

        out = torch.einsum('n c h w, n c d h w -> n d h w', xms, kernel) #+ xms

        out = self.expand_ms(out)

        # 1 nb*channels, h, w
        return out


class SCModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SCModule, self).__init__()

        self.neck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU()
        )
        self.a_head = nn.Conv2d(in_channels, in_channels, 1)
        self.b_head = nn.Conv2d(in_channels, in_channels, 1)

        self.norm = nn.InstanceNorm2d(in_channels)

    def forward(self, xpan, xms):
        """

        Args:
            xpan: bn, dim, h, w
            xms: bn, ns, dim
        Returns:

        """
        nb, c, h, w = xpan.shape
        xpan = self.norm(xpan)
        out = self.neck(xms)
        gamma = self.a_head(out)
        bias = self.b_head(out)
        out = xpan * gamma + bias

        return out

class ABFNet(nn.Module):
    def __init__(self, dim=32, band=4):
        super(ABFNet, self).__init__()

        dims = [dim] * 4

        self.PanModule = nn.ModuleList()

        self.MSModule = nn.ModuleList()

        self.SpectralModule = nn.ModuleList()
        self.SpatialModule = nn.ModuleList()

        for i, dim in enumerate(dims):
            if i == 0:
                self.PanModule.append(ConvBlock(1, dim, 3, 1, 1))
                self.MSModule.append(ConvBlock(band, dim, 3, 1, 1))
            else:
                self.PanModule.append(ConvBlock(dim, dim, 3, 1, 1))
                self.MSModule.append(ConvBlock(dim, dim, 3, 1, 1))
            self.SpectralModule.append(SCModule(dim, dim))
            self.SpatialModule.append(SRCModule(dim))

        self.out = nn.Conv2d(dims[-1]*2, band, 1)



    def forward(self, X_MS, X_PAN):

        nb, c, h, w = X_PAN.shape
        X_MS = F.interpolate(X_MS, size=(h, w), mode='bicubic')

        xms = X_MS
        xpan = X_PAN

        for pan_cb, ms_cb, sc_module, src_module in zip(self.PanModule, self.MSModule, self.SpectralModule, self.SpatialModule):
            xms_t = ms_cb(xms)
            xpan_t = pan_cb(xpan)
            xpan = sc_module(xpan_t, xms_t)
            xms = src_module(xpan, xms_t)

        out = torch.cat((xms, xpan), 1)

        pr = self.out(out) + X_MS        
        return pr


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    net = ABFNet()
    lms = torch.randn(1, 4, 64, 64)
    pan = torch.randn(1, 1, 256, 256)
    flops = FlopCountAnalysis(net, (lms, pan))
    print(flop_count_table(flops))