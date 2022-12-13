import torch
from torch import nn
import functools

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvBlock3D(nn.Module):

    def __init__(self, input_channel, out_channel,norm_type = "BN",normalized_shape=None):
        super().__init__()
        if norm_type == "BN":
            self.norm1 = nn.BatchNorm3d(num_features=out_channel)
            self.norm2 = nn.BatchNorm3d(num_features=out_channel)
        elif norm_type == "GN":
            self.norm1 = nn.GroupNorm(num_groups=8,num_channels=out_channel)
            self.norm2 = nn.GroupNorm(num_groups=8,num_channels=out_channel)
        elif norm_type == "LN":
            self.norm1 = nn.LayerNorm(normalized_shape=(out_channel,)+normalized_shape)
            self.norm2 = nn.LayerNorm(normalized_shape=(out_channel,)+normalized_shape)
        else:
            self.norm1 = nn.InstanceNorm3d(num_features=out_channel)
            self.norm2 = nn.InstanceNorm3d(num_features=out_channel)
        self.conv1 = nn.Conv3d(input_channel, out_channel, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm3d(out_channel)
        self.leak_relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm3d(out_channel)
        self.leak_relu2 = nn.LeakyReLU(inplace=True)


    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.leak_relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.leak_relu2(out)
        return out


class Upsample_3D(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample_3D, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

class UpBlock_3D(nn.Module):
    def __init__(self, input_channel, out_channel,norm_type="BN"):
        super().__init__()
        # if norm_type == "BN":
        #     norm_func = nn.BatchNorm3d(num_features=out_channel)
        # else:
        #     norm_func = nn.InstanceNorm3d(num_features=out_channel)
        up_block = [Upsample_3D(scale_factor=2, mode='bilinear'),
                    nn.Conv3d(in_channels=input_channel, out_channels=out_channel,
                              kernel_size=3,
                              padding=1,
                              bias=False),
                    nn.BatchNorm3d(num_features=out_channel),
                    nn.LeakyReLU(inplace=True)
                    ]
        self.up_block = nn.Sequential(*up_block)

    def forward(self, x):
        return self.up_block(x)

class UpBlock_3D_adaptnorm(nn.Module):
    def __init__(self, input_channel, out_channel,norm_type="BN"):
        super().__init__()
        if norm_type == "BN":
            norm_func = nn.BatchNorm3d(num_features=out_channel)
        else:
            norm_func = nn.InstanceNorm3d(num_features=out_channel)
        up_block = [Upsample_3D(scale_factor=2, mode='bilinear'),
                    nn.Conv3d(in_channels=input_channel, out_channels=out_channel,
                              kernel_size=3,
                              padding=1,
                              bias=False),
                    norm_func,
                    nn.LeakyReLU(inplace=True)
                    ]
        self.up_block = nn.Sequential(*up_block)

    def forward(self, x):
        return self.up_block(x)

class ModalitySpecificBatchnorm_3D(nn.Module):
    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()

        self.bns = nn.ModuleList([nn.BatchNorm3d(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                                 track_running_stats=track_running_stats) \
                                  for _ in range(num_classes)])

    def forward(self, x, domain_label):
        # print(domain_label)
        bn = self.bns[domain_label[0]]
        return bn(x), domain_label

class ModalitySpeciUpBlock_3D(nn.Module):
    def __init__(self, input_channel, out_channel,num_domains):
        super().__init__()
        self.up = Upsample_3D(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv3d(in_channels=input_channel, out_channels=out_channel,
                              kernel_size=3,
                              padding=1,
                              bias=False)
        self.bn = ModalitySpecificBatchnorm_3D(out_channel,num_domains)
        self.relu = nn.LeakyReLU(inplace=True)


    def forward(self, x,domain_label):
        out = self.up(x)
        out = self.conv(out)
        out,_ = self.bn(out,domain_label)
        out = self.relu(out)
        return out

class SpeciModalConvBlock_3D(nn.Module):

    def __init__(self, input_channel, out_channel, num_domains):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = ModalitySpecificBatchnorm_3D(out_channel,num_domains)
        self.leak_relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = ModalitySpecificBatchnorm_3D(out_channel, num_domains)
        self.leak_relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x,domain_label):
        out = self.conv1(x)
        out,_ = self.bn1(out,domain_label)
        out = self.leak_relu1(out)
        out = self.conv2(out)
        out,_ = self.bn2(out,domain_label)
        out = self.leak_relu2(out)
        return out

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class UpBlock(nn.Module):
    def __init__(self, input_channel, out_channel):
        super().__init__()
        up_block = [Upsample(scale_factor=2, mode='bilinear'),
                    nn.Conv2d(in_channels=input_channel, out_channels=out_channel,
                              kernel_size=3,
                              padding=1,
                              bias=False),
                    nn.BatchNorm2d(num_features=out_channel),
                    nn.LeakyReLU(inplace=True)
                    ]
        self.up_block = nn.Sequential(*up_block)

    def forward(self, x):
        return self.up_block(x)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)