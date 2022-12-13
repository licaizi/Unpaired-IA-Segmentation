from models.Multi_Modal_Seg.network2D import Upsample,UpBlock,ModalitySpecificBatchnorm,ResidualBlock,ConvBlock,SpeciModalConvBlock,ModalitySpeciUpBlock
from models.Multi_Modal_Seg.model_utils import Upsample_3D,ModalitySpeciUpBlock_3D,SpeciModalConvBlock_3D
from torch import nn
import torch
from torch.nn import init

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
                    nn.LeakyReLU(inplace=False)
                    ]
        self.up_block = nn.Sequential(*up_block)

    def forward(self, x):
        return self.up_block(x)

class ConvBlock3D(nn.Module):

    def __init__(self, input_channel, out_channel,norm_type = "IN"):
        super().__init__()
        if norm_type == "BN":
            self.bn1 = nn.BatchNorm3d(num_features=out_channel)
            self.bn2 = nn.BatchNorm3d(num_features=out_channel)
        else:
            self.bn1 = nn.InstanceNorm3d(num_features=out_channel)
            self.bn2 = nn.InstanceNorm3d(num_features=out_channel)
        self.conv1 = nn.Conv3d(input_channel, out_channel, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm3d(out_channel)
        self.leak_relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm3d(out_channel)
        self.leak_relu2 = nn.LeakyReLU(inplace=True)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leak_relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leak_relu2(out)
        return out

class Encoder(nn.Module):
    def __init__(self,input_channel,init_channel,norm_type = "IN"):
        super(Encoder, self).__init__()
        self.encoder1 = ConvBlock3D(input_channel, init_channel,norm_type)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock3D(init_channel, init_channel * 2,norm_type)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = ConvBlock3D(init_channel * 2, init_channel * 4,norm_type)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock3D(init_channel * 4, init_channel * 8,norm_type)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        if norm_type == 'IN':
            self.botlle_neck = nn.Sequential(
                nn.Conv3d(init_channel*8,init_channel*16, kernel_size=3, padding=1),
                nn.InstanceNorm3d(init_channel*16),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.botlle_neck = nn.Sequential(
                nn.Conv3d(init_channel * 8, init_channel * 16, kernel_size=3, padding=1),
                nn.BatchNorm3d(init_channel * 16),
                nn.LeakyReLU(inplace=True)
            )


    def forward(self,x):
        enc1 = self.encoder1(x)  # 256
        enc2 = self.encoder2(self.pool1(enc1))  # 128
        enc3 = self.encoder3(self.pool2(enc2))  # 64
        enc4 = self.encoder4(self.pool3(enc3))  # 32

        bottleneck = self.botlle_neck(self.pool4(enc4))
        return bottleneck

class LatentEncoder(nn.Module):
    def __init__(self,input_channel, out_channel):
        super(LatentEncoder, self).__init__()
        self.conv1 = nn.Conv3d(input_channel, out_channel, kernel_size=3, padding=1)
        self.In1 = nn.InstanceNorm3d(out_channel)
        self.leak_relu1 = nn.LeakyReLU(inplace=False)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1)
        self.In2 = nn.InstanceNorm3d(out_channel)
        self.leak_relu2 = nn.LeakyReLU(inplace=False)
        self.leak_relu3 = nn.LeakyReLU(inplace=False)

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.In1(out)
        out = self.leak_relu1(out)
        out = self.conv2(out)
        out = self.In2(out)
        out = self.leak_relu2(out)

        out = out + residual
        out = self.leak_relu3(out)
        return out


class Decoder(nn.Module):
    def __init__(self,init_channel,num_class,upsample=False,norm_type = "IN"):
        super(Decoder, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upconv4 = UpBlock_3D(init_channel * 16,init_channel *8)
        else:
            self.upconv4 = nn.ConvTranspose3d(init_channel * 16,init_channel *8,kernel_size=2,stride=2)
        self.decoder4 = ConvBlock3D(init_channel * 8,init_channel * 8)
        if upsample:
            self.upconv3 = UpBlock_3D(init_channel*8,init_channel*4)
        else:
            self.upconv3 = nn.ConvTranspose3d(init_channel*8,init_channel*4,kernel_size=2,stride=2)
        self.decoder3 = ConvBlock3D(init_channel*4,init_channel*4)
        if upsample:
            self.upconv2 = UpBlock_3D(init_channel*4,init_channel*2)
        else:
            self.upconv2 = nn.ConvTranspose3d(init_channel*4,init_channel*2,kernel_size=2,stride=2)
        self.decoder2 = ConvBlock3D(init_channel*2,init_channel*2)
        if upsample:
            self.upconv1 = UpBlock_3D(init_channel*2,init_channel)
        else:
            self.upconv1 = nn.ConvTranspose3d(init_channel*2,init_channel,kernel_size=2,stride=2)
        self.decoder1 = ConvBlock3D(init_channel,init_channel,norm_type)
        self.conv = nn.Conv3d(init_channel,num_class,kernel_size=1)

    def forward(self,x):
        if self.upsample:
            dec4 = self.upconv4(x)
        else:
            dec4 = self.upconv4(x)  # 32
        # print('dec4',dec4.shape,enc4.shape)
        dec4 = self.decoder4(dec4)
        # print('dec4',dec4.shape)
        if self.upsample:
            dec3 = self.upconv3(dec4)
        else:
            dec3 = self.upconv3(dec4)  # 64

        dec3 = self.decoder3(dec3)
        if self.upsample:
            dec2 = self.upconv2(dec3)
        else:
            dec2 = self.upconv2(dec3)  # 128
        dec2 = self.decoder2(dec2)
        # print('dec2:',dec2.shape)
        if self.upsample:
            dec1 = self.upconv1(dec2)
        else:
            dec1 = self.upconv1(dec2)  # 256
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)


class Unet_3D(nn.Module):

    def __init__(self,input_channel,init_channel,num_class=2,upsample=False):
        super().__init__()
        self.upsample = upsample
        self.encoder1 = ConvBlock3D(input_channel,init_channel)
        self.pool1 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.encoder2 = ConvBlock3D(init_channel,init_channel*2)
        self.pool2 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.encoder3 = ConvBlock3D(init_channel*2, init_channel * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock3D(init_channel*4, init_channel * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.botlle_neck = ConvBlock3D(init_channel*8,init_channel*16)

        if upsample:
            self.upconv4 = UpBlock_3D(init_channel * 16,init_channel *8)
        else:
            self.upconv4 = nn.ConvTranspose3d(init_channel * 16,init_channel *8,kernel_size=2,stride=2)
        self.decoder4 = ConvBlock3D(init_channel * 8 *2,init_channel * 8)
        if upsample:
            self.upconv3 = UpBlock_3D(init_channel*8,init_channel*4)
        else:
            self.upconv3 = nn.ConvTranspose3d(init_channel*8,init_channel*4,kernel_size=2,stride=2)
        self.decoder3 = ConvBlock3D(init_channel*4*2,init_channel*4)
        if upsample:
            self.upconv2 = UpBlock_3D(init_channel*4,init_channel*2)
        else:
            self.upconv2 = nn.ConvTranspose3d(init_channel*4,init_channel*2,kernel_size=2,stride=2)
        self.decoder2 = ConvBlock3D(init_channel*2*2,init_channel*2)
        if upsample:
            self.upconv1 = UpBlock_3D(init_channel*2,init_channel)
        else:
            self.upconv1 = nn.ConvTranspose3d(init_channel*2,init_channel,kernel_size=2,stride=2)
        self.decoder1 = ConvBlock3D(init_channel*2,init_channel)
        self.conv = nn.Conv3d(init_channel,num_class,kernel_size=1)

    def forward(self,x,return_latent=False,return_feat=False):
        enc1 = self.encoder1(x)  # 256
        # print('enc1：',enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1))  # 128
        enc3 = self.encoder3(self.pool2(enc2))  # 64
        enc4 = self.encoder4(self.pool3(enc3))  # 32

        bottleneck = self.botlle_neck(self.pool4(enc4))  # 16
        # print('bb',bottleneck.shape)
        if self.upsample:
            dec4 = self.upconv4(bottleneck)
        else:
            dec4 = self.upconv4(bottleneck)  # 32
        # print('dec4',dec4.shape,enc4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        # print('dec4',dec4.shape)
        if self.upsample:
            dec3 = self.upconv3(dec4)
        else:
            dec3 = self.upconv3(dec4)  # 64

        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        if self.upsample:
            dec2 = self.upconv2(dec3)
        else:
            dec2 = self.upconv2(dec3)  # 128
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        # print('dec2:',dec2.shape)
        if self.upsample:
            dec1 = self.upconv1(dec2)
        else:
            dec1 = self.upconv1(dec2)  # 256
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        # print('dec1:',torch.sum(torch.isnan(dec1)))
        if return_feat:
            return self.conv(dec1),dec1
        if return_latent:
            return self.conv(dec1),dec2
        return self.conv(dec1)

class MatricLayer(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(MatricLayer, self).__init__()
        self.avgpool = nn.AvgPool3d(kernel_size=32,stride=32)
        self.mlp = nn.ModuleList([
            nn.Linear(in_dim,out_dim),
            nn.ReLU(),
            nn.Linear(out_dim,out_dim)
        ])

    def normolize(self,x):
        norm_x = torch.pow(torch.sum(torch.pow(x,2),dim=2,keepdim=True),1/2)
        x = torch.div(x,norm_x)
        return x

    def forward(self,x):
        pooled_x = self.avgpool(x)
        pooled_x = pooled_x.flatten(2)
        lineared_x = pooled_x
        for module in self.mlp:
            lineared_x = module(lineared_x)
        # print(pooled_x.shape)
        x = self.normolize(lineared_x)

        return x

class ContrastUnet_3D(nn.Module):

    def __init__(self,input_channel,init_channel,num_class=2,upsample=False):
        super().__init__()
        self.upsample = upsample
        self.encoder1 = ConvBlock3D(input_channel,init_channel)
        self.pool1 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.encoder2 = ConvBlock3D(init_channel,init_channel*2)
        self.pool2 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.encoder3 = ConvBlock3D(init_channel*2, init_channel * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock3D(init_channel*4, init_channel * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.botlle_neck = ConvBlock3D(init_channel*8,init_channel*16)

        if upsample:
            self.upconv4 = UpBlock_3D(init_channel * 16,init_channel *8)
        else:
            self.upconv4 = nn.ConvTranspose3d(init_channel * 16,init_channel *8,kernel_size=2,stride=2)
        self.decoder4 = ConvBlock3D(init_channel * 8 *2,init_channel * 8)
        if upsample:
            self.upconv3 = UpBlock_3D(init_channel*8,init_channel*4)
        else:
            self.upconv3 = nn.ConvTranspose3d(init_channel*8,init_channel*4,kernel_size=2,stride=2)
        self.decoder3 = ConvBlock3D(init_channel*4*2,init_channel*4)
        if upsample:
            self.upconv2 = UpBlock_3D(init_channel*4,init_channel*2)
        else:
            self.upconv2 = nn.ConvTranspose3d(init_channel*4,init_channel*2,kernel_size=2,stride=2)
        self.decoder2 = ConvBlock3D(init_channel*2*2,init_channel*2)
        if upsample:
            self.upconv1 = UpBlock_3D(init_channel*2,init_channel)
        else:
            self.upconv1 = nn.ConvTranspose3d(init_channel*2,init_channel,kernel_size=2,stride=2)
        self.decoder1 = ConvBlock3D(init_channel*2,init_channel)
        self.conv = nn.Conv3d(init_channel,num_class,kernel_size=1)
        self.contrast_model = MatricLayer(64,32)



    def forward(self,x,return_latent=False,return_feat=False):
        enc1 = self.encoder1(x)  # 256
        # print('enc1：',enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1))  # 128
        enc3 = self.encoder3(self.pool2(enc2))  # 64
        enc4 = self.encoder4(self.pool3(enc3))  # 32

        bottleneck = self.botlle_neck(self.pool4(enc4))  # 16
        # print('bb',bottleneck.shape)
        if self.upsample:
            dec4 = self.upconv4(bottleneck)
        else:
            dec4 = self.upconv4(bottleneck)  # 32
        # print('dec4',dec4.shape,enc4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        # print('dec4',dec4.shape)
        if self.upsample:
            dec3 = self.upconv3(dec4)
        else:
            dec3 = self.upconv3(dec4)  # 64

        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        if self.upsample:
            dec2 = self.upconv2(dec3)
        else:
            dec2 = self.upconv2(dec3)  # 128
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        # print('dec2:',dec2.shape)
        if self.upsample:
            dec1 = self.upconv1(dec2)
        else:
            dec1 = self.upconv1(dec2)  # 256
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        contrast_out = self.contrast_model(dec1)
        # print(contrast_out.shape)
        if return_feat:
            return self.conv(dec1),contrast_out
        if return_latent:
            return self.conv(dec1),dec2
        return self.conv(dec1)
# model = ContrastUnet_3D(1,16,2,upsample=False).cuda()
# data = torch.rand(1,1,128,128,128).cuda()
# encoder = Encoder(1,16).cuda()
# latentEnc = LatentEncoder(256,256).cuda()
# decoder = Decoder(16,2).cuda()
# out = encoder(data)
# out2 = latentEnc(out)
# out3 = decoder(out2)
# print(out.shape,out2.shape,out3.shape)
# out,feat = model(data,return_feat=True)
# print('feat',feat.shape)
# # out = model(data,[0])
# out,latent = model(data,True)
# print(out.shape,latent.shape)
# for layer,param in model.state_dict().items():
#     print(layer)
# print(out.shape)
# print(model)

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            print('......')
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
# print(model.__class__.__name__)
# init_weights(model)
