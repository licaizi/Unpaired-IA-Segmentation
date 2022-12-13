from models.Multi_Modal_Seg.network2D import Upsample,UpBlock,ModalitySpecificBatchnorm,ResidualBlock,ConvBlock,SpeciModalConvBlock,ModalitySpeciUpBlock
from models.Multi_Modal_Seg.model_utils import UpBlock_3D_adaptnorm,UpBlock_3D,ConvBlock3D,ModalitySpeciUpBlock_3D,\
    SpeciModalConvBlock_3D,Upsample_3D
import torch.nn.functional as F
from torch import nn
import torch
from torch.nn import init

class SeperateBnUnet_3D(nn.Module):

    def __init__(self,input_channel,init_channel,num_class,num_domains,upsample=False):
        super().__init__()
        self.upsample = upsample
        self.encoder1 = SpeciModalConvBlock_3D(input_channel,init_channel,num_domains)
        self.pool1 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.encoder2 = SpeciModalConvBlock_3D(init_channel,init_channel*2,num_domains)
        self.pool2 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.encoder3 = SpeciModalConvBlock_3D(init_channel*2, init_channel * 4,num_domains)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = SpeciModalConvBlock_3D(init_channel*4, init_channel * 8,num_domains)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.botlle_neck = SpeciModalConvBlock_3D(init_channel*8,init_channel*16,num_domains)

        if upsample:
            self.upconv4 = ModalitySpeciUpBlock_3D(init_channel * 16,init_channel *8,num_domains)
        else:
            self.upconv4 = nn.ConvTranspose3d(init_channel * 16,init_channel *8,kernel_size=2,stride=2)
        self.decoder4 = SpeciModalConvBlock_3D(init_channel * 8 *2,init_channel * 8,num_domains)
        if upsample:
            self.upconv3 = ModalitySpeciUpBlock_3D(init_channel*8,init_channel*4,num_domains)
        else:
            self.upconv3 = nn.ConvTranspose3d(init_channel*8,init_channel*4,kernel_size=2,stride=2)
        self.decoder3 = SpeciModalConvBlock_3D(init_channel*4*2,init_channel*4,num_domains)
        if upsample:
            self.upconv2 = ModalitySpeciUpBlock_3D(init_channel*4,init_channel*2,num_domains)
        else:
            self.upconv2 = nn.ConvTranspose3d(init_channel*4,init_channel*2,kernel_size=2,stride=2)
        self.decoder2 = SpeciModalConvBlock_3D(init_channel*2*2,init_channel*2,num_domains)
        if upsample:
            self.upconv1 = ModalitySpeciUpBlock_3D(init_channel*2,init_channel,num_domains)
        else:
            self.upconv1 = nn.ConvTranspose3d(init_channel*2,init_channel,kernel_size=2,stride=2)
        self.decoder1 = SpeciModalConvBlock_3D(init_channel*2,init_channel,num_domains)
        self.conv = nn.Conv3d(init_channel,num_class,kernel_size=1)

    def forward(self,x,domain_label,return_logits=False):
        enc1 = self.encoder1(x,domain_label)  # 256
        # print('enc1：',enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1),domain_label)  # 128
        enc3 = self.encoder3(self.pool2(enc2),domain_label)  # 64
        enc4 = self.encoder4(self.pool3(enc3),domain_label)  # 32

        bottleneck = self.botlle_neck(self.pool4(enc4),domain_label)  # 16
        # print('bb',bottleneck.shape)
        if self.upsample:
            dec4 = self.upconv4(bottleneck,domain_label)
        else:
            dec4 = self.upconv4(bottleneck)  # 32
        # print('dec4',dec4.shape,enc4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4,domain_label)
        # print('dec4',dec4.shape)
        if self.upsample:
            dec3 = self.upconv3(dec4,domain_label)
        else:
            dec3 = self.upconv3(dec4)  # 64

        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3,domain_label)
        if self.upsample:
            dec2 = self.upconv2(dec3,domain_label)
        else:
            dec2 = self.upconv2(dec3)  # 128
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2,domain_label)
        if self.upsample:
            dec1 = self.upconv1(dec2,domain_label)
        else:
            dec1 = self.upconv1(dec2)  # 256
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1,domain_label)
        if return_logits:
            return self.conv(dec1),dec1
        return self.conv(dec1)

class Unet_3D(nn.Module):

    def __init__(self,input_channel,init_channel,num_class=2,upsample=False,norm_type="BN"):
        super().__init__()
        self.upsample = upsample
        self.encoder1 = ConvBlock3D(input_channel,init_channel,norm_type)
        self.pool1 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.encoder2 = ConvBlock3D(init_channel,init_channel*2,norm_type)
        self.pool2 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.encoder3 = ConvBlock3D(init_channel*2, init_channel * 4,norm_type)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock3D(init_channel*4, init_channel * 8,norm_type)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.botlle_neck = ConvBlock3D(init_channel*8,init_channel*16,norm_type)

        if upsample:
            self.upconv4 = UpBlock_3D(init_channel * 16,init_channel *8,norm_type)
        else:
            self.upconv4 = nn.ConvTranspose3d(init_channel * 16,init_channel *8,kernel_size=2,stride=2)
        self.decoder4 = ConvBlock3D(init_channel * 8 *2,init_channel * 8,norm_type)
        if upsample:
            self.upconv3 = UpBlock_3D(init_channel*8,init_channel*4,norm_type)
        else:
            self.upconv3 = nn.ConvTranspose3d(init_channel*8,init_channel*4,kernel_size=2,stride=2)
        self.decoder3 = ConvBlock3D(init_channel*4*2,init_channel*4,norm_type)
        if upsample:
            self.upconv2 = UpBlock_3D(init_channel*4,init_channel*2,norm_type)
        else:
            self.upconv2 = nn.ConvTranspose3d(init_channel*4,init_channel*2,kernel_size=2,stride=2)
        self.decoder2 = ConvBlock3D(init_channel*2*2,init_channel*2,norm_type)
        if upsample:
            self.upconv1 = UpBlock_3D(init_channel*2,init_channel,norm_type)
        else:
            self.upconv1 = nn.ConvTranspose3d(init_channel*2,init_channel,kernel_size=2,stride=2)
        self.decoder1 = ConvBlock3D(init_channel*2,init_channel,norm_type)
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

class Unet_3D_IN(nn.Module):

    def __init__(self,input_channel,init_channel,num_class=2,upsample=False,norm_type="IN"):
        super().__init__()
        self.upsample = upsample
        self.encoder1 = ConvBlock3D(input_channel,init_channel,norm_type)
        self.pool1 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.encoder2 = ConvBlock3D(init_channel,init_channel*2,norm_type)
        self.pool2 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.encoder3 = ConvBlock3D(init_channel*2, init_channel * 4,norm_type)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock3D(init_channel*4, init_channel * 8,norm_type)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.botlle_neck = ConvBlock3D(init_channel*8,init_channel*16,norm_type)

        if upsample:
            self.upconv4 = UpBlock_3D_adaptnorm(init_channel * 16,init_channel *8,norm_type)
        else:
            self.upconv4 = nn.ConvTranspose3d(init_channel * 16,init_channel *8,kernel_size=2,stride=2)
        self.decoder4 = ConvBlock3D(init_channel * 8 *2,init_channel * 8,norm_type)
        if upsample:
            self.upconv3 = UpBlock_3D(init_channel*8,init_channel*4,norm_type)
        else:
            self.upconv3 = nn.ConvTranspose3d(init_channel*8,init_channel*4,kernel_size=2,stride=2)
        self.decoder3 = ConvBlock3D(init_channel*4*2,init_channel*4,norm_type)
        if upsample:
            self.upconv2 = UpBlock_3D(init_channel*4,init_channel*2,norm_type)
        else:
            self.upconv2 = nn.ConvTranspose3d(init_channel*4,init_channel*2,kernel_size=2,stride=2)
        self.decoder2 = ConvBlock3D(init_channel*2*2,init_channel*2,norm_type)
        if upsample:
            self.upconv1 = UpBlock_3D(init_channel*2,init_channel,norm_type)
        else:
            self.upconv1 = nn.ConvTranspose3d(init_channel*2,init_channel,kernel_size=2,stride=2)
        self.decoder1 = ConvBlock3D(init_channel*2,init_channel,norm_type)
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

class Unet_3D_MultiScale_Contrast(nn.Module):

    def __init__(self,input_channel,init_channel,num_class=2,upsample=False,norm_type="IN",return_2scale=False):
        super().__init__()
        self.upsample = upsample
        self.encoder1 = ConvBlock3D(input_channel,init_channel,norm_type)
        self.pool1 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.encoder2 = ConvBlock3D(init_channel,init_channel*2,norm_type)
        self.pool2 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.encoder3 = ConvBlock3D(init_channel*2, init_channel * 4,norm_type)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock3D(init_channel*4, init_channel * 8,norm_type)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.botlle_neck = ConvBlock3D(init_channel*8,init_channel*16,norm_type)

        self.up_for_co1 = Upsample_3D(scale_factor=2,mode='trilinear')
        self.up_for_co2 = Upsample_3D(scale_factor=4, mode='trilinear')
        # self.conv_for_co = ConvBlock3D(init_channel+init_channel*2+init_channel*4,init_channel*2)

        if upsample:
            self.upconv4 = UpBlock_3D_adaptnorm(init_channel * 16,init_channel *8,norm_type)
        else:
            self.upconv4 = nn.ConvTranspose3d(init_channel * 16,init_channel *8,kernel_size=2,stride=2)
        self.decoder4 = ConvBlock3D(init_channel * 8 *2,init_channel * 8,norm_type)
        if upsample:
            self.upconv3 = UpBlock_3D(init_channel*8,init_channel*4,norm_type)
        else:
            self.upconv3 = nn.ConvTranspose3d(init_channel*8,init_channel*4,kernel_size=2,stride=2)
        self.decoder3 = ConvBlock3D(init_channel*4*2,init_channel*4,norm_type)
        if upsample:
            self.upconv2 = UpBlock_3D(init_channel*4,init_channel*2,norm_type)
        else:
            self.upconv2 = nn.ConvTranspose3d(init_channel*4,init_channel*2,kernel_size=2,stride=2)
        self.decoder2 = ConvBlock3D(init_channel*2*2,init_channel*2,norm_type)
        if upsample:
            self.upconv1 = UpBlock_3D(init_channel*2,init_channel,norm_type)
        else:
            self.upconv1 = nn.ConvTranspose3d(init_channel*2,init_channel,kernel_size=2,stride=2)
        self.decoder1 = ConvBlock3D(init_channel*2,init_channel,norm_type)
        self.conv = nn.Conv3d(init_channel,num_class,kernel_size=1)
        if return_2scale:
            self.cat_conv = nn.Conv3d(init_channel*3,init_channel,kernel_size=1)
        else:
            self.cat_conv = nn.Conv3d(init_channel*7,init_channel,kernel_size=1)

    def forward(self,x,return_reco=False,return_logits=False,return_2scale=False):
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
        if return_reco:
            dec2_co = self.up_for_co1(dec2)
            dec3_co = self.up_for_co2(dec3)
            out_co = torch.cat((dec1,dec2_co,dec3_co),dim=1)
            out_co = self.cat_conv(out_co)
            return self.conv(dec1),out_co
        if return_2scale:
            dec2_co = self.up_for_co1(dec2)
            out_co = torch.cat((dec1, dec2_co), dim=1)
            out_co = self.cat_conv(out_co)
            return self.conv(dec1), out_co
        # print('dec1:',torch.sum(torch.isnan(dec1)))
        if return_logits:
            dec2_co = self.up_for_co1(dec2)
            dec3_co = self.up_for_co2(dec3)
            out_co = torch.cat((dec1, dec2_co, dec3_co), dim=1)
            out_co = self.cat_conv(out_co)
            return self.conv(dec1), out_co
        else:
            return self.conv(dec1)

class Unet_3D_Contrast(nn.Module):

    def __init__(self,input_channel,init_channel,num_class=2,upsample=False,norm_type="IN"):
        super().__init__()
        self.upsample = upsample
        self.encoder1 = ConvBlock3D(input_channel,init_channel,norm_type)
        self.pool1 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.encoder2 = ConvBlock3D(init_channel,init_channel*2,norm_type)
        self.pool2 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.encoder3 = ConvBlock3D(init_channel*2, init_channel * 4,norm_type)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock3D(init_channel*4, init_channel * 8,norm_type)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.botlle_neck = ConvBlock3D(init_channel*8,init_channel*16,norm_type)

        self.up_for_co1 = Upsample_3D(scale_factor=2,mode='trilinear')
        self.up_for_co2 = Upsample_3D(scale_factor=4, mode='trilinear')
        # self.conv_for_co = ConvBlock3D(init_channel+init_channel*2+init_channel*4,init_channel*2)

        if upsample:
            self.upconv4 = UpBlock_3D_adaptnorm(init_channel * 16,init_channel *8,norm_type)
        else:
            self.upconv4 = nn.ConvTranspose3d(init_channel * 16,init_channel *8,kernel_size=2,stride=2)
        self.decoder4 = ConvBlock3D(init_channel * 8 *2,init_channel * 8,norm_type)
        if upsample:
            self.upconv3 = UpBlock_3D(init_channel*8,init_channel*4,norm_type)
        else:
            self.upconv3 = nn.ConvTranspose3d(init_channel*8,init_channel*4,kernel_size=2,stride=2)
        self.decoder3 = ConvBlock3D(init_channel*4*2,init_channel*4,norm_type)
        if upsample:
            self.upconv2 = UpBlock_3D(init_channel*4,init_channel*2,norm_type)
        else:
            self.upconv2 = nn.ConvTranspose3d(init_channel*4,init_channel*2,kernel_size=2,stride=2)
        self.decoder2 = ConvBlock3D(init_channel*2*2,init_channel*2,norm_type)
        if upsample:
            self.upconv1 = UpBlock_3D(init_channel*2,init_channel,norm_type)
        else:
            self.upconv1 = nn.ConvTranspose3d(init_channel*2,init_channel,kernel_size=2,stride=2)
        self.decoder1 = ConvBlock3D(init_channel*2,init_channel,norm_type)
        self.conv = nn.Conv3d(init_channel,num_class,kernel_size=1)

    def forward(self,x,return_reco=False,return_logits=False):
        enc1 = self.encoder1(x)  # 256
        # print('enc1：',enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1))  # 128
        enc3 = self.encoder3(self.pool2(enc2))  # 64
        enc4 = self.encoder4(self.pool3(enc3))  # 32

        if return_reco:
            enc2_co = self.up_for_co1(enc2)
            enc3_co = self.up_for_co2(enc3)
            out_co = torch.cat((enc1,enc2_co,enc3_co),dim=1)

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
        if return_reco:
            return self.conv(dec1),out_co
        if return_logits:
            return self.conv(dec1),dec1
        else:
            return self.conv(dec1)

class RegionModule(nn.Module):
    def __init__(self,in_chan=16*7,out_chan=16,out_dim=16,size=16):
        super().__init__()
        linear_size = size//4
        in_dim = linear_size**3 * out_chan
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_chan,out_chan,kernel_size=3,padding=1),
            nn.InstanceNorm3d(out_chan),
            nn.LeakyReLU()
        )
        self.avg_pool = nn.AvgPool3d(kernel_size=4,stride=4)
        self.mlp = nn.ModuleList([
            nn.Linear(in_dim,out_dim),
            nn.LeakyReLU()
        ])
        # self.mlp = nn.ModuleList([
        #     nn.Linear(in_dim, out_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(out_dim, out_dim)
        # ])

    def normolize(self,x):
        norm_x = torch.pow(torch.sum(torch.pow(x,2),dim=1,keepdim=True),1/2)
        # print('nan in norm x:', torch.sum(torch.isnan(norm_x)))
        x = torch.div(x,norm_x)
        return x

    def forward(self,x):
        x = self.conv_block(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        lineared_x = x
        for module in self.mlp:
            lineared_x = module(lineared_x)
        return F.normalize(lineared_x,dim=1,p=2)

class MatricLayer(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim):
        super(MatricLayer, self).__init__()
        self.avgpool = nn.AvgPool3d(kernel_size=32,stride=32)
        in_dim = in_dim*(4**3)
        self.mlp = nn.ModuleList([
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim)
        ])

    def forward(self,x):
        pooled_x = self.avgpool(x)
        pooled_x = pooled_x.flatten(1)
        lineared_x = pooled_x
        for module in self.mlp:
            lineared_x = module(lineared_x)

        x = F.normalize(lineared_x,dim=1,p=2)
        return x



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

