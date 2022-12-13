import torch
from torch import nn
from config.train_config import get_arguments
from torch.optim import lr_scheduler
from loss.Dice_loss import SoftDiceLoss
from Multi_Modal_Seg.Dual_Stream_Network import Encoder,LatentEncoder,Decoder
from Multi_Modal_Seg.UNet import Unet_3D,SeperateBnUnet_3D,MatricLayer,\
    Unet_3D_IN,Unet_3D_Contrast,RegionModule,Unet_3D_MultiScale_Contrast

from Multi_Modal_Seg.mb_builder import RegionCo

args = get_arguments()
# cross validation
VENDOR = args.vendor
INPUT_CHANNELS = args.input_channel
BASE_NUM_FEATURES = args.base_num_features
MAX_FILTERS = args.max_filters
NUM_POOL = args.num_pool
NUM_CLASSES = args.num_classes
INITIAL_LR = args.initial_lr
LR_STEP_SIZE = args.lr_step_size
LR_GAMMA = args.lr_gamma
WEIGHT_DECAY = args.weight_decay
NONLIN = args.nonlin
NORM_OP = args.norm_op
conv_op = nn.Conv3d
dropout_op = nn.Dropout3d

if NORM_OP == 1:
    norm_op = nn.InstanceNorm3d
elif NORM_OP == 2:
    norm_op = nn.BatchNorm3d
else:
    raise Exception('Norm_OP Invalid!')
norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
if NONLIN == 1:
    net_nonlin = nn.ReLU
elif NONLIN == 2:
    net_nonlin = nn.LeakyReLU
else:
    raise Exception('nonlin invalid!')

net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
final_nonlin = lambda x: x

def config_model(contains_feat = False):
    #define model
    model = Unet_3D(1,16,2)
    # define loss func
    criterion = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, do_bg=False, smooth=1e-7)
    if contains_feat:
        feat_model = MatricLayer(64,32)
        optimizer = torch.optim.Adam([{'params':model.parameters()},
                                      {'params':feat_model.parameters()}],INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
        scheduler = lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)
        return model,feat_model,criterion,optimizer,scheduler
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    scheduler = lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)
    return model,criterion,optimizer,scheduler

def config_dsbn_model():
    #define model
    model = SeperateBnUnet_3D(1,16,2,2)
    # define loss func
    criterion = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, do_bg=False, smooth=1e-7)
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    scheduler = lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)
    return model,criterion,optimizer,scheduler

def config_region_dsbn_model(cotain_logits=False,size=16):
    #define model
    model = SeperateBnUnet_3D(1,16,2,2)
    if cotain_logits:
        region_model = RegionModule(16,16,16,size)
    else:
        region_model = RegionModule()
    # define loss func
    criterion = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, do_bg=False, smooth=1e-7)
    optimizer = torch.optim.Adam([{'params':model.parameters()},
                                      {'params':region_model.parameters()}],INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    scheduler = lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)
    return model,region_model,criterion,optimizer,scheduler

def config_IN_model(contains_feat=False):
    #define model
    model = Unet_3D_IN(1,16,2,norm_type="IN")
    # define loss func
    criterion = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, do_bg=False, smooth=1e-7)
    if contains_feat:
        feat_model = MatricLayer(1024,256,32)
        optimizer = torch.optim.Adam([{'params':model.parameters()},
                                      {'params':feat_model.parameters()}]
                                     , INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
        scheduler = lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)
        return model,feat_model,criterion,optimizer,scheduler
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    scheduler = lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)
    return model,criterion,optimizer,scheduler


def config_region_model(cotain_logits=False,size=16,out_dim=128):
    #define model
    model = Unet_3D_Contrast(1,16,2)
    if cotain_logits:
        region_model = RegionModule(16,16,out_dim,size)
    else:
        region_model = RegionModule()
    # define loss func
    criterion = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, do_bg=False, smooth=1e-7)
    optimizer = torch.optim.Adam([{'params':model.parameters()},
                                      {'params':region_model.parameters()}],INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    scheduler = lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)
    return model,region_model,criterion,optimizer,scheduler

def config_LMI_region_model(cotain_logits=False,size=16,out_dim=128):
    #define model
    model = Unet_3D_Contrast(1,16,2)
    if cotain_logits:
        region_model = RegionModule(16,16,out_dim,size)
    else:
        region_model = RegionModule()
    global_embed = MatricLayer(16,out_dim,out_dim)
    # define loss func
    criterion = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, do_bg=False, smooth=1e-7)
    optimizer = torch.optim.Adam([{'params':model.parameters()},
                                      {'params':region_model.parameters()},
                                  {'params':global_embed.parameters()}],INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    scheduler = lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)
    return model,region_model,global_embed,criterion,optimizer,scheduler

def config_multiscale_region_model(cotain_logits=False,size=16,out_dim=128,scales=3):
    #define model
    return_2scale=False
    if scales == 2:
        return_2scale=True
    model = Unet_3D_MultiScale_Contrast(1,16,2,return_2scale=return_2scale)
    if cotain_logits:
        region_model = RegionModule(16,16,out_dim,size)
    else:
        region_model = RegionModule()
    # define loss func
    criterion = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, do_bg=False, smooth=1e-7)
    optimizer = torch.optim.Adam([{'params':model.parameters()},
                                      {'params':region_model.parameters()}],INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    scheduler = lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)
    return model,region_model,criterion,optimizer,scheduler

def config_multi_region_model(cotain_logits=False,src_size=24,trg_size=16,out_dim=128):
    #define model
    model = Unet_3D_Contrast(1,16,2)
    if cotain_logits:
        src_region_model = RegionModule(16,16,out_dim,src_size)
        trg_region_model = RegionModule(16, 16, out_dim, trg_size)
    else:
        region_model = RegionModule()
    # define loss func
    criterion = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, do_bg=False, smooth=1e-7)
    optimizer = torch.optim.Adam([{'params':model.parameters()},{'params':trg_region_model.parameters()},
                                      {'params':src_region_model.parameters()}],INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    scheduler = lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)
    return model,src_region_model,trg_region_model,criterion,optimizer,scheduler

def config_region_moco_model(cotain_logits=False,size=16,T=0.1,k=256,sample_k=16):
    #define model
    model = Unet_3D_Contrast(1,16,2)
    if cotain_logits:
        region_moco_model = RegionCo(dim=16,K=k,m=0.999,T=T,region_size=size,sample_k=sample_k)
    else:
        region_moco_model = RegionModule()
    # define loss func
    criterion = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, do_bg=False, smooth=1e-7)
    optimizer = torch.optim.Adam([{'params':model.parameters()},
                                      {'params':region_moco_model.parameters()}],INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    scheduler = lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)
    return model,region_moco_model,criterion,optimizer,scheduler


def config_dual_model():
    #define model
    src_encoder = Encoder(1, 16)
    trg_encoder = Encoder(1, 16)
    latent_encoder = LatentEncoder(256, 256)
    src_decoder = Decoder(16, 2)
    trg_decoder = Decoder(16, 2)
    # define loss func
    criterion = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, do_bg=False, smooth=1e-7)
    # define optimizer
    src_optimizer = torch.optim.Adam([
        {'params': src_encoder.parameters()},
        {'params': latent_encoder.parameters()},
        {'params': src_decoder.parameters()}
    ], INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    src_scheduler = lr_scheduler.StepLR(src_optimizer, LR_STEP_SIZE, LR_GAMMA)
    trg_optimizer = torch.optim.Adam([
        {'params': trg_encoder.parameters()},
        {'params': latent_encoder.parameters()},
        {'params': trg_decoder.parameters()}
    ], INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    trg_scheduler = lr_scheduler.StepLR(trg_optimizer, LR_STEP_SIZE, LR_GAMMA)
    return src_encoder, src_decoder, trg_encoder, trg_decoder, latent_encoder, criterion, \
           src_optimizer, src_scheduler, trg_optimizer, trg_scheduler

def config_dualy_model():
    #define model
    # src_encoder = Encoder(1, 16)
    encoder = Encoder(1, 16)
    # latent_encoder = LatentEncoder(256, 256)
    src_decoder = Decoder(16, 2)
    trg_decoder = Decoder(16, 2)
    # define loss func
    criterion = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, do_bg=False, smooth=1e-7)
    # define optimizer
    src_optimizer = torch.optim.Adam([
        {'params': encoder.parameters()},
        {'params': src_decoder.parameters()}
    ], INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    src_scheduler = lr_scheduler.StepLR(src_optimizer, LR_STEP_SIZE, LR_GAMMA)
    trg_optimizer = torch.optim.Adam([
        {'params': encoder.parameters()},
        {'params': trg_decoder.parameters()}
    ], INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    trg_scheduler = lr_scheduler.StepLR(trg_optimizer, LR_STEP_SIZE, LR_GAMMA)
    return encoder, src_decoder, trg_decoder, criterion, \
           src_optimizer, src_scheduler, trg_optimizer, trg_scheduler


def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

