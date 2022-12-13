from data_preprocessing.Data_Augmentation import get_default_augmentation, default_3D_augmentation_params
from data_preprocessing.Data_Reader_CADA import get_labeled_data, mutithread_get_data,get_all_labeled_data
from data_preprocessing.Data_Generator import DataGenerator3D
from data_preprocessing.Data_Augmentation import get_default_augmentation
from data_preprocessing.Data_Utils import split_data
from models.Model_config import config_LMI_region_model
from utils.region_contrast import generate_batch_mulmod_allother_regions,\
    is_no_target,generate_batch_mulmod_context_regions
from config.train_config import get_arguments
from loss.region_contrast_loss import intra_inter_contrast_loss,LMI_cross_contrast_loss
from utils_Train.Utils_Train import validation, print_log, save_checkpoint, \
    get_current_consistency_weight
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
import os
import argparse
import random

np.random.seed(123)
torch.manual_seed(1234)
if torch.cuda.is_available():
    print('torch.cuda.is_available()')
    torch.cuda.manual_seed_all(123456)
cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args = get_arguments()
LOG_PATH = args.log_path.format(args.fold)
CHECKPOINT_PATH = args.checkpoint_path.format(args.fold)
CHECKPOINT_NAME1 = args.checkpoint_name1
CHECKPOINT_NAME2 = args.checkpoint_name2
SUMMARY_WRITER = args.summary_writer.format(args.fold)
PATCH_SIZE = (args.patch_size_x, args.patch_size_y, args.patch_size_z)
BATCH_SIZE = args.batch_size
NUM_CLASSES = args.num_classes
BATCHES_OF_EPOCH = args.batches_of_epoch
EPOCHES = args.epoches
NUM_POOL = args.num_pool
FOLD = args.fold
K_FOLD = args.k_fold
K_FOLD_SHUFFLE = True if args.k_fold_shuffle == 1 else False
# initialize log
now = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
log_path = LOG_PATH
if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)
log = open(os.path.join(log_path, 'log_{}.txt'.format(now)), 'w')
print_log('save path : {}'.format(log_path), log)
# initialize summarywriter
writer = SummaryWriter(SUMMARY_WRITER)
FULL_TRAINING = True if args.full_training == 1 else False

train_num = args.train_num
data_type = args.data_type

# read data get_labeled_data

src_dataset = get_all_labeled_data(norm=True, one_hot=True, Data_type='DSA')
dsa_num = args.dsa_num
random.seed(2333)
if args.filter_dsa:
    print('filter dsa data......')
    print_log('total number of dsa:{},and used number:{}'.format(len(src_dataset),args.dsa_num),log)
    all_src_keys = list(src_dataset.keys())
    random.shuffle(all_src_keys)
    src_keys = all_src_keys[:dsa_num]
    src_dataset = {k:v for k,v in src_dataset.items() if k in src_keys}

region_size = args.region_size
num_keys = args.num_keys
contrast_dim = args.contrast_dim
mine_hard = args.mine_hard
hard_samples = args.hard_samples
dataset = get_all_labeled_data(norm=True, one_hot=True, Data_type='CTA')

splits = split_data(list(dataset.keys()), K=K_FOLD, shuffle=K_FOLD_SHUFFLE)
trg_train_dataset = {k: v for k, v in dataset.items() if k in splits[FOLD]['train']}
trg_test_dataset = {k: v for k, v in dataset.items() if k in splits[FOLD]['val']}
print('train:', [k for k, v in trg_train_dataset.items()])
print('test:', [k for k, v in trg_test_dataset.items()])
# n-fold or full
if FULL_TRAINING:
    trg_train_dataset = dict(trg_train_dataset.items() + trg_test_dataset.items())
    trg_val_dataset = trg_train_dataset
    trg_test_dataset = trg_train_dataset
else:
    random.seed(2333)
    all_keys = list(trg_train_dataset.keys())
    random.shuffle(all_keys)
    train_keys = all_keys[:train_num]
    val_keys = all_keys[train_num:]
    new_train_dataset = {k: v for k, v in trg_train_dataset.items() if k in train_keys}
    trg_val_dataset = {k: v for k, v in trg_train_dataset.items() if k in val_keys}
    trg_train_dataset = new_train_dataset

print('trg train set: {}'.format(len(trg_train_dataset.keys())))
print('trg_val set: {}'.format(len(trg_val_dataset.keys())))
print('trg_test set: {}'.format(len(trg_test_dataset.keys())))
print_log('trg_train_keys:' + str(trg_train_dataset.keys()), log)
print_log('trg_val_keys:' + str(trg_val_dataset.keys()), log)
print_log('trg_test_keys:' + str(trg_test_dataset.keys()), log)
print_log('trg_train:val:test={}:{}:{}'.format(len(trg_train_dataset.keys()), len(trg_val_dataset.keys()),
                                               len(trg_test_dataset.keys())), log)

trg_train_loader = DataGenerator3D(trg_train_dataset, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE)
trg_train_gen, _ = get_default_augmentation(trg_train_loader, None, PATCH_SIZE, params=default_3D_augmentation_params)
src_train_loader = DataGenerator3D(src_dataset, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE)
src_train_gen, _ = get_default_augmentation(src_train_loader, None, PATCH_SIZE, params=default_3D_augmentation_params)

model,region_model,global_embed, criterion, optimizer, scheduler = config_LMI_region_model(cotain_logits=True,size=region_size,out_dim=contrast_dim)
if torch.cuda.is_available():
    model.cuda()
    region_model.cuda()
    global_embed.cuda()
best_dice = 0.

# training
for epoch in range(EPOCHES):
    model.train()
    src_losses, trg_losses, intra_co_losses, inter_co_losses, total_losses,lmi_losses = [], [], [], [], [],[]
    for iter in range(BATCHES_OF_EPOCH):
        # loading data
        trg_train_batch = next(trg_train_gen)
        while is_no_target(trg_train_batch):
            trg_train_batch = next(trg_train_gen)
        trg_train_img = trg_train_batch['data']
        trg_train_label = trg_train_batch['target']
        src_train_batch = next(src_train_gen)
        while is_no_target(src_train_batch):
            src_train_batch = next(src_train_gen)
        src_train_img = src_train_batch['data']
        src_train_label = src_train_batch['target']

        if not isinstance(trg_train_img, torch.Tensor):
            trg_train_img = torch.from_numpy(trg_train_img).float()
            src_train_img = torch.from_numpy(src_train_img).float()
        if not isinstance(trg_train_label, torch.Tensor):
            trg_train_label = torch.from_numpy(trg_train_label).float()
            src_train_label = torch.from_numpy(src_train_label).float()
        if torch.cuda.is_available():
            trg_train_img = trg_train_img.cuda(non_blocking=True)
            trg_train_label = trg_train_label.cuda(non_blocking=True)
            src_train_img = src_train_img.cuda(non_blocking=True)
            src_train_label = src_train_label.cuda(non_blocking=True)

        output, latent = model(trg_train_img, return_logits=True)

        src_output, src_latent = model(src_train_img, return_logits=True)
        co_weight = get_current_consistency_weight(epoch,args)
        src_sample_regions, trg_sample_regions = generate_batch_mulmod_context_regions(src_latent, src_train_label,
                                                            latent, trg_train_label,num_pos=2, size=region_size,num_keys=num_keys)

        trg_embed,src_embed = global_embed(latent),global_embed(src_latent)
        src_sample_regions = region_model(src_sample_regions)
        trg_sample_regions = region_model(trg_sample_regions)
        inter_co_loss,intra_co_loss = intra_inter_contrast_loss(src_sample_regions,trg_sample_regions, num_keys,
                                                        args.temperature,mine_hard=mine_hard,hard_samples=hard_samples)
        lmi_loss = LMI_cross_contrast_loss(trg_embed,src_embed,src_sample_regions,trg_sample_regions,num_keys,args.temperature)
        intra_co_losses.append(intra_co_loss)
        inter_co_losses.append(inter_co_loss)
        trg_loss = criterion(output, trg_train_label)
        trg_losses.append(trg_loss)
        writer.add_scalar("trg_loss", trg_loss.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        writer.flush()
        src_loss = criterion(src_output, src_train_label)
        src_losses.append(src_loss)
        lmi_losses.append(lmi_loss)

        writer.add_scalar("src_loss", src_loss.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        writer.add_scalar("intra_co_loss", intra_co_loss.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        writer.add_scalar("inter_co_loss", inter_co_loss.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        writer.add_scalar("lmi_loss", lmi_loss.data.item(), epoch * BATCHES_OF_EPOCH + iter)

        writer.flush()
        loss = src_loss + trg_loss + co_weight*intra_co_loss + co_weight * inter_co_loss + co_weight*lmi_loss
        total_losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)
        writer.add_scalar("total_loss", loss.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        writer.flush()

        log_str = 'epoch: {}, iter: {}, lr: {}, loss1: {}, ' \
            .format(epoch, iter, scheduler.get_lr()[0], loss.data.item())
        print_log(log_str, log)

    writer.add_scalar("mean_src_loss", sum(src_losses).data.item()/len(src_losses), epoch)
    writer.add_scalar("mean_trg_loss", sum(trg_losses).data.item() / len(trg_losses), epoch)
    writer.add_scalar("mean_trg_selfco_loss", sum(intra_co_losses).data.item() / len(intra_co_losses), epoch)
    writer.add_scalar("mean_trg_co_loss", sum(inter_co_losses).data.item() / len(inter_co_losses), epoch)
    writer.add_scalar("mean_total_loss", sum(total_losses).data.item() / len(total_losses), epoch)
    writer.add_scalar("mean_lmi_loss", sum(lmi_losses).data.item() / len(lmi_losses), epoch)
    writer.flush()
    # evaluation
    with torch.no_grad():
        model.eval()
        mean_dice, dices = validation(model, trg_val_dataset)
        log_str = 'val: epoch: {}, mean_dice: {}, dices: {}' \
            .format(epoch, mean_dice, dices)
        print_log(log_str, log)
        writer.add_scalar("val acc/Dice", mean_dice, epoch * BATCHES_OF_EPOCH)
        writer.flush()
        print_log("saving checkpoint...", log)
        is_best = False
        if mean_dice > best_dice:
            best_model = model
            best_dice = mean_dice
            is_best = True
            log_str = 'best dice: mean_dice: {}'.format(mean_dice)
            print_log(log_str, log)

            state = {
                'epoch': epoch,
                'best_epoch': best_dice,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(state, is_best, CHECKPOINT_PATH, CHECKPOINT_NAME1.format(epoch), 'model_best.pth')

mean_dice, dices = validation(best_model, trg_test_dataset)
print_log("final validation on test set: mean_dice: {}".format(mean_dice), log)
writer.close()
log.close()