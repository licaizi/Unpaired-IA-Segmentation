from data_preprocessing.Data_Augmentation import get_default_augmentation, default_3D_augmentation_params
from loss.Dice_loss import SoftDiceLoss, SoftDiceLossSquared, DC_and_CE_loss
from models.Baseline.UNet import Generic_UNet
from data_preprocessing.Data_Reader_CADA import get_labeled_data,mutithread_get_data,get_all_labeled_data
from data_preprocessing.Data_Generator import DataGenerator3D
from data_preprocessing.Data_Augmentation import get_default_augmentation
from models.Model_config import config_model
from data_preprocessing.Data_Utils import split_data
from config.train_config import get_arguments
from utils_Train.Utils_Train import validation, print_log, save_checkpoint,save_final_checkpoint, pad_img_to_fit_network
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
deep_sup_size = None
if args.multi_size == 1:
    deep_sup_size = [[1,1,1],[0.5,0.5,0.5]]

train_num = args.train_num
data_type = args.data_type
if data_type == 'DSA':
    train_num = 33
# read data get_labeled_data
src_dataset = get_all_labeled_data(norm=True, one_hot=True, Data_type='DSA')
region_size = args.region_size
num_keys = args.num_keys
contrast_dim = args.contrast_dim
mine_hard = args.mine_hard
hard_samples = args.hard_samples
dataset = get_all_labeled_data(norm=True, one_hot=True, Data_type='CTA')
random.seed(2333)
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
# train_dataset_1 = train_dataset
trg_train_loader = DataGenerator3D(trg_train_dataset, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE)
trg_train_gen, _ = get_default_augmentation(trg_train_loader, None, PATCH_SIZE, params=default_3D_augmentation_params)
src_train_loader = DataGenerator3D(src_dataset, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE)
src_train_gen, _ = get_default_augmentation(src_train_loader, None, PATCH_SIZE, params=default_3D_augmentation_params)

model,criterion,optimizer,scheduler = config_model()
if torch.cuda.is_available():
    # print('cuda is avaliable')
    model.cuda()
best_dice = 0.

def transform(train_img,train_label,scaled_label=None):
    if not isinstance(train_img, torch.Tensor):
        train_img = torch.from_numpy(train_img).float()
    if not isinstance(train_label, torch.Tensor):
        train_label = torch.from_numpy(train_label).float()
        scaled_label = torch.from_numpy(scaled_label).float() if scaled_label is not None else None
    if torch.cuda.is_available():
        train_img = train_img.cuda(non_blocking=True)
        train_label = train_label.cuda(non_blocking=True)
        scaled_label = scaled_label.cuda(non_blocking=True) if scaled_label is not None else None
    if not scaled_label:
        return train_img,train_label
    return train_img,train_label,scaled_label

def get_data(train_batch,multi_size=False):
    train_img = train_batch['data']
    train_label = train_batch['target']
    if multi_size:
        label,scaled_label = train_label[0],train_label[1]
        return transform(train_img,label,scaled_label)
    else:
        return transform(train_img,train_label)

# training
for epoch in range(EPOCHES):
    model.train()
    for iter in range(BATCHES_OF_EPOCH):
        # loading data
        trg_train_batch = next(trg_train_gen)
        trg_train_img = trg_train_batch['data']
        trg_train_label = trg_train_batch['target']
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

        output,latent = model(trg_train_img,return_latent=True)
        # trg_que.put(latent)
        src_output,src_latent = model(src_train_img,return_latent=True)
        # src_que.put(src_latent)
        trg_loss = criterion(output,trg_train_label)
        writer.add_scalar("trg_loss/Dice", trg_loss.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        writer.flush()
        src_loss = criterion(src_output, src_train_label)
        writer.add_scalar("src_loss/Dice", src_loss.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        writer.flush()
        loss = (src_loss+trg_loss)/2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)
        writer.add_scalar("total_loss/Dice", loss.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        writer.flush()

        log_str = 'epoch: {}, iter: {}, lr: {}, loss1: {}, '\
            .format(epoch, iter, scheduler.get_lr()[0], loss.data.item())
        print_log(log_str, log)

    with torch.no_grad():
        model.eval()
        mean_dice, dices = validation(model, trg_val_dataset)
        log_str = 'val: epoch: {}, mean_dice: {}, dices: {}'\
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

mean_dice, dices = validation(model, trg_test_dataset)
print_log("final validation on test set: mean_dice: {}".format(mean_dice), log)
writer.close()
log.close()


