from data_preprocessing.Data_Augmentation import default_3D_augmentation_params
from data_preprocessing.Data_Reader_CADA import get_all_labeled_data
from data_preprocessing.Data_Generator import DataGenerator3D
from data_preprocessing.Data_Augmentation import get_default_augmentation
from data_preprocessing.Data_Utils import split_data
from models.Model_config import config_IN_model
from config.train_config import get_arguments
from utils_Train.Utils_Train import validation, print_log, save_checkpoint
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
if data_type == 'DSA':
    train_num = 33

dataset = get_all_labeled_data(norm=True, one_hot=True,Data_type=args.data_type)
random.seed(2333)
splits = split_data(list(dataset.keys()), K=K_FOLD, shuffle=K_FOLD_SHUFFLE)
train_dataset = {k:v for k,v in dataset.items() if k in splits[FOLD]['train']}
test_dataset = {k:v for k,v in dataset.items() if k in splits[FOLD]['val']}
print('train:',[k for k,v in train_dataset.items()])
print('test:',[k for k,v in test_dataset.items()])

if FULL_TRAINING:
    train_dataset = dict(train_dataset.items() + test_dataset.items())
    val_dataset = train_dataset
    test_dataset = train_dataset
else:
    random.seed(2333)
    all_keys = list(train_dataset.keys())
    random.shuffle(all_keys)
    train_keys = all_keys[:train_num]
    val_keys = all_keys[train_num:]
    new_train_dataset = {k: v for k, v in train_dataset.items() if k in train_keys}
    val_dataset = {k: v for k, v in train_dataset.items() if k in val_keys}
    train_dataset = new_train_dataset

print('train set: {}'.format(len(train_dataset.keys())))
print('val set: {}'.format(len(val_dataset.keys())))
print('test set: {}'.format(len(test_dataset.keys())))
print_log('train_keys:'+str(train_dataset.keys()), log)
print_log('val_keys:'+str(val_dataset.keys()), log)
print_log('test_keys:'+str(test_dataset.keys()), log)
print_log('train:val:test={}:{}:{}'.format(len(train_dataset.keys()),len(val_dataset.keys()),len(test_dataset.keys())),log)
# print(test_dataset.set())
train_dataset_1 = train_dataset
train_loader = DataGenerator3D(train_dataset, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE)
train_gen, _ = get_default_augmentation(train_loader, None, PATCH_SIZE, params=default_3D_augmentation_params)

model,criterion,optimizer,scheduler = config_IN_model()

if torch.cuda.is_available():
    # print('cuda is avaliable')
    model.cuda()
best_dice1 = 0.
# training
for epoch in range(EPOCHES):
    model.train()
    for iter in range(BATCHES_OF_EPOCH):
        # loading data
        train_batch = next(train_gen)
        train_img = train_batch['data']
        train_label = train_batch['target']

        if not isinstance(train_img, torch.Tensor):
            train_img = torch.from_numpy(train_img).float()
        if not isinstance(train_label, torch.Tensor):
            train_label = torch.from_numpy(train_label).float()
        if torch.cuda.is_available():
            train_img = train_img.cuda(non_blocking=True)
            train_label = train_label.cuda(non_blocking=True)

        output = model(train_img)
        loss = criterion(output,train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)
        writer.add_scalar("loss1/Dice", loss.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        writer.flush()

        log_str = 'epoch: {}, iter: {}, lr: {}, loss1: {}, '\
            .format(epoch, iter, scheduler.get_lr()[0], loss.data.item())
        print_log(log_str, log)
    # update_ema_variables(model1, ema_model1, args.ema_decay, epoch * BATCHES_OF_EPOCH + iter)
    # evaluation
    with torch.no_grad():
        model.eval()
        mean_dice1, dices1 = validation(model, val_dataset)
        log_str = 'val: epoch: {}, mean_dice1: {}, dices: {}'\
            .format(epoch, mean_dice1, dices1)
        print_log(log_str, log)
        writer.add_scalar("val acc1/Dice", mean_dice1, epoch * BATCHES_OF_EPOCH)
        writer.flush()
        print_log("saving checkpoint...", log)
        is_best1 = False
        if mean_dice1 > best_dice1:
            best_model = model
            best_dice1 = mean_dice1
            is_best1 = True
            log_str = 'best dice1: mean_dice1: {}'.format(mean_dice1)
            print_log(log_str, log)
        
            state1 = {
                'epoch1': epoch,
                'best_epoch1': best_dice1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(state1, is_best1, CHECKPOINT_PATH, CHECKPOINT_NAME1.format(epoch), 'model_best.pth')
state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }

mean_dice, dices = validation(best_model, test_dataset)
print_log("final validation on test set: mean_dice: {}".format(mean_dice), log)
writer.close()
log.close()


