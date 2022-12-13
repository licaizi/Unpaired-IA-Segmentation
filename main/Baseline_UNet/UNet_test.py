from data_preprocessing.Data_Reader_CADA import get_all_labeled_data
from data_preprocessing.Data_Utils import split_data
from models.Model_config import config_model,config_dsbn_model,config_IN_model
from config.train_config import get_arguments
from utils_Train.Utils_Train import validation, print_log
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

log = open(os.path.join(log_path, 'log_{}.txt'.format(now)), 'w')
print_log('save path : {}'.format(log_path), log)
# initialize summarywriter
writer = SummaryWriter(SUMMARY_WRITER)
FULL_TRAINING = True if args.full_training == 1 else False

train_num = args.train_num
data_type = args.data_type
if data_type == 'DSA':
    train_num = 33
# read data get_labeled_data
dataset = get_all_labeled_data(norm=True, one_hot=True, Data_type='CTA')
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
contrast_dim = args.contrast_dim
model,criterion,optimizer,scheduler = config_IN_model()


if torch.cuda.is_available():
    print('cuda is avaliable')
    model.cuda()
model_path = args.pretrained_model.format(args.fold)
static = torch.load(model_path)
optimizer_static = static['optimizer']
model.load_state_dict(static['model'])

allindice = args.all_indice
if allindice:
    print('test on all indeces')
    mean_dice,mean_jaccard,mean_sensitivity,mean_hd95 = validation(model,test_dataset,all_indice=True)
    print_log('Data:{},mean_dice:{},mean_jaccard:{},mean_sensitivity:{},mean_hd95:{}'.format(
        data_type,mean_dice,mean_jaccard,mean_sensitivity,mean_hd95),log)

else:
    mean_dice, dices = validation(model, test_dataset)
    print_log('Data:{},mean_dice:{},dice:{}'.format(data_type,mean_dice,dices),log)



