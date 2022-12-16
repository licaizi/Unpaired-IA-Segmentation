from data_preprocessing.Data_Utils import save_nii
from data_preprocessing.Data_Reader_CADA import get_all_labeled_data
from data_preprocessing.Data_Utils import split_data
from models.Model_config import config_model,config_IN_model
from config.train_config import get_arguments
from utils_Train.Utils_Train import validation, print_log,test_data, save_checkpoint, pad_img_to_fit_network
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import time
import os
import argparse
import random

cri = nn.CrossEntropyLoss()
np.random.seed(123)
torch.manual_seed(1234)
if torch.cuda.is_available():
    print('torch.cuda.is_available()')
    torch.cuda.manual_seed_all(123456)
cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args = get_arguments()

saved_path = args.saved_root.format(args.fold)
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
if not os.path.isdir(saved_path):
    os.makedirs(saved_path)
FULL_TRAINING = True if args.full_training == 1 else False

train_num = args.train_num
data_type = args.data_type
if data_type == 'DSA':
    train_num = 33
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

model,criterion,optimizer,scheduler = config_IN_model()

if torch.cuda.is_available():
    print('cuda is avaliable')
    model.cuda()
model_path = args.pretrained_model.format(args.fold)
static = torch.load(model_path)
model_state = static['model']
new_state = {}
for key,value in model_state.items():
    if key.startswith('module'):
        key = key[7:]
    new_state[key] = value
optimizer_static = static['optimizer']
model.load_state_dict(new_state)
test_data(model,test_dataset,saved_path)
