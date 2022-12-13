import torch
import os
from collections import OrderedDict
from models.Model_Utils import get_scheduler
import shutil
import numpy as np
from models.Model_Utils import softmax_helper
from utils.Metrics import dice


class BaseModel():

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        # print('gpu_ids',self.gpu_ids)
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        # self.device = torch.device('cuda') if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []
        self.optimizers = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # load and print networks; create schedulers
    def setup(self, opt, parser=None):
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        # self.print_networks(opt.verbose)

    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def poly_update_lr(self,opt,iter_num):
        lr_ = opt.initial_lr * (1.0 - iter_num / (self.EPOCHES * self.BATCHES_OF_EPOCH)) ** 0.9
        for param_group in self.optimizers[0].param_groups:
            param_group['lr'] = lr_

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net_' + name)
                # print(self.gpu_ids,'.....')
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # torch.save(net.module.cpu().state_dict(), save_path)
                    torch.save(net.state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def validation(self,model, dataset, dims=3):
        dices = []
        for key in dataset.keys():
            img_ed = dataset[key]['img']  # (13, 216, 256)
            img_ed_gt = dataset[key]['gt']  # (13, 4, 216, 256)

            # crop image
            patch_size = (128, 128)
            img_ed = np.expand_dims(img_ed, axis=0)
            center_pt = dataset[key]['center'][0]  # random.choice(dataset[key]['center'])
            lt_x = int(max(0, center_pt[0] - patch_size[0] / 2))
            lt_y = int(max(0, center_pt[1] - patch_size[1] / 2))

            if dims == 3:
                rb_x = int(min(img_ed.shape[2], lt_x + patch_size[0]))
                rb_y = int(min(img_ed.shape[3], lt_y + patch_size[1]))
                patch_size = (128, 128, 128)
                lt_s = int(max(0, center_pt[2] - patch_size[2] / 2))
                rb_s = int(min(img_ed.shape[1], lt_s + patch_size[2]))
                crop_img = np.zeros((1, 128, 128, 128))
                crop_img[:, :rb_s - lt_s, :rb_x - lt_x, :rb_y - lt_y] = img_ed[:, lt_s:rb_s, lt_x:rb_x,
                                                                        lt_y:rb_y]  # in case that crop length < 128
            else:
                rb_x = int(min(img_ed.shape[1], lt_x + patch_size[0]))
                rb_y = int(min(img_ed.shape[2], lt_y + patch_size[1]))
                crop_img = np.zeros((1, 128, 128))
                crop_img[:,  :rb_x - lt_x, :rb_y - lt_y] = img_ed[:, lt_x:rb_x,
                                                                        lt_y:rb_y]  # in case that crop length < 128
            crop_img_pad = crop_img
            input_data = np.expand_dims(crop_img_pad, axis=0)
            crop_output = softmax_helper(model(torch.from_numpy(input_data).float().cuda())).detach().cpu().numpy()
            # crop_output = model(torch.from_numpy(input_data).float().cuda()).detach().cpu().numpy()
            patch_pred = crop_output.squeeze().argmax(0)
            pred_map = np.zeros(img_ed.shape[1:], dtype=np.int64)
            if dims == 3:
                pred_map[lt_s:rb_s, lt_x:rb_x, lt_y:rb_y] = patch_pred[:rb_s - lt_s, :rb_x - lt_x, :rb_y - lt_y]
            else:
                pred_map[lt_x:rb_x, lt_y:rb_y] = patch_pred[ :rb_x - lt_x, :rb_y - lt_y]
            dices.append(dice(pred_map, img_ed_gt.argmax(0)))

        return np.mean(dices), dices

    def save_checkpoint(self,state, is_best, save_path, filename, bestname):
        filename = os.path.join(save_path, filename)
        torch.save(state, filename)
        if is_best:
            bestname = os.path.join(save_path, bestname)
            shutil.copyfile(filename, bestname)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad