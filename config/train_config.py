import argparse

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for UNet training")
    parser.add_argument('--fold', type=int, default=0,
                        help='cross validation fold No.', )
    parser.add_argument('--k_fold', type=int, default=5,
                        help='cross validation: number of folds.', )
    parser.add_argument('--k_fold_shuffle', type=int, default=1,
                        help='whether shuffle data list before split dataset.', )
    parser.add_argument('--dsa_num', type=int, default=50,
                        help='number of dsa dataset.', )
    parser.add_argument('--full_training', type=int, default=0,
                        help='whether to use all samples to train', )

    parser.add_argument('--vendor', type=str, default='A',
                        help='where the dataset comes from', )
    parser.add_argument('--patch_size_x', type=int, default=128,
                        help='training patch size x', )
    parser.add_argument('--patch_size_y', type=int, default=128,
                        help='training patch size y', )
    parser.add_argument('--patch_size_z', type=int, default=128,
                        help='training patch size z', )
    parser.add_argument('--batch_size', type=int, default=2,
                        help='training batch size', )
    parser.add_argument('--region_size', type=int, default=16,
                        help='region size for contrast', )
    parser.add_argument('--src_region_size', type=int, default=16,
                        help='src region size for contrast', )
    parser.add_argument('--trg_region_size', type=int, default=16,
                        help='trg region size for contrast', )
    parser.add_argument('--contrast_dim', type=int, default=16,
                        help='out_dim for contrast', )
    parser.add_argument('--num_pos', type=int, default=8,
                        help='number of positive samples for contrast', )
    parser.add_argument('--all_indice',action='store_true',help='whether test all indeices', )
    parser.add_argument('--filter_dsa', action='store_true', help='whether test all indeices', )
    parser.add_argument('--mine_hard',action='store_true')
    parser.add_argument('--broad_cast', action='store_true'
                        ,help='whether broad cast the negative samples among a batch for contrast')
    parser.add_argument('--hard_samples', type=int, default=8,
                        help='number of hard samples for contrast', )
    parser.add_argument('--num_keys', type=int, default=8,
                        help='number of keys for contrast', )
    parser.add_argument('--sample_k', type=int, default=16,
                        help='number of sampled keys during each iteration', )
    parser.add_argument('--temperature', type=float, default=1.,
                        help='temperature for contrast', )
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of targets', )
    parser.add_argument('--multi_size', type=int, default=0,
                        help='prepare mutltisize if 1', )
    parser.add_argument('--num_samples', type=int, default=400,
                        help='number of contrastive samples', )
    parser.add_argument('--lamda', type=float, default=0.5,
                        help='initial weight for kd_loss', )
    parser.add_argument('--beta', type=float, default=1.0,
                        help='initial args for cutmix or HCI', )
    parser.add_argument('--tau_plus', type=float, default=0.05,
                        help='parameters for possiblity of neg in pos for HCI', )
    parser.add_argument('--L', type=float, default=0.1,
                        help='initial weight for fda', )
    parser.add_argument('--belta', type=float, default=0.5,
                        help='initial weight for contrastive loss', )
    parser.add_argument('--input_channel', type=int, default=1,
                        help='number of channels of input data', )
    parser.add_argument('--base_num_features', type=int, default=16,
                        help='number of features in the first stage of UNet', )
    parser.add_argument('--max_filters', type=int, default=512,
                        help='max number of features in UNet', )
    parser.add_argument('--num_pool', type=int, default=5,
                        help='number of pool ops in UNet', )

    parser.add_argument('--initial_lr', type=float, default=1e-4,
                        help='initial learning rate', )
    parser.add_argument('--lr_step_size', type=int, default=60,
                        help='lr * lr_gamma every lr_step_size', )
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='lr * lr_gamma every lr_step_size', )
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='used in optimizer', )
    parser.add_argument('--batches_of_epoch', type=int, default=400,
                        help='iterations in an epoch', )
    parser.add_argument('--epoches', type=int, default=50,
                        help='training epoches in total', )
    parser.add_argument('--train_num', type=int, default=36,
                        help='the number of training data', )
    parser.add_argument('--labeled_num', type=int, default=38,
                        help='the number of training data', )
    parser.add_argument('--nonlin', type=int, default=2,
                        help='1:ReLU, 2: LReLU', )
    parser.add_argument('--norm_op', type=int, default=1,
                        help='1:InstanceNorm, 2: BatchNorm', )
    parser.add_argument('--data_type', type=str, default='CTA',
                        help='type of data', )
    parser.add_argument('--log_path', type=str, default='../../logs/Baseline_UNet/day24_1e-4{}',
                        help='log path', )
    parser.add_argument('--teacher_path', type=str, default='../../logs/Baseline_UNet/day24_1e-4{}',
                        help='saved path of teacher model', )
    parser.add_argument('--src_pretrained_path', type=str, default='../../logs/Baseline_UNet/day24_1e-4{}',
                        help='saved path of teacher model', )
    parser.add_argument('--model_path', type=str, default='', help='model path')
    parser.add_argument('--checkpoint_path', type=str, default='../../checkpoints/Baseline_UNet/day24_1e-4{}',
                        help='checkpoint path', )
    parser.add_argument('--saved_root', type=str, default='../../results/Baseline_UNet/day24_1e-4{}',
                        help='save path', )
    parser.add_argument('--checkpoint_name1', type=str, default='checkpoint-CTA-{}.pth',
                        help='checkpoint name', )
    parser.add_argument('--final_model', type=str, default='checkpoint-CTA-{}.pth',
                        help='final checkpoint name', )
    parser.add_argument('--checkpoint_name2', type=str, default='checkpoint2-{}.pth',
                        help='checkpoint name', )
    parser.add_argument('--summary_writer', type=str, default='../../logs/Baseline_UNet/day24_1e-4{}',
                        help='checkpoint name', )
    parser.add_argument('--pretrained_model', type=str, default='../../checkpoints/Baseline_UNet/day24_1e-4{}',
                        help='checkpoint name', )
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
    parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
    parser.add_argument('--differ_data', type=int, default=0, help='input different dataset into the two mutual models')

    return parser.parse_args()
# import time
# print('start:...')
# time.sleep(20)
# print('end...')
# path = '../'
# import os
# print(os.listdir(path))

