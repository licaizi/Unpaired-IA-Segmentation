from config.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=200, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=200, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_lr', type=float, default=0.0001, help='initial learning rate for adam of Discrimitor')
        parser.add_argument('--G_lr', type=float, default=0.001, help='initial learning rate for adam of generator')
        parser.add_argument('--no_lsgan', default=True, help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=200, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--fold', type=int, default=0,
                            help='cross validation fold No.', )
        parser.add_argument('--k_fold', type=int, default=5,
                            help='cross validation: number of folds.', )
        parser.add_argument('--k_fold_shuffle', type=int, default=1,
                            help='whether shuffle data list before split dataset.', )
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
        parser.add_argument('--num_classes', type=int, default=2,
                            help='number of targets', )
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
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='used in optimizer', )
        parser.add_argument('--batches_of_epoch', type=int, default=400,
                            help='iterations in an epoch', )#src_train_num
        parser.add_argument('--epoches', type=int, default=50,
                            help='training epoches in total', )
        parser.add_argument('--labeled_num', type=int, default=38,
                            help='the number of training data', )
        parser.add_argument('--nonlin', type=int, default=2,
                            help='1:ReLU, 2: LReLU', )
        parser.add_argument('--norm_op', type=int, default=1,
                            help='1:InstanceNorm, 2: BatchNorm', )
        parser.add_argument('--data_type', type=str, default='DSA',
                            help='type of data', )
        parser.add_argument('--log_path', type=str, default='../../logs/Image_translate/day7_cycleGan',
                            help='log path', )
        parser.add_argument('--model_path', type=str, default='', help='model path')
        parser.add_argument('--checkpoint_path', type=str, default='../../checkpoints/Image_translate/day7_cycleGan',
                            help='checkpoint path', )
        parser.add_argument('--checkpoint_name', type=str, default='checkpoint-CTA-{}.pth',
                            help='checkpoint name', )
        parser.add_argument('--checkpoint_name2', type=str, default='checkpoint2-{}.pth',
                            help='checkpoint name', )
        parser.add_argument('--summary_writer', type=str, default='../../logs/Image_translate/day7_cycleGan',
                            help='checkpoint name', )
        parser.add_argument('--src_train_num', type=int, default=33,
                            help='number of src_train_set', )  # src_train_num
        parser.add_argument('--trg_train_num', type=int, default=37,
                            help='the number of training data', )
        parser.add_argument('--pretrain_seg', type=bool, default=False,
                            help='whther use the pretrained segmentation model', )
        parser.add_argument('--pretrain_G', type=bool, default=True,
                            help='whther use the pretrained generator model', )#cut_low
        parser.add_argument('--cut_low', type=float, default=0.75,
                            help='low value of the random uniform', )
        parser.add_argument('--cut_high', type=float, default=0.85,
                            help='high value of the random uniform', )
        parser.add_argument('--pretrained_seg_path', type=str, default='../../checkpoints/Baseline_UNet/day26_1e-4_roicropDSA/model_best.pth',
                            help='path of the pretrained segmentation model', )
        parser.add_argument('--pretrained_G_path', type=str, default='/cta-seg/checkpoints/Image_translate/day10_image_transfer_BN_dropout_nolsGAN_diflr_Gfirst/experiment_name/latest_net_G.pth',
                            help='path of the pretrained generator model', )
        parser.add_argument('--pretrained_D_path', type=str, default='/cta-projects-main/cta-seg/checkpoints/Image_translate/day10_image_transfer_BN_dropout_nolsGAN_diflr_Gfirst/experiment_name/latest_net_D.pth',
                            help='path of the pretrained segmentation model', )
        parser.add_argument('--semmantic_lossname', type=str,
                            default='l1_loss',
                            help='name of the semantic loss', )
        parser.add_argument('--parallel_train', type=bool, default=True,
                            help='whther train in paralleled way', )#lamda_consistency
        parser.add_argument('--lamda_consistency', type=int, default=0,
                            help='weight of consistency loss', )#is_tr
        parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
        parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
        parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
        parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')#consis_lossname
        parser.add_argument('--consis_lossname', type=str, default="kl_loss", help='name of consistency loss')
        parser.add_argument('--G_first', default=True, help='whether update generator first')
        parser.add_argument('--Pre_train', default=False, help='whether use the pretrained G and D')#norm_type='max-min'
        parser.add_argument('--norm_type', type=str,default='z-score', help='z-score,max-min or max-abs,for normalization')
        # parser.add_argument('--lambda_L1', type=float, default=1, help='weight for L1 loss')
        parser.add_argument('--gan_mode', type=str, default='lsgan',
                            help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')

        self.isTrain = True
        return parser


