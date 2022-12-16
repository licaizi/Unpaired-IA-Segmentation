import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.system('python ../baseline_train/UNet_test.py --fold=0 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --epoches=60 --nonlin=2 --norm_op=1 '
# '--log_path="../../logs/baseline_unet/test_multi_modality_regioncontrast_fold{}" '
# '--pretrained_model="../../checkpoints/baseline_unet/region_contrast_CTA_IN_fold{}/model_best.pth" '
# '--checkpoint_path="../../checkpoints/baseline_unet/test_multi_modality_regioncontrast_fold{}" '
# '--summary_writer="../../logs/baseline_unet/test_multi_modality_regioncontrast_fold{}" ')