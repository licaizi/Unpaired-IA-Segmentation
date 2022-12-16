import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.system('python ../multiModality/multi_modality_baseline.py --fold=1  --data_type="CTA" --batch_size=2 --multi_size=0 '
'--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --epoches=60 --beta=1.0 --nonlin=2 --norm_op=1 --lamda=0.5 '
'--log_path="../../logs/multiModality/multi_mod_baseline_IN_fold{}" '
'--checkpoint_path="../../checkpoints/multiModality/multi_mod_baseline_IN_fold{}" '
'--summary_writer="../../logs/MultiModality/multi_mod_baseline_IN_fold{}" ')

os.system('python ../../main/baseline_train/UNet_Train.py --fold=1 --labeled_num=20 --batch_size=2 '
'--initial_lr=1e-3 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --epoches=50 --nonlin=2 --norm_op=1 '
'--log_path="../../logs/baseline_unet/UNet_train{}" '
'--checkpoint_path="../../checkpoints/baseline_unet/UNet_train{}" '
'--summary_writer="../../logs/baseline_unet/UNet_train{}" ')