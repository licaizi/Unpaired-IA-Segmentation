import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # os.system('python ../Baseline_UNet/save_predict.py --fold=2 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# # '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# # '--pretrained_model="../../checkpoints/MultiModality/multi_mod_allother_region_contrast_key64_fold{}/model_best.pth" '
# # '--saved_root="../../results/MultiModality/multi_mod_allother_region_contrast_key64_fold{}/" '
# # '--summary_writer="../../logs/MultiModality/test_multi_modality_allother_region_contrast_keys64_fold{}" ')
# # os.system('python ../Baseline_UNet/save_predict.py --fold=0 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# # '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# # '--pretrained_model="../../checkpoints/MultiModality/multi_modality_intermodal_crossLMI_region_contrast_v1_fold{}/model_best.pth" '
# # '--saved_root="../../results/MultiModality/multi_modality_intermodal_crossLMI_region_contrast_v1_fold{}/" '
# # '--summary_writer="../../logs/MultiModality/test_multi_modality_intermodal_crossLMI_region_contrast_v1_fold{}" ')
# #
# # os.system('python ../Baseline_UNet/save_predict.py --fold=1 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# # '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# # '--pretrained_model="../../checkpoints/MultiModality/multi_modality_intermodal_crossLMI_region_contrast_v1_fold{}/model_best.pth" '
# # '--saved_root="../../results/MultiModality/multi_modality_intermodal_crossLMI_region_contrast_v1_fold{}/" '
# # '--summary_writer="../../logs/MultiModality/test_multi_modality_intermodal_crossLMI_region_contrast_v1_fold{}" ')
# #
# # os.system('python ../Baseline_UNet/save_predict.py --fold=2 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# # '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# # '--pretrained_model="../../checkpoints/MultiModality/multi_modality_intermodal_crossLMI_region_contrast_v1_fold{}/model_best.pth" '
# # '--saved_root="../../results/MultiModality/multi_modality_intermodal_crossLMI_region_contrast_v1_fold{}/" '
# # '--summary_writer="../../logs/MultiModality/test_multi_modality_intermodal_crossLMI_region_contrast_v1_fold{}" ')
# # os.system('python ../Baseline_UNet/save_predict.py --fold=3 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# # '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# # '--pretrained_model="../../checkpoints/MultiModality/multi_modality_intermodal_crossLMI_region_contrast_v1_fold{}/model_best.pth" '
# # '--saved_root="../../results/MultiModality/multi_modality_intermodal_crossLMI_region_contrast_v1_fold{}/" '
# # '--summary_writer="../../logs/MultiModality/test_multi_modality_intermodal_crossLMI_region_contrast_v1_fold{}" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=0 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/" '
# '--summary_writer="../../logs/Baseline_UNet/test_UNet_Train_CTA_filteredData_fold{}" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=1 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/" '
# '--summary_writer="../../logs/Baseline_UNet/test_UNet_Train_CTA_filteredData_fold{}" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=2 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/" '
# '--summary_writer="../../logs/Baseline_UNet/test_UNet_Train_CTA_filteredData_fold{}" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=3 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/" '
# '--summary_writer="../../logs/Baseline_UNet/test_UNet_Train_CTA_filteredData_fold{}" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=4 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/" '
# '--summary_writer="../../logs/Baseline_UNet/test_UNet_Train_CTA_filteredData_fold{}" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=0 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/newmulti_modality_joint_and_intra_region_contrast_v1_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/newmulti_modality_joint_and_intra_region_contrast_v1_fold{}/" '
# '--summary_writer="../../logs/MultiModality/test_newmulti_modality_joint_and_intra_region_contrast_v1_fold{}" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=1 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/newmulti_modality_joint_and_intra_region_contrast_v1_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/newmulti_modality_joint_and_intra_region_contrast_v1_fold{}/" '
# '--summary_writer="../../logs/MultiModality/test_newmulti_modality_joint_and_intra_region_contrast_v1_fold{}" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=2 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/newmulti_modality_joint_and_intra_region_contrast_v1_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/newmulti_modality_joint_and_intra_region_contrast_v1_fold{}/" '
# '--summary_writer="../../logs/MultiModality/test_newmulti_modality_joint_and_intra_region_contrast_v1_fold{}" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=3 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/newmulti_modality_joint_and_intra_region_contrast_v1_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/newmulti_modality_joint_and_intra_region_contrast_v1_fold{}/" '
# '--summary_writer="../../logs/MultiModality/test_newmulti_modality_joint_and_intra_region_contrast_v1_fold{}" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=4 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/newmulti_modality_joint_and_intra_region_contrast_v1_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/newmulti_modality_joint_and_intra_region_contrast_v1_fold{}/" '
# '--summary_writer="../../logs/MultiModality/test_newmulti_modality_joint_and_intra_region_contrast_v1_fold{}" ')
#
# os.system('python ../Baseline_UNet/save_predict.py --fold=0 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/UNet_intra_modal_contrast_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/UNet_intra_modal_contrast_filterCTA_fold{}/" '
# '--summary_writer="../../logs/Baseline_UNet/test_UNet_intra_modal_contrast_filterCTA_fold{}" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=1 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/UNet_intra_modal_contrast_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/UNet_intra_modal_contrast_filterCTA_fold{}/" '
# '--summary_writer="../../logs/Baseline_UNet/test_UNet_intra_modal_contrast_filterCTA_fold{}" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=2 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/UNet_intra_modal_contrast_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/UNet_intra_modal_contrast_filterCTA_fold{}/" '
# '--summary_writer="../../logs/Baseline_UNet/test_UNet_intra_modal_contrast_filterCTA_fold{}" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=3 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/UNet_intra_modal_contrast_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/UNet_intra_modal_contrast_filterCTA_fold{}/" '
# '--summary_writer="../../logs/Baseline_UNet/test_UNet_intra_modal_contrast_filterCTA_fold{}" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=4 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/UNet_intra_modal_contrast_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/UNet_intra_modal_contrast_filterCTA_fold{}/" '
# '--summary_writer="../../logs/Baseline_UNet/test_UNet_intra_modal_contrast_filterCTA_fold{}" ')
#
# # os.system('python ../Baseline_UNet/save_predict.py --fold=0 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# # '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# # '--pretrained_model="../../checkpoints/MultiModality/newmulti_modality_joint_and_intra_region_contrast_v1_fold{}/model_best.pth" '
# # '--saved_root="../../results/MultiModality/multi_modality_yshape_filterCTA_fold{}/" '
# # '--summary_writer="../../logs/MultiModality/test_newmulti_modality_joint_and_intra_region_contrast_v1_fold{}" ')
# os.system('python ../Baseline_UNet/save_yshape_predict.py --fold=0 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_yshape_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/multi_modality_yshape_filterCTA_fold{}/" '
# '--summary_writer="../../logs/MultiModality/test_multi_modality_yshape_filterCTA_fold{}" ')
# os.system('python ../Baseline_UNet/save_yshape_predict.py --fold=1 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_yshape_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/multi_modality_yshape_filterCTA_fold{}/" '
# '--summary_writer="../../logs/MultiModality/test_multi_modality_yshape_filterCTA_fold{}" ')
# os.system('python ../Baseline_UNet/save_yshape_predict.py --fold=2 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_yshape_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/multi_modality_yshape_filterCTA_fold{}/" '
# '--summary_writer="../../logs/MultiModality/test_multi_modality_yshape_filterCTA_fold{}" ')
# os.system('python ../Baseline_UNet/save_yshape_predict.py --fold=0 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_yshape_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_yshape_filterCTA_fold{}/" ')
# os.system('python ../Baseline_UNet/save_yshape_predict.py --fold=1 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_yshape_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_yshape_filterCTA_fold{}/" ')
# os.system('python ../Baseline_UNet/save_yshape_predict.py --fold=2 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_yshape_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_yshape_filterCTA_fold{}/" ')
# os.system('python ../Baseline_UNet/save_yshape_predict.py --fold=3 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_yshape_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_yshape_filterCTA_fold{}/" ')
# os.system('python ../Baseline_UNet/save_yshape_predict.py --fold=4 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_yshape_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_yshape_filterCTA_fold{}/" ')
os.system('python ../Baseline_UNet/save_predict.py --fold=0 --labeled_num=20 --data_type="CTA" --batch_size=2 '
'--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
'--pretrained_model="../../checkpoints/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
'--saved_root="../../results/Baseline_UNet/new_UNet_Train_CTA_filteredData_fold{}/" ')
os.system('python ../Baseline_UNet/save_predict.py --fold=1 --labeled_num=20 --data_type="CTA" --batch_size=2 '
'--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
'--pretrained_model="../../checkpoints/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
'--saved_root="../../results/Baseline_UNet/new_UNet_Train_CTA_filteredData_fold{}/" ')
os.system('python ../Baseline_UNet/save_predict.py --fold=2 --labeled_num=20 --data_type="CTA" --batch_size=2 '
'--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
'--pretrained_model="../../checkpoints/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
'--saved_root="../../results/Baseline_UNet/new_UNet_Train_CTA_filteredData_fold{}/" ')
os.system('python ../Baseline_UNet/save_predict.py --fold=3 --labeled_num=20 --data_type="CTA" --batch_size=2 '
'--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
'--pretrained_model="../../checkpoints/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
'--saved_root="../../results/Baseline_UNet/new_UNet_Train_CTA_filteredData_fold{}/" ')
os.system('python ../Baseline_UNet/save_predict.py --fold=4 --labeled_num=20 --data_type="CTA" --batch_size=2 '
'--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
'--pretrained_model="../../checkpoints/Baseline_UNet/UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
'--saved_root="../../results/Baseline_UNet/new_UNet_Train_CTA_filteredData_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=1 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_dsbn_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_dsbn_filterCTA_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=2 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_dsbn_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_dsbn_filterCTA_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=3 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_dsbn_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_dsbn_filterCTA_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=4 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_dsbn_filterCTA_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_dsbn_filterCTA_fold{}/" ')
#
# os.system('python ../Baseline_UNet/save_predict.py --fold=0 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_dsbn_kd_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_dsbn_kd_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=1 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_dsbn_kd_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_dsbn_kd_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=2 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_dsbn_kd_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_dsbn_kd_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=3 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_dsbn_kd_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_dsbn_kd_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=4 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_dsbn_kd_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_dsbn_kd_fold{}/" ')

# os.system('python ../Baseline_UNet/save_predict.py --fold=0 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_baselineBN_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_baselineBN_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=1 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_baselineBN_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_baselineBN_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=2 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_baselineBN_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_baselineBN_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=3 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_baselineBN_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_baselineBN_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=4 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/MultiModality/multi_modality_baselineBN_fold{}/model_best.pth" '
# '--saved_root="../../results/MultiModality/new_multi_modality_baselineBN_fold{}/" ')
#
# os.system('python ../Baseline_UNet/save_predict.py --fold=0 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/BN_UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/new_BN_UNet_Train_CTA_filteredData_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=1 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/BN_UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/new_BN_UNet_Train_CTA_filteredData_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=2 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/BN_UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/new_BN_UNet_Train_CTA_filteredData_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=3 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/BN_UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/new_BN_UNet_Train_CTA_filteredData_fold{}/" ')
# os.system('python ../Baseline_UNet/save_predict.py --fold=4 --labeled_num=20 --data_type="CTA" --batch_size=2 '
# '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --region_size=16 --epoches=60 --nonlin=2 --norm_op=1 '
# '--pretrained_model="../../checkpoints/Baseline_UNet/BN_UNet_Train_CTA_filteredData_fold{}/model_best.pth" '
# '--saved_root="../../results/Baseline_UNet/new_BN_UNet_Train_CTA_filteredData_fold{}/" ')
