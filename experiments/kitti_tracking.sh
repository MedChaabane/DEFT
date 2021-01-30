cd src
# train the model
python train.py tracking --exp_id kitti_train --dataset kitti_tracking --dataset_version train --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0 --batch_size 1 --load_model models/model_kitti.pth
# train the motion model
python train_prediction.py tracking --exp_id kitti_motion_model --dataset kitti_tracking --dataset_version train --gpus 0 --batch_size 1
# test
python test.py tracking --exp_id kitti_fulltrain --dataset kitti_tracking --dataset_version test --pre_hm --track_thresh 0.4 --load_model models/model_kitti.pth --load_model_traj models/model_kitti_lstm.pth
