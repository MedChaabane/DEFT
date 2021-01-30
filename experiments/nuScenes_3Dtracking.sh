cd src
# train 
python train.py tracking,ddd --exp_id nuScenes_3Dtracking --dataset nuscenes --pre_hm --shift 0.01 --scale 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --hm_disturb 0.05 --batch_size 1 --gpus 0 --lr 2.5e-4 --save_point 80
# train the motion model
python train_prediction.py tracking,ddd --exp_id nuScenes_3Dtracking_motion_model --dataset nuscenes --batch_size 4 --gpus 0 --lr 2.5e-4
# test
python test.py tracking,ddd --exp_id nuScenes_3Dtracking --dataset nuscenes --load_model models/model_nuscenes.pth --load_model_traj models/model_nuscenes_lstm.pth