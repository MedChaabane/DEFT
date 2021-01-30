cd src
# train
python train.py tracking --exp_id mot17_train --dataset mot --dataset_version 17trainval --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0 --load_model ../models/mot17_fulltrain.pth
# train the motion model
python train_prediction.py tracking --exp_id mot17_motion_model --dataset mot --dataset_version 17trainval --gpus 0
# test
python test.py tracking --exp_id mot17_train --dataset mot --dataset_version 17test --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model models/model_mot.pth