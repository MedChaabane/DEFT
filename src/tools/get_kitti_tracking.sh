# first download data_tracking_calib.zip , data_tracking_image_2.zip, data_tracking_label_2.zip and place them in ../../data/kitti_tracking
mkdir ../../data/kitti_tracking/data_tracking_calib
mkdir ../../data/kitti_tracking/label_02
mkdir ../../data/kitti_tracking/data_tracking_image_2
unzip ../../data/kitti_tracking/data_tracking_calib.zip -d ../../data/kitti_tracking/data_tracking_calib
unzip ../../data/kitti_tracking/data_tracking_image_2.zip -d ../../data/kitti_tracking/data_tracking_image_2
unzip ../../data/kitti_tracking/data_tracking_label_2.zip -d ../../data/kitti_tracking/label_02
mv ../../data/kitti_tracking/label_02/training/label_02/* ../../data/kitti_tracking/label_02/
python convert_kittitrack_to_coco.py
