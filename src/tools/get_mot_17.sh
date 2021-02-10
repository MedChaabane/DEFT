mkdir -p ../../data/mot17
cd ../../data/mot17
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip
rm MOT17.zip
mkdir annotations
cd ../../src/tools/
python convert_mot_to_coco.py
python convert_mot_det_to_results.py
mv ../../data/mot17/MOT17/* ../../data/mot17/
rm -r ../../data/mot17/MOT17
