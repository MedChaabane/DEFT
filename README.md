# DEFT

DEFT: Detection Embeddings for Tracking


Contact: [chaabane@colostate.edu](mailto:chaabane@colostate.edu). Any questions or discussion are welcome! 

## Abstract
Most modern multiple object tracking (MOT) systems follow the tracking-by-detection paradigm, consisting of a detector followed by a method for associating detections into tracks. There is a long history in tracking of combining motion and appearance features to provide robustness to occlusions and other challenges, but typically this comes with the trade-off of a more complex and slower implementation. Recent successes on popular 2D tracking benchmarks indicate that top-scores can be achieved using a state-of-the-art detector and relatively simple associations relying on single-frame spatial offsets -- notably outperforming contemporary methods that leverage learned appearance features to help re-identify lost tracks. In this paper, we propose an efficient joint detection and tracking model named DEFT, or Detection Embeddings for Tracking. Our approach relies on an appearance-based object matching network jointly-learned with an underlying object detection network. An LSTM is also added to capture motion constraints. DEFT has comparable accuracy and speed to the top methods on 2D online tracking leaderboards while having significant advantages in robustness when applied to more challenging tracking data. DEFT raises the bar on the nuScenes monocular 3D tracking challenge, more than doubling the performance of the previous top method.

## Installation
* Clone this repo, and run the following commands.
* create a new conda environment and activate the environment.
```
git clone git@github.com:MedChaabane/DEFT.git
cd DEFT
conda create -y -n DEFT python=3.7
conda activate DEFT
```
* Install PyTorch and the dependencies.
```
conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt  
```
* Install [COCOAPI](https://github.com/cocodataset/cocoapi):
```
pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
* Compile [DCNv2](https://github.com/CharlesShang/DCNv2)
```
cd src/lib/model/networks/
git clone https://github.com/CharlesShang/DCNv2
cd DCNv2
./make.sh
```
* Download the [pretrained models](https://drive.google.com/drive/folders/1dlVoV-4fMYlttdj2ba0unn6WX-nxaC48?usp=sharing) and move them to src/models/ . 

