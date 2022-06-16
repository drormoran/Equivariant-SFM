# Deep Permutation Equivariant Structure from Motion <br>

### [Paper](https://arxiv.org/abs/2104.06703) | [Poster](/media/ICCV_Slide.pdf)
<p align="center">
  <img width="100%" src="media/four_optimization_results.gif"/>
</p>

This repository contains an implementation for the ICCV 2021 paper <a href="https://arxiv.org/abs/2104.06703">Deep Permutation Equivariant Structure from Motion</a>.

The paper proposes a neural network architecture that, given a set of point tracks in multiple images of a static scene, recovers both the camera parameters and a (sparse) scene structure by minimizing an unsupervised reprojection loss. The method does not require initialization of camera parameters or 3D point locations and is implemented for two setups: (1) single scene reconstruction and (2) learning from multiple scenes.



### Table of Contents

- [Setup](#Setup)
- [How to use](#How-to-use)
- [Citation](#Citation)

---

## Setup
This repository is implemented with python 3.8, and in order to run bundle adjustment requires linux.

### Folders
The repository should contain the following folders:
```
Equivariant-SFM
├── bundle_adjustment
├── code
├── datasets
│   ├── Euclidean
│   └── Projective
├── environment.yml
├── results
```

### Conda envorinment
Create the environment using one of the following commands:

```
conda create -n ESFM -c pytorch -c conda-forge -c comet_ml -c plotly  -c fvcore -c iopath -c bottler -c anaconda -c pytorch3d python=3.8 pytorch cudatoolkit=10.2 torchvision pyhocon comet_ml plotly pandas opencv openpyxl xlrd cvxpy fvcore iopath nvidiacub pytorch3d eigen cmake glog gflags suitesparse gxx_linux-64 gcc_linux-64 dask matplotlib
conda activate ESFM
```

Or:

```
conda env create -f environment.yml
conda activate ESFM
```

And follow the <a href="bundle_adjustment/README.md">bundle adjustment instructions</a>.

### Data
Download the data from this <a href="https://www.dropbox.com/sh/s2714jqsstwp9uc/AAAhFdqDoyK0naDG7eA6dd3Ra?dl=0">link</a>.

The model can work on both calibrated camera setting (euclidean reconstruction) *and* on uncalibrated cameras (projective reconstruction).

The input for the model is an observed points matrix of size ```[m,n,2]``` where the entry ```[i,j]``` is a 2D image point that corresponds to camera (image) number ```i``` and 3D point (point track) number ```j```.

In practice we use a correspondence matrix representation of size ```[2*m,n]```, where the entries ```[2*i,j]``` and ```[2*i+1,j]``` form the ```[i,j]``` image point.

For the calibrated setting, the input must include ```m``` calibration matrices of size ```[3,3]```.

## How to use


### Optimization
For a calibrated scene optimization run:
```
python single_scene_optimization.py --conf Optimization_Euc.conf
```
For an uncalibrated scene optimization run:
```
python single_scene_optimization.py --conf Optimization_Proj.conf
```
The following examples are for the calibrated settings but are clearly the same for the uncalibrated setting.

You can choose which scene to optimize either by changing the config file in the field 'dataset.scan' or from the command line:
```
python single_scene_optimization.py --conf Optimization_Euc.conf --scan [scan_name]
```
Similarly, you can override any value of the config file from the command line. For example, to change the number of training epochs and the evaluation frequency use:
```
python single_scene_optimization.py --conf Optimization_Euc.conf --external_params "train:num_of_epochs:1e+5,train:eval_intervals:100"
```


### Learning
To run the learning setup run:
```
python multiple_scenes_learning.py --conf Learning_Euc.conf
```
Or for the uncalibrated setting:
```
python multiple_scenes_learning.py --conf Learning_Proj.conf
```
To override some parameters from the config file, you can either change the file itself or use the same command as in the optimization setting:
```
python multiple_scenes_learning.py --conf Learning_Euc.conf --external_params "train:num_of_epochs:1e+5,train:eval_intervals:100"
```


## Citation
If you find this work useful please cite:

```
@InProceedings{Moran_2021_ICCV,
    author    = {Moran, Dror and Koslowsky, Hodaya and Kasten, Yoni and Maron, Haggai and Galun, Meirav and Basri, Ronen},
    title     = {Deep Permutation Equivariant Structure From Motion},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {5976-5986}
}
```
