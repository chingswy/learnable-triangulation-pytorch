# 预处理

```bash
git clone https://github.com/anibali/h36m-fetch.git
cd h36m-fetch
python3 process_all.py
cd ..
cd mvn/datasets/human36m_preprocessing
python3 collect-bboxes.py $THIS_REPOSITORY/data/human36m 32 # 多线程
```

对图像畸变矫正：

```bash
python3 undistort-h36m.py $THIS_REPOSITORY/data/human36m $THIS_REPOSITORY/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy 32
```


# 测试：

```bash
python3 train.py \
  --eval --eval_dataset val \
  --config experiments/human36m/eval/human36m_alg.yaml \
  --logdir ./logs
```

他需要这个东西的结果，获得root点的估计，所以要跑完这个才能跑vol的

```bash
python3 train.py \
  --eval --eval_dataset val \
  --config experiments/human36m/eval/human36m_vol_softmax.yaml \
  --logdir ./logs
```

这个配置文件里有个log路径，需要改一下

TODO：

- [ ] 等待alg的跑完，看一眼alg的结果，然后再跑起来vol的
- [ ] 把这个resnet152搬到Easy Pose的代码里，对比一下HRNet的
- [ ] 在EasyPose里读他的dataset
  - `easypose/dataset/top_down/top_down.py`, Line 238, 这里增加了一个type
  - `easypose/datasest/bottom_up/bottom_up.py`, Line 140, 这里仿照`_load_info_from_easymocap`增加一个函数，只有保持内容一致就行


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/190505754/3d-human-pose-estimation-on-human36m)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-human36m?p=190505754)

# Learnable Triangulation of Human Pose
This repository is an official PyTorch implementation of the paper ["Learnable Triangulation of Human Pose"](https://arxiv.org/abs/1905.05754) (ICCV 2019, oral). Here we tackle the problem of 3D human pose estimation from multiple cameras. We present 2 novel methods — Algebraic and Volumetric learnable triangulation — that **outperform** previous state of the art.

If you find a bug, have a question or know to improve the code - please open an issue!

:arrow_forward: [ICCV 2019 talk](https://youtu.be/zem03fZWLrQ?t=3477)

<p align="center">
  <a href="http://www.youtube.com/watch?v=z3f3aPSuhqg">
    <img width=680 src="docs/video-preview.jpg">
  </a>
</p>

# How to use
This project doesn't have any special or difficult-to-install dependencies. All installation can be done with:
```bash
pip install -r requirements.txt
```

## Data
Sorry, only [Human3.6M](http://vision.imar.ro/human3.6m/description.php) dataset training/evaluation is available right now. We cannot add [CMU Panoptic](http://domedb.perception.cs.cmu.edu/), sorry for that.

#### Human3.6M
1. Download and preprocess the dataset by following the instructions in [mvn/datasets/human36m_preprocessing/README.md](https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/datasets/human36m_preprocessing/README.md).
2. Download pretrained backbone's weights from [here](https://disk.yandex.ru/d/hv-uH_7TY0ONpg) and place them here: `./data/pretrained/human36m/pose_resnet_4.5_pixels_human36m.pth` (ResNet-152 trained on COCO dataset and finetuned jointly on MPII and Human3.6M).
3. If you want to train Volumetric model, you need rough estimations of the pelvis' 3D positions both for train and val splits. In the paper we estimate them using the Algebraic model. You can use the [pretrained](#model-zoo) Algebraic model to produce predictions or just take [precalculated 3D skeletons](#model-zoo).

## Model zoo
In this section we collect pretrained models and configs. All **pretrained weights** and **precalculated 3D skeletons** can be downloaded at once [from here](https://disk.yandex.ru/d/jbsyN8XVzD-Y7Q) and placed to `./data/pretrained`, so that eval configs can work out-of-the-box (without additional setting of paths). Alternatively, the table below provides separate links to those files.

**Human3.6M:**

| Model                | Train config                                                                                                                                                  | Eval config                                                                                                                                                 | Weights                                                                                    | Precalculated results                                                                                                                                  | MPJPE (relative to pelvis), mm |
|----------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------:|-------------------------------:|
| Algebraic            |         [train/human36m_alg.yaml](https://github.com/karfly/learnable-triangulation-pytorch/blob/master/experiments/human36m/train/human36m_alg.yaml)         |         [eval/human36m_alg.yaml](https://github.com/karfly/learnable-triangulation-pytorch/blob/master/experiments/human36m/eval/human36m_alg.yaml)         | [link](https://disk.yandex.ru/d/3TJMKaa6iKaymw) | [train](https://disk.yandex.ru/d/2Gwk7JZ_QWpFvw), [val](https://disk.yandex.ru/d/ZsQ4GV5EX_Wsog)  | 22.5                           |
| Volumetric (softmax) | [train/human36m_vol_softmax.yaml](https://github.com/karfly/learnable-triangulation-pytorch/blob/master/experiments/human36m/train/human36m_vol_softmax.yaml) | [eval/human36m_vol_softmax.yaml](https://github.com/karfly/learnable-triangulation-pytorch/blob/master/experiments/human36m/eval/human36m_vol_softmax.yaml) | [link](https://disk.yandex.ru/d/MvD3orcBc6wqRA) |                                               —                                                                                                        | **20.4**                       |

## Train
Every experiment is defined by `.config` files. Configs with experiments from the paper can be found in the `./experiments` directory (see [model zoo](#model-zoo)).

#### Single-GPU
To train a Volumetric model with softmax aggregation using **1 GPU**, run:
```bash
python3 train.py \
  --config experiments/human36m/train/human36m_vol_softmax.yaml \
  --logdir ./logs
```

The training will start with the config file specified by `--config`, and logs (including tensorboard files) will be stored in `--logdir`.

#### Multi-GPU (*in testing*)
Multi-GPU training is implemented with PyTorch's [DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#distributeddataparallel). It can be used both for single-machine and multi-machine (cluster) training. To run the processes use the PyTorch [launch utility](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py).

To train a Volumetric model with softmax aggregation using **2 GPUs on single machine**, run:
```bash
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=2345 \
  train.py  \
  --config experiments/human36m/train/human36m_vol_softmax.yaml \
  --logdir ./logs
```

## Tensorboard
To watch your experiments' progress, run tensorboard:
```bash
tensorboard --logdir ./logs
```

## Evaluation
After training, you can evaluate the model. Inside the same config file, add path to the learned weights (they are dumped to `logs` dir during training):
```yaml
model:
    init_weights: true
    checkpoint: {PATH_TO_WEIGHTS}
```

Also, you can change other config parameters like `retain_every_n_frames_test`.

Run:
```bash
python3 train.py \
  --eval --eval_dataset val \
  --config experiments/human36m/eval/human36m_vol_softmax.yaml \
  --logdir ./logs
```
Argument `--eval_dataset` can be `val` or `train`. Results can be seen in `logs` directory or in the tensorboard.

# Results
* We conduct experiments on two available large multi-view datasets: Human3.6M [\[2\]](#references) and CMU Panoptic [\[3\]](#references).
* The main metric is **MPJPE** (Mean Per Joint Position Error) which is L2 distance averaged over all joints.

## Human3.6M
* We significantly improved upon the previous state of the art (error is measured relative to pelvis, without alignment).
* Our best model reaches **17.7 mm** error in absolute coordinates, which was unattainable before.
* Our Volumetric model is able to estimate 3D human pose using **any number of cameras**, even using **only 1 camera**. In single-view setup, we get results comparable to current state of the art [\[6\]](#references) (49.9 mm vs. 49.6 mm).

<br>
MPJPE relative to pelvis:

|                             	|  MPJPE (averaged across all actions), mm	|
|-----------------------------	|:--------:	|
| Multi-View Martinez [\[4\]](#references)          |   57.0   	|
| Pavlakos et al. [\[8\]](#references)   	          |   56.9   	|
| Tome et al. [\[4\]](#references)                 	|   52.8   	|
| Kadkhodamohammadi & Padoy [\[5\]](#references)   	|   49.1   	|
| [Qiu et al.](https://github.com/microsoft/multiview-human-pose-estimation-pytorch) [\[9\]](#references)   	|   26.2   	|
| RANSAC (our implementation) 	|   27.4   	|
| **Ours, algebraic**          	|   22.4   	|
| **Ours, volumetric**         	| **20.5** 	|

<br>
MPJPE absolute (scenes with invalid ground-truth annotations are excluded):

|                             	|  MPJPE (averaged across all actions), mm 	|
|-----------------------------	|:--------:	|
| RANSAC (our implementation) 	|   22.8   	|
| **Ours, algebraic**          	|   19.2   	|
| **Ours, volumetric**         	| **17.7** 	|

<br>
MPJPE relative to pelvis (single-view methods):

|                             	| MPJPE (averaged across all actions), mm 	|
|-----------------------------	|:-----------------------------------:	|
| Martinez et al. [\[7\]](#references)                  	|                 62.9               	|
| Sun et al. [\[6\]](#references)                  	|                 **49.6**                	|
| **Ours, volumetric single view** 	|                 **49.9**                	|


## CMU Panoptic
* Our best model reaches **13.7 mm** error in absolute coordinates for 4 cameras
* We managed to get much smoother and more accurate 3D pose annotations compared to dataset annotations (see [video demonstration](http://www.youtube.com/watch?v=z3f3aPSuhqg))

<br>
MPJPE relative to pelvis [4 cameras]:

|                             	|  MPJPE, mm 	|
|-----------------------------	|:--------:	|
| RANSAC (our implementation) 	|   39.5   	|
| **Ours, algebraic**          	|   21.3   	|
| **Ours, volumetric**         	| **13.7** 	|

# Method overview
We present 2 novel methods of learnable triangulation: Algebraic and Volumetric.

## Algebraic
![algebraic-model](docs/algebraic-model.svg)

Our first method is based on Algebraic triangulation. It is similar to the previous approaches, but differs in 2 critical aspects:
1. It is **fully differentiable**. To achieve this, we use soft-argmax aggregation and triangulate keypoints via a differentiable SVD.
2. The neural network additionally predicts **scalar confidences for each joint**, passed to the triangulation module, which successfully deals with outliers and occluded joints.

For the most popular Human3.6M dataset, this method already dramatically reduces error by **2.2 times (!)**, compared to the previous art.


## Volumetric
![volumetric-model](docs/volumetric-model.svg)

In Volumetric triangulation model, intermediate 2D feature maps are densely unprojected to the volumetric cube and then processed with a 3D-convolutional neural network. Unprojection operation allows **dense aggregation from multiple views** and the 3D-convolutional neural network is able to model **implicit human pose prior**.

Volumetric triangulation additionally improves accuracy, drastically reducing the previous state-of-the-art error by **2.4 times!** Even compared to the best parallelly developed [method](https://github.com/microsoft/multiview-human-pose-estimation-pytorch) by MSRA group, our method still offers significantly lower error of **21 mm**.

<p align="center">
  <img src="docs/unprojection.gif">
</p>


# Cite us!
```bibtex
@inproceedings{iskakov2019learnable,
  title={Learnable Triangulation of Human Pose},
  author={Iskakov, Karim and Burkov, Egor and Lempitsky, Victor and Malkov, Yury},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

# Contributors
 - [Karim Iskakov](https://github.com/karfly)
 - [Egor Burkov](https://github.com/shrubb)
 - [Victor Lempitsky](https://scholar.google.com/citations?user=gYYVokYAAAAJ&hl=ru)
 - [Yury Malkov](https://github.com/yurymalkov)
 - [Rasul Kerimov](https://github.com/rrkarim)
 - [Ivan Bulygin](https://github.com/blufzzz)

# News
- **26 Nov 2019:** Updataed [precalculated results](#model-zoo) (see [this issue](https://github.com/karfly/learnable-triangulation-pytorch/issues/37)).
- **18 Oct 2019:** Pretrained models (algebraic and volumetric) for Human3.6M are released.
- **8 Oct 2019:** Code is released!

# References
* [\[1\]](#references) R. Hartley and A. Zisserman. **Multiple view geometry in computer vision**.
* [\[2\]](#references) C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu. **Human3.6m: Large scale datasets and predictive methods for 3d human sensing in natural environments**.
* [\[3\]](#references) H. Joo, T. Simon, X. Li, H. Liu, L. Tan, L. Gui, S. Banerjee, T.  S. Godisart, B. Nabbe, I. Matthews, T. Kanade,S. Nobuhara, and Y. Sheikh. **Panoptic studio: A massively multiview system for social interaction capture**.
* [\[4\]](#references) D. Tome, M. Toso, L. Agapito, and C. Russell.  **Rethinking Pose in 3D: Multi-stage Refinement and Recovery for Markerless Motion Capture**.
* [\[5\]](#references) A. Kadkhodamohammadi and N. Padoy. **A generalizable approach for multi-view 3D human pose regression**.
* [\[6\]](#references) X. Sun, B. Xiao, S. Liang, and Y. Wei. **Integral human pose regression**.
* [\[7\]](#references) J. Martinez, R. Hossain, J. Romero, and J. J. Little. **A simple yet effective baseline for 3d human pose estimation**.
* [\[8\]](#references) G. Pavlakos, X. Zhou, K. G. Derpanis, and  K. Daniilidis. **Harvesting multiple views for marker-less 3D human pose annotations**.
* [\[9\]](#references) H. Qiu, C. Wang, J. Wang, N. Wang and W. Zeng. (2019). **Cross View Fusion for 3D Human Pose Estimation**, [GitHub](https://github.com/microsoft/multiview-human-pose-estimation-pytorch)
