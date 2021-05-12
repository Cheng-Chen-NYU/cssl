# Contrastive Visual Representation Learning
- Course Project for ECE-GY 9123 Deep Learning Spring 2021

- Contrastive self-supervised methods MoCo and SimCLR on CIFAR10

*Implementation adapted from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb and https://github.com/leftthomas/SimCLR.*

```
git clone https://github.com/ChengChen2020/cvrl
```

Requirements

```python
numpy==1.20.2
pandas==1.2.4
Pillow==8.2.0
thop==0.0.31-2005241907
torch==1.8.1
torchvision==0.9.1
# tqdm==4.60.0
```
One GPU for MoCo
```python
python train_model.py --model_name mocov1 --batch_size 512 --epochs 200 --arch resnet18 --learning_rate 0.06 --temperature 0.1 --weight_decay 5e-4

python train_model.py --model_name mocov2 --batch_size 512 --epochs 200 --arch resnet18 --learning_rate 0.06 --temperature 0.1 --weight_decay 5e-4

python linear.py --model_name <xxx> --model_path <xxx> --batch_size 512 --epochs 100
```
Four GPUs for SimCLR
```python
python train_model.py --model_name simclrv1 --batch_size 512 --epochs 500 --arch resnet50 --learning_rate 1e-3 --temperature 0.5 --weight_decay 1e-6

python train_model.py --model_name simclrv2 --batch_size 512 --epochs 500 --arch resnet50 --learning_rate 1e-3 --temperature 0.5 --weight_decay 1e-6

python linear.py --model_name <xxx> --model_path <xxx> --batch_size 512 --epochs 100
```

#### References

[^1]: Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. CVPR, arXiv:1911.05722, 2019.
[^2]: Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. ICML, arXiv:2002.05709, 2020.
[^3]: Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv:2003.04297, 2020.
[^4]: Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey Hinton. Big self-supervised models are strong semi-supervised learners. NeurIPS, arXiv:2006.10029, 2020.