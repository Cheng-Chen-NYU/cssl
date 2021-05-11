# Contrastive Visual Representation Learning
Course Project for ECE-GY 9123 Deep Learning Spring 2021

Requirements

```python
pip install torch
pip install torchvision
pip install tqdm
pip install thop
```
One GPU for MoCo
```python
python train_model.py --model_name mocov1 --batch_size 512 --epochs 200 --arch resnet18 --learning_rate 0.06 --temperature 0.1 --weight_decay 5e-4

python train_model.py --model_name mocov2 --batch_size 512 --epochs 200 --arch resnet18 --learning_rate 0.06 --temperature 0.1 --weight_decay 5e-4

python linear.py --batch_size 512 --epochs 100
```
Four GPUs for SimCLR
```python
python train_model.py --model_name simclrv1 --batch_size 512 --epochs 500 --arch resnet50 —learning_rate 1e-3 --temperature 0.5 --weight_decay 1e-6

python train_model.py --model_name simclrv2 --batch_size 512 --epochs 500 --arch resnet50 —learning_rate 1e-3 --temperature 0.5 --weight_decay 1e-6

python linear.py --batch_size 512 --epochs 100
```

[MoCo](https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb)
[SimCLR](https://github.com/leftthomas/SimCLR)