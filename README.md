# KD-template
This repository implements the most basic knowledge distillation. Since no template of knowledge distillation satisfies me, I make one for myself.

## Main requirements

This repo is tested with:

- python==3.7.9
- CUDA==10.1
- torch==1.7.0
- torchvision==0.8.1

But it should be runnable with other PyTorch versions.

## Models & Datasets

I use a Resnet-50 as a teacher net and an AlexNet as a student net. The dataset is CIFAR-10. 