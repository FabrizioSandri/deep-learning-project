# ClipRPN - Deep Learning Project

This repository contains the implementation of ClipRPN, a visual grounding
framework developed as a project for the Deep Learning course at the University
of Trento.

## Requirements
Before getting started, make sure you have the following dependencies installed:
torch, torchvision, and [CLIP](https://github.com/openai/CLIP). Additionally, if
you plan to use the Baseline module, which is based on Yolo, you'll need to
install its dependencies as well.

To install the CLIP dependencies, run the following commands:
```
$ pip install ftfy regex tqdm torchmetrics torchvision
$ pip install git+https://github.com/openai/CLIP.git
```

To install the Yolo dependencies, use the following command:
```
$ pip install -U ultralytics
```

## Architecture
A high-level representation of our proposed framework. The architecture begins
by converting an image and a textual description into a feature map and text
embedding. This is achieved using the ResNet50 of CLIP (excluding the last
attention layer) for the image and the text encoder of CLIP for the textual
input. The feature map and text embedding are then jointly processed using a
fusion module, resulting in a fused feature map. This fused feature map is
subsequently feeded into the Region Proposal Network to generate region
proposals that are conditioned on the textual description. The model returns the
region with the highest score as the final outcome.

![Architecture](./notebooks/figures/our_architecture.png)