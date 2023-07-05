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