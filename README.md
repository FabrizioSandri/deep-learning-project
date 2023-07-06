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

## Project goal
Visual grounding involves linking language and perception by grounding
linguistic symbols in the visual world. The goal of this assignment is to build,
fine-tune, and evaluate a deep learning framework that can perform visual
grounding on a given dataset.

The assignment focuses on training a deep learning framework for visual
grounding using the CLIP (Contrastive Language-Image Pre-training) model as a
foundation. CLIP is a pre-trained model that provides a starting point for
transfer learning, allowing us to leverage its capabilities in image-text
understanding for the visual grounding task. The objective is to fine-tune CLIP
specifically for visual grounding, which requires predicting bounding boxes in
images corresponding to the entities described in textual descriptions.

## Dataset
The visual grounding task utilizes the RefCOCOg dataset, a variant of the
Referring Expression Generation (REG) dataset. It consists of approximately
25,799 images, each with an average of 3.7 referring expressions. The dataset
contains appearance-based descriptions independent of viewer perspective, making
it suitable for visual grounding. Accurate bounding boxes need to be generated
around the referred objects in the images, considering the context and visual
properties.

## ClipRPN
In the following diagram we present a high-level representation of our proposed
framework. The architecture begins by converting an image and a textual
description into a feature map and text embedding. This is achieved using the
ResNet50 of CLIP (excluding the last attention layer) for the image and the text
encoder of CLIP for the textual input. The feature map and text embedding are
then jointly processed using a fusion module, resulting in a fused feature map.
This fused feature map is subsequently feeded into the Region Proposal Network
to generate region proposals that are conditioned on the textual description.
The model returns the region with the highest score as the final outcome.

![Architecture](./notebooks/figures/our_architecture.png)