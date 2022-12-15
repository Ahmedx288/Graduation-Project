
## Face-mask Detection in Real-time
This repository is for my graduation project.

**Face-mask Detection in Real-time** is a software that helps users automatically identify whether someone or a group of people are wearing a face mask or not or in the wrong way in a real-time fashion.

This is software for **Custom Object Detection**, a paradigm that appeared in the **Computer Vision Applications** field of research relating to **Deep Learning**.

![Final Mask Detection](https://user-images.githubusercontent.com/34996582/207867800-de03c487-eeb4-4754-936d-d8b5bf2ea865.gif)
___
**Table of Contents**:
|  |Description|
|--|--|
|**Overview**|The idea itself and how it lead to the creation of the project.|
|**Phase 1**|Detection using static images for one mask only.|
|**Phase 2**| Detection using static images as well as in real-time video streams for more than one mask (groups).|
___
## Overview
![The big picture of mask detection](https://user-images.githubusercontent.com/34996582/207868998-a2a6bdee-8507-40eb-b2d8-94a075bfd088.png)

In our project, we try to enforce the culture of wearing masks to further prevent the spread of the virus by monitoring vital facilities.

- It can be used to encourage and help people to wear masks in important areas.
- It can be used to prevent people from entering some other areas without it as well.

Masks are a simple barrier to help prevent your respiratory droplets from reaching others.
Studies show that masks reduce the spray of droplets when worn over the nose and mouth. 
You should wear a mask, even if you do not feel sick.
___
## Phase 1 - Static Inference
![Static Inference Example](https://user-images.githubusercontent.com/34996582/207869172-eda7d886-2ae0-4e58-8e5a-de711cf50862.png)


Used Dataset: [Face Mask Detection ~12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
This dataset consists of images belonging to two classes (With Mask / Without Mask) divided into:
- Train Data: 5000 Without Mask Images / 5000 With Mask Images
- Test Data :  509 Without Mask Images /  483 With Mask Images
- Validation Data: 400 Without Mask Images / 400 With Mask Images

Complete Project (Notebook): [face-mask-detection.ipynb](https://github.com/Ahmedx288/Graduation-Project/blob/main/Phase%201%20-%20PyTorch%20Model/face-mask-detection.ipynb)

Feel free to use the raw python scripts to train or inference:

1- Train: [train.py](https://github.com/Ahmedx288/Graduation-Project/blob/main/Phase%201%20-%20PyTorch%20Model/Scripts/train.py)

2- Inference: [predict.py](https://github.com/Ahmedx288/Graduation-Project/blob/main/Phase%201%20-%20PyTorch%20Model/Scripts/predict.py)

![Scripts instructions](https://user-images.githubusercontent.com/34996582/207873019-e1060648-7cd6-4140-b17a-79af56a5bead.png)

Note: For more info about this phase of the project feel free to read the SRS Document. [Link](https://github.com/Ahmedx288/Graduation-Project/blob/main/Documents/SRS%20Document.pdf).

___
