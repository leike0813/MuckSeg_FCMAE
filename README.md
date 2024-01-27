# MuckSeg_FCMAE: A self-supervised pre-training framework for real-time instance segmentation of TBM muck images

## Intruduction

MuckSeg_FCMAE is a self-supervised pre-training framework to real-time instance segmentation of TBM muck images.

<img src="/docs/img1.png" alt="result1" width="768" height="512"> 

## Requirements

- pytorch 1.8.0
- pytorch-lightning 1.6.5
- minkowskiengine 0.5.4
- cuda 11.1 support


## Train

Use the cli to train MuckSeg_FCMAE:

```bash
python train.py --cfg <path-to-config-file> --data-path <path-to-train-dataset>
```

## Data availablity statement

If you wish to use the complete dataset for training MuckSeg_FCMAE, please contact zlzhou1@bjtu.edu.cn.