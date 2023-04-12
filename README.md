# MCAR.pytorch
This repository is a PyTorch implementation of [Learning to Discover Multi-Class Attentional Regions for Multi-Label Image Recognition](https://arxiv.org/abs/2007.01755). The paper is accepted at [IEEE Trans. Image Processing ([TIP 2021](https://signalprocessingsociety.org/publications-resources/ieee-transactions-image-processing)). This repo is created by [Bin-Bin Gao](https://csgaobb.github.io/).


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-image-recognition-with-multi/multi-label-classification-on-pascal-voc-2012)](https://paperswithcode.com/sota/multi-label-classification-on-pascal-voc-2012?p=multi-label-image-recognition-with-multi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-image-recognition-with-multi/multi-label-classification-on-pascal-voc-2007)](https://paperswithcode.com/sota/multi-label-classification-on-pascal-voc-2007?p=multi-label-image-recognition-with-multi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-image-recognition-with-multi/multi-label-classification-on-ms-coco)](https://paperswithcode.com/sota/multi-label-classification-on-ms-coco?p=multi-label-image-recognition-with-multi)

### MCAR Framework
<img src="./images/MCAR.png" style="zoom:50%;" />

### Requirements

Please, install the following packages
- numpy
- torch-0.4.1
- torchnet
- torchvision-0.2.0
- tqdm


### Options
- `topN`: number of local regions
- `threshold`: threshold of localization 
- `ps`: global pooling style, e.g., 'avg', 'max', 'gwp'
- `lr`: learning rate
- `lrp`: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is `lr * lrp`
- `batch-size`: number of images per batch
- `image-size`: size of the image
- `epochs`: number of training epochs
- `evaluate`: evaluate model on validation set
- `resume`: path to checkpoint

### MCAR Training and Evaluation

```sh
bash run.sh
```

| Model        | Input-Size | VOC-2007 | VOC-2012 | COCO-2014 |
| ------------ | ---------- | -------- | -------- | --------- |
| MobileNet-v2 | 256 x 256  | 88.1 [model](https://1drv.ms/u/s!AgTaArT8QOyggQ2xQLFhwzogJfDj?e=H9khhZ)    | -        | 69.8     [model](https://1drv.ms/u/s!AgTaArT8QOyggQtv-NQWTBYaCC_H?e=BfBFpb)   |
| ResNet-50    | 256 x 256  | 92.3 [model](https://1drv.ms/u/s!AgTaArT8QOyggQExQwv5zDQEwoSQ?e=YscdLN)      | -        | 78.0     [model](https://1drv.ms/u/s!AgTaArT8QOygenA2APxaL7e8vZI?e=eqYxmB)   |
| ResNet-101   | 256 x 256  | 93.0 [model](https://1drv.ms/u/s!AgTaArT8QOyggQWYSmGxGxivxhY5?e=AYmw7z)     | -        | 79.4      [model](https://1drv.ms/u/s!AgTaArT8QOyggQDiVkdQwhUtzcAc?e=dkD2Y7)  |
| MobileNet-v2 | 448 x 448  | 91.3 [model](https://1drv.ms/u/s!AgTaArT8QOyggQ6ZV1xFrNqkOFqo?e=AqxRbe)     | [91.0](http://host.robots.ox.ac.uk:8080/anonymous/UB2GQR.html)     | 75.0    [Model](https://1drv.ms/u/s!AgTaArT8QOyggQzTaiTH5IgreAKg?e=TQccDI)    |
| ResNet-50    | 448 x 448  | 94.1  [model](https://1drv.ms/u/s!AgTaArT8QOyggQN0dsoTxKuuC5KQ?e=L6VN28)     | [93.5](http://host.robots.ox.ac.uk:8080/anonymous/NKXC8W.html)     | 82.1    [model](https://1drv.ms/u/s!AgTaArT8QOygfbvLTx0jM-68drk?e=KyVPSB)    |
| ResNet-101   | 448 x 448  | 94.8  [model](https://1drv.ms/u/s!AgTaArT8QOyggQZl0gOOkpU5Juwq?e=ohtXFe)     | [94.3](http://host.robots.ox.ac.uk:8080/anonymous/D9S0RH.html)     | 83.8    [model](https://1drv.ms/u/s!AgTaArT8QOyggQIWnAdWB7R1S5VZ?e=kZ0XrD)    |


### MCAR Demo

```
bash run_demo.sh
```
![mcar-demo](./images/mcar-demo.png)

## Citing this repository

If you find this code useful in your research, please consider citing us:

```
@ARTICLE{MCAR_TIP_2021,
         author = {Bin-Bin Gao, Hong-Yu Zhou},
         title = {{Learning to Discover Multi-Class Attentional Regions for Multi-Label Image Recognition}},
         booktitle = {IEEE Transactions on Image Processing (TIP)},
         year={2021},
         volume={30},
         pages={5920-5932},
}
```
## Reference
This project is based on the following implementations:
- https://github.com/durandtibo/wildcat.pytorch
- https://github.com/Megvii-Nanjing/ML_GCN

## Tips
If you have any questions about our work, please do not hesitate to contact us by emails.
