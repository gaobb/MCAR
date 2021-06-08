import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from logger import Logger
from util import *

import argparse
from engine import *
from models import *
from voc import *
from coco import *
import pdb
import cv2
import glob
import numpy as np

parser = argparse.ArgumentParser(description='MCAR Demo')
parser.add_argument('--data-path', metavar='DIR',
                    help='path to testing images (e.g. ./images/coco14val')
parser.add_argument('--dataset-name', default='voc2012', choices=['voc2012', 'coco2014'], type=str,
                    help='dataset name (e.g. voc2012 or coco2014)')
parser.add_argument('--image-size', '-i', default=256, type=int,
                    metavar='N', help='image size (default: 256)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--bm', '--base-model', type=str, metavar='basemodel',
                    help='pre-trained model (e.g. mobilenetv2 or resnet50 or resnet101')
parser.add_argument('--ps', '--pooling-style', type=str, metavar='poolingstyle',
                    help='pooling style (e.g. avg or max or gwp')
parser.add_argument('--topN', default=4, type=int, metavar='topN',
                    help='number of potensial objects')
parser.add_argument('--threshold', default=0.5, type=float, metavar='threshold',
                    help='threshold of localization')
parser.add_argument('--sp', '--save-path', default='glmodels', type=str, metavar='savepath',
                    help='path to save models (default: glmodels)')

class MCARDemo():
    def __init__(self, base_model, ps, topN, threshold, resume, dataset_name, imgsize=448):
        super(MCARDemo, self).__init__() 
        print(dataset_name)
        if dataset_name == 'voc2012':
           num_classes = 20
           self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

        if args.dataset_name == 'coco2014':
           num_classes = 80
           self.class_names = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', \
                    'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bs', 'cake', 'car', 'carrot', \
                    'cat', 'cell phone', 'chair', 'clock', 'coch', 'cow', 'cp', 'dining table', 'dog', 'dont',\
                    'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard',\
                    'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mose', 'orange', 'oven', 'parking meter', 'person',\
                    'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis',\
                    'snowboard', 'spoon', 'sports ball', 'stop sign', 'sitcase', 'srfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster',\
                    'toilet', 'toothbrsh', 'traffic light', 'train', 'trck', 'tv', 'mbrella', 'vase', 'wine glass', 'zebra']

        if base_model == 'resnet101':
           self.model = mcar_resnet101(num_classes=num_classes, ps=args.ps, topN=args.topN, threshold=args.threshold, pretrained=True, vis=True)
        if base_model == 'resnet50':
           self.model = mcar_resnet50(num_classes=num_classes,  ps=args.ps, topN=args.topN, threshold=args.threshold, pretrained=True, vis=True)
        if base_model == 'mobilenetv2':
           self.model = mcar_mobilenetv2(num_classes=num_classes, ps=args.ps, topN=args.topN, threshold=args.threshold, pretrained=True, vis=True)
      
        #loading checkpoint from  resume
        checkpoint = torch.load(args.resume)
        self.model.load_state_dict(checkpoint['state_dict'])

        normalize = transforms.Normalize(mean=self.model.image_normalization_mean,
                                         std=self.model.image_normalization_std)
        cudnn.benchmark = True
        self.model = torch.nn.DataParallel(self.model, device_ids=None).cuda()
        self.model.eval()        

        self.image_size = imgsize
        self.topN = topN
        self.threshold =  threshold
                
        self.transform = transforms.Compose([
                transforms.Resize((imgsize, imgsize)),
                transforms.ToTensor(),
                normalize,
            ])      
 
    def image_forward(self, img_name):
        img = Image.open(img_name).convert('RGB')
        timg = self.transform(img)
        timg = timg.reshape(1,3,timg.size(1), timg.size(2)).cuda().float()

        gscore, lscore, region_bboxs = self.model(timg)
        region_bboxs = region_bboxs.squeeze(0)
      
        #score = torch.max(gscore, lscore)

        return gscore, lscore, region_bboxs

def main():
    global args, use_gpu
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    if not os.path.exists(args.sp):
       os.makedirs(args.sp)
    
    # init mcar 
    mcar = MCARDemo(args.bm, args.ps, args.topN, args.threshold, args.resume, args.dataset_name, args.image_size)

    # vislization
    res_csv = os.path.join(args.sp, args.dataset_name + '-test.csv')
   
    f = open(res_csv, 'w')
    for img_name in glob.glob(args.data_path +'/*.jpg'):
        print(img_name)
        gscore, lscore, region_bboxs = mcar.image_forward(img_name)
        score = torch.max(gscore, lscore)

        cvimg = cv2.imread(img_name)
        re_img = cv2.resize(cvimg, (args.image_size, args.image_size), interpolation=cv2.INTER_CUBIC)
        vis_img = draw_bbox(re_img, region_bboxs, score, mcar.class_names, args.image_size)
        vis_img_path = args.sp +'/' + img_name.split('/')[-1]
  
        cv2.imwrite(vis_img_path, vis_img)
        num_classes = len(mcar.class_names)
        f.write(img_name.split('/')[-1])
        for i in range(num_classes):
            f.write(',')
            f.write(format(score.data.cpu()[0,i]))
        f.write('\n')
    f.close()   
     

if __name__ == '__main__':
    main()
