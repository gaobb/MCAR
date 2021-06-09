import torchvision.models as models
from mobilenetv2 import *
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import time 

def feats_pooling(x, method='avg', sh=8, sw=8):
    if method == 'avg':
       x = F.avg_pool2d(x, (sh, sw))
    if method == 'max':
       x = F.max_pool2d(x, (sh, sw))
    if method == 'gwp':
       x1 = F.max_pool2d(x, (sh, sw))
       x2 = F.avg_pool2d(x, (sh, sw))
       x = (x1 + x2)/2
    return x


class MCARMobilenetv2(nn.Module):
    def __init__(self, model, num_classes, ps, topN=4, threshold=0.5, vis=False):
        super(MCARMobilenetv2, self).__init__()
        self.features = nn.Sequential(
            model.features,
            model.conv         
        )
        num_features = model.conv[0].out_channels
        self.convclass = nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.ps = ps
        self.num_classes = num_classes
        self.num_features = num_features
        self.topN = topN
        self.threshold = threshold
        self.vis = vis
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, inputs):
        # global stream
        b, c, h, w = inputs.size()
        ga = self.features(inputs)          # activation map 
        gf = feats_pooling(ga, method=self.ps, sh=int(h/32), sw=int(w/32))
        gs = self.convclass(gf)             # bxcx1x1
        gs = torch.sigmoid(gs)              # bxcx1x1          
        gs = gs.view(gs.size(0), -1)        # bxc

        # from global to local
        torch.cuda.synchronize()
        start_time = time.time() 
        camscore = self.convclass(ga.detach()) 
        camscore = torch.sigmoid(camscore)                 
        camscore = F.interpolate(camscore, size=(h, w), mode='bilinear', align_corners=True)
        wscore = F.max_pool2d(camscore, (h, 1)).squeeze(dim=2)  
        hscore = F.max_pool2d(camscore, (1, w)).squeeze(dim=3)
       

        linputs = torch.zeros([b, self.topN, 3, h, w]).cuda() 
        if self.vis == True:
           region_bboxs = torch.FloatTensor(b, self.topN, 6)
        for i in range(b): 
            gs_inv, gs_ind = gs[i].sort(descending=True)
            for j in range(self.topN):
                xs = wscore[i,gs_ind[j],:].squeeze()
                ys = hscore[i,gs_ind[j],:].squeeze()
                if xs.max() == xs.min():
                   xs = xs/xs.max()
                else: 
                   xs = (xs-xs.min())/(xs.max()-xs.min())
                if ys.max() == ys.min():
                   ys = ys/ys.max()
                else:
                   ys = (ys-ys.min())/(ys.max()-ys.min())
                x1, x2 = obj_loc(xs, self.threshold)
                y1, y2 = obj_loc(ys, self.threshold)
                linputs[i:i+1, j ] = F.interpolate(inputs[i:i+1, :, y1:y2, x1:x2], size=(h, w), mode='bilinear', align_corners=True)
                if self.vis == True:
                   region_bboxs[i,j] = torch.Tensor([x1, x2, y1, y2, gs_ind[j].item(), gs[i, gs_ind[j]].item()]) 

        
        # local stream 
        linputs = linputs.view(b * self.topN, 3, h, w)         
        la = self.features(linputs.detach())
        lf = feats_pooling(la, method=self.ps, sh=int(h/32), sw=int(w/32))
        lf = self.convclass(lf)
        ls = torch.sigmoid(lf)
        ls = F.max_pool2d(ls.reshape(b, self.topN, self.num_classes, 1).permute(0,3,1,2), (self.topN, 1))
        ls = ls.view(ls.size(0), -1)  #bxc
        
        if self.vis == True:
           return gs, ls, region_bboxs
        else:
           return gs, ls  

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.convclass.parameters(), 'lr': lr},
                ]


class MCARResnet(nn.Module):
    def __init__(self, model, num_classes, ps, topN, threshold,  vis=False):
        super(MCARResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        num_features = model.layer4[1].conv1.in_channels
        self.convclass = nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.ps = ps
        self.num_classes = num_classes
        self.num_features = num_features
        self.topN = topN
        self.threshold = threshold
        self.vis = vis
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, inputs):
        # global stream
        b, c, h, w = inputs.size()
        ga= self.features(inputs)
        gf = feats_pooling(ga, method=self.ps, sh=int(h/32), sw=int(w/32))                
        gf = self.convclass(gf)             #bxcx1x1
        gs = torch.sigmoid(gf)              #bxcx1x1          
        gs = gs.view(gs.size(0), -1)        #bxc
        

        # from global to local
        camscore = self.convclass(ga.detach())
        camscore = torch.sigmoid(camscore)
        camscore = F.interpolate(camscore, size=(h, w), mode='bilinear', align_corners=True)
        wscore = F.max_pool2d(camscore, (h, 1)).squeeze(dim=2)
        hscore = F.max_pool2d(camscore, (1, w)).squeeze(dim=3)
        
        linputs = torch.zeros([b, self.topN, 3, h, w]).cuda() 
        if self.vis == True:
           region_bboxs = torch.FloatTensor(b, self.topN, 6)
        for i in range(b): 
            # topN for MCAR method
            gs_inv, gs_ind = gs[i].sort(descending=True)    
        
            # bootomN for ablation study
            # gs_inv, gs_ind = gs[i].sort(descending=False)
            
            # randomN for ablation study
            # perm = torch.randperm(gs[i].size(0))
            # gs_inv = gs[i][perm]
            # gs_ind = perm

            for j in range(self.topN):
                xs = wscore[i,gs_ind[j],:].squeeze()
                ys = hscore[i,gs_ind[j],:].squeeze()
                if xs.max() == xs.min():
                   xs = xs/xs.max()
                else: 
                   xs = (xs-xs.min())/(xs.max()-xs.min())
                if ys.max() == ys.min():
                   ys = ys/ys.max()
                else:
                   ys = (ys-ys.min())/(ys.max()-ys.min())
                x1, x2 = obj_loc(xs, self.threshold)
                y1, y2 = obj_loc(ys, self.threshold)
                linputs[i:i+1, j ] = F.interpolate(inputs[i:i+1, :, y1:y2, x1:x2], size=(h, w), mode='bilinear', align_corners=True)
                if self.vis == True:
                   region_bboxs[i,j] = torch.Tensor([x1, x2, y1, y2, gs_ind[j].item(), gs[i, gs_ind[j]].item()])

        # local stream
        linputs = linputs.view(b * self.topN, 3, h, w)         
        la = self.features(linputs.detach())
        lf = feats_pooling(la, method=self.ps, sh=int(h/32), sw=int(w/32))
        lf = self.convclass(lf)
        ls = torch.sigmoid(lf)
        ls = F.max_pool2d(ls.reshape(b, self.topN, self.num_classes, 1).permute(0,3,1,2), (self.topN, 1))
        ls = ls.view(ls.size(0), -1)
        
        if self.vis == True:
           return gs, ls, region_bboxs
        else:
           return gs, ls

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.convclass.parameters(), 'lr': lr},
                ]

def mcar_resnet50(num_classes, ps, topN, threshold, pretrained=True,  vis=False):
    model = models.resnet50(pretrained=pretrained)
    return MCARResnet(model, num_classes, ps, topN, threshold,  vis)

def mcar_resnet101(num_classes, ps, topN, threshold, pretrained=True,  vis=False):
    model = models.resnet101(pretrained=pretrained)
    return MCARResnet(model, num_classes, ps, topN, threshold,  vis)

def mcar_mobilenetv2(num_classes, ps, topN, threshold, pretrained=True, vis=False):
    model = mobilenetv2()
    if pretrained == True:
       m2net = torch.load('../pretrained/mobilenetv2_1.0-0c6065bc.pth')
       model.load_state_dict(m2net)
    return  MCARMobilenetv2(model, num_classes, ps, topN, threshold, vis)



