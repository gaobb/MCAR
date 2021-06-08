#from __future__ import division
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
import pdb

tqdm.monitor_interval = 0
class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 256

        if self._state('batch_size') is None:
            self.state['batch_size'] = 16

        if self._state('workers') is None:
            self.state['workers'] = 4

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []
        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        self.state['meter_global_loss'] = tnt.meter.AverageValueMeter()
        self.state['meter_local_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]
    
    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['meter_global_loss'].reset()
        self.state['meter_local_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]
        global_loss = self.state['meter_global_loss'].value()[0]
        local_loss = self.state['meter_local_loss'].value()[0]
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}, Global Loss {gloss:.4f}, Local Loss {lloss: 0.4f}'.format(self.state['epoch'], loss=loss, gloss=global_loss,lloss=local_loss))
            else:
                print('Test: \t Loss {loss:.4f}, Global Loss {gloss:.4f}, Local Loss {lloss: 0.4f}'.format(loss=loss, gloss=global_loss,lloss=local_loss))
        return loss, global_loss, local_loss
   
    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # record loss
        self.state['loss_batch'] = self.state['loss'].item()
        self.state['meter_loss'].add(self.state['loss_batch'])
        self.state['meter_global_loss'].add(self.state['global_loss'].item())
        self.state['meter_local_loss'].add(self.state['local_loss'].item())

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def init_learning(self, model, criterion):
        if self._state('train_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)

            self.state['train_transform'] = transforms.Compose([
                transforms.Resize((self.state['image_size'], self.state['image_size'])),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3,
                                       contrast=0.3,
                                       saturation=0.3,
                                       hue=0),
                transforms.ToTensor(),
                normalize,
            ])
               
        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            
            self.state['val_transform'] = transforms.Compose([transforms.Resize((self.state['image_size'], self.state['image_size'])),
                                             transforms.CenterCrop(self.state['image_size']),
                                             transforms.ToTensor(),
                                             normalize])
        self.state['best_score'] = 0

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):

        self.init_learning(model, criterion)

        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['workers'])

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            print(self.state['resume'])
            #pdb.set_trace()
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))

        if not os.path.exists(self.state['save_path']):
                os.makedirs(self.state['save_path'])
        #curdate = time.strftime('%Y%m%d%H%M') 
        if self.state['evaluate']:
            #logfile = curdate + '-test.log'
            logfile = 'test.log'
            logger = Logger(os.path.join(self.state['save_path'], logfile), title='ml')
            logger.set_names(['Valid Loss', 'Valid mAP.'])
        else:
            #logfile = curdate + '-train.log'
            logfile = 'train.log'
            logger = Logger(os.path.join(self.state['save_path'], logfile), title='ml')
            logger.set_names(['Epoch', 'Train Loss', 'Train GLoss', 'Train LLoss','Valid Loss', 'Valid GLoss', 'Valid LLoss','Train mAP.', 'Valid mAP.'])

        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True

            #print('hahahha', self.state['device_ids'])
            model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()


            criterion = criterion.cuda()

        if self.state['evaluate']:
            
            self.validate(val_loader, model, criterion)
            return

        # TODO define optimizer

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:',lr)

            # train for one epoch
            train_loss, train_gloss, train_lloss, train_map = self.train(train_loader, model, criterion, optimizer, epoch)
            
            # evaluate on validation set
            #if self.state['evaluate']:
            val_loss, val_gloss, val_lloss, val_map = self.validate(val_loader, model, criterion)
            #else:
            #   val_loss, val_map = 0, 0

            # append log
            logger.append([epoch, train_loss, train_gloss, train_lloss, val_loss, val_gloss, val_lloss, train_map, val_map])
            
            # remember best prec@1 and save checkpoint
            is_best = val_map > self.state['best_score']
            self.state['best_score'] = max(val_map, self.state['best_score'])
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
            }, is_best)

            print(' *** best={best:.3f}'.format(best=self.state['best_score']))
        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch):

        # switch to train mode
        model.train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            #lr = self.adjust_learning_rate(optimizer, epoch, i, len(data_loader))

            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']: 
                #self.state['target'] = self.state['target'].cuda(async=True)
                self.state['target'] = self.state['target'].cuda()

            self.on_forward(True, model, criterion, data_loader, optimizer)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)

        loss, gloss, lloss, score = self.on_end_epoch(True, model, criterion, data_loader, optimizer)
       
        return loss, gloss, lloss, score

    def validate(self, data_loader, model, criterion):

        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()

        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu']:
                #self.state['target'] = self.state['target'].cuda(async=True)
                self.state['target'] = self.state['target'].cuda()
            self.on_forward(False, model, criterion, data_loader)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)
             
        loss, gloss, lloss, score = self.on_end_epoch(False, model, criterion, data_loader)

        return loss, gloss, lloss, score

    def save_checkpoint(self, state, is_best, model_name='model.pth.tar'):
        if self._state('save_path') is not None:
            #filename_ = str(self.state['epoch']) + '_'+ model_name
            filename_ = model_name
            filename = os.path.join(self.state['save_path'], filename_)
            if not os.path.exists(self.state['save_path']):
                os.makedirs(self.state['save_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_path') is not None:
                filename_best = os.path.join(self.state['save_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_list = []
        decay = 0.1 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)
    

class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])
    
    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()
        #self.state['meter_loss'].reset()
        #self.state['batch_time'].reset()
        #self.state['data_time'].reset()        
    
    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        ap = 100 * self.state['ap_meter'].value()
        map = ap.mean()
        loss = self.state['meter_loss'].value()[0]
        global_loss = self.state['meter_global_loss'].value()[0]
        local_loss = self.state['meter_local_loss'].value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map))
                print('AP:', ap)
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
            else:
                print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))
                print('AP:', ap) 
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
                print('OP_3: {OP:.4f}\t'
                      'OR_3: {OR:.4f}\t'
                      'OF1_3: {OF1:.4f}\t'
                      'CP_3: {CP:.4f}\t'
                      'CR_3: {CR:.4f}\t'
                      'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))

        return loss, global_loss, local_loss, map

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            global_loss = self.state['meter_global_loss'].value()[0]
            local_loss = self.state['meter_local_loss'].value()[0]

            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))


class MCARMultiLabelMAPEngine(MultiLabelMAPEngine):
    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        input_var = torch.autograd.Variable(self.state['input']).float()
        target_var = torch.autograd.Variable(self.state['target']).float()
        #inp_var = torch.autograd.Variable(self.state['input']).float().detach()  # one hot
        if not training:
            input_var.volatile = True
            target_var.volatile = True
        if self.vis == True:
           gscore, lscore, region_bboxs = model(input_var)
        else:
           gscore, lscore = model(input_var)
        self.state['output'] = torch.max(gscore, lscore)
               
        # global and local loss
        self.state['global_loss'] = criterion(gscore, target_var) 
        self.state['local_loss'] = criterion(lscore, target_var)
        #self.state['loss'] = criterion(gscore, target_var) + criterion(lscore, target_var)
        self.state['loss'] = self.state['global_loss'] + self.state['local_loss']
        
        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            #nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
            optimizer.step()

        #return  gs_time, gr_time, ls_time

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        inputs = self.state['input']
        self.state['input'] = inputs[0]
        self.state['out'] = inputs[1]
        #self.state['input'] = input[2]

