import argparse
from  engine import *
from models import *
from voc import *

parser = argparse.ArgumentParser(description='MCAR Training')
parser.add_argument('--data-path', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--dataset-name', default='voc2007', choices=['voc2007', 'voc2012', 'coco2014'], type=str, 
                    help='dataset name (e.g. voc07 or coco14)') 
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30,50], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--bs', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--sp', '--save-path', default='glmodels', type=str, metavar='savepath',
                    help='path to save models (default: glmodels)')
parser.add_argument('--bm', '--base-model', type=str, metavar='basemodel',
                    help='pre-trained model (e.g. resnet50 or resnet101')
parser.add_argument('--ps', '--pooling-style', type=str, metavar='poolingstyle',
                    help='pooling style (e.g. avg or max')
parser.add_argument('--topN', default=4, type=int, metavar='topN',
                    help='number of potensial objects')
parser.add_argument('--threshold', default=0.5, type=float, metavar='threshold',
                    help='threshold of localization')


def main():
    global args, use_gpu
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # define dataset
    if args.dataset_name == 'voc2007':
       train_dataset = Voc2007Classification(args.data_path, 'trainval')
       val_dataset = Voc2007Classification(args.data_path, 'test')
       num_classes = 20
    if args.dataset_name == 'voc2012':
       train_dataset = Voc2012Classification(args.data_path, 'trainval')
       val_dataset = Voc2007Classification(args.data_path, 'test')
       num_classes = 20 
    if args.dataset_name == 'coco2014':
       train_dataset = COCO2014(args.data_path, phase='train') 
       val_dataset = COCO2014(args.data_path, phase='val') 
       num_classes = 80
 
    # load model
    if args.bm == 'resnet101':
       model = mcar_resnet101(num_classes=num_classes, ps=args.ps, topN=args.topN, threshold=args.threshold, pretrained=True)
       args.epoch_step = [30, 50]
       args.lrp = 0.1
       args.lr = 0.01
    if args.bm == 'resnet50':
       model = mcar_resnet50(num_classes=num_classes,  ps=args.ps, topN=args.topN, threshold=args.threshold, pretrained=True)
       args.epoch_step = [30, 50]
       args.lrp = 0.1
       args.lr = 0.01
    
    if args.bm == 'vgg16':
       model = mcar_vgg16(num_classes=num_classes,  ps=args.ps, topN=args.topN, threshold=args.threshold, pretrained=True)
       args.epoch_step = [30, 50]
       args.lrp = 0.1
       args.lr = 0.01

    if args.bm == 'mobilenetv2':
       model = mcar_mobilenetv2(num_classes=num_classes, ps=args.ps, topN=args.topN, threshold=args.threshold, pretrained=True) 
       if args.image_size == 448 and args.ps == 'avg':
          args.epoch_step = [10, 30, 50]
          args.lrp = 0.1
          args.lr = 0.1
       else:
          args.epoch_step = [30, 50]
          args.lrp = 0.1
          args.lr = 0.01
    # define loss function (criterion)
    criterion = nn.BCELoss()
    
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.bs, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_path'] = args.sp
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    state['use_pb'] = True
    if args.evaluate:
        state['evaluate'] = True
    engine = MCARMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)



if __name__ == '__main__':
    main()
