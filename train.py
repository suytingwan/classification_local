
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
from dataset import  NoisyDataset as Dataset
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models
import torchvision.datasets as datasets
from weight_sampling import ImageWeightSampling

from utils import  Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, MixLoss
from transform import *
import json

default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

parser = argparse.ArgumentParser(description='PyTorch Imagenet Training')

#Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--root', default='', type=str,
                    help='root dir of training data')
parser.add_argument('--root-val', default='', type=str,
                    help='root dir of validation data')
parser.add_argument('--train-list', default='', type=str,
                    help='training data list file')
parser.add_argument('--val-list', default='', type=str,
                    help='validation data list file')

# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-epoch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-epoch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schdule.')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default:1e-4)')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

#Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet101)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNext cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNext base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64 ...')

#Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--num-classes', default=9036, type=int, help='number of classes')
parser.add_argument('--BC-learning', dest='BC_learning', action='store_true',
                            help='use between class training method')
parser.add_argument('--data-augmentation', dest='data_augmentation', action='store_true',
                    help='use data augmentation strategy in training')
parser.add_argument('--attention', dest='attention', action='store_true',
                    help='use spatial attention after last conv layer')

parser.add_argument('--weight-loss', dest='weight_loss', action='store_true',
                    help='use weighted loss on criterion')
parser.add_argument('--weight-file', default='', type=str,
                    help='file of weighted ratio on each class')
parser.add_argument('--mix-loss', default='mix_loss', action='store_true',
                    help='use crossentropy loss and cosine loss both')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0

def main():
    global best_acc
    start_epoch = args.start_epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229,0.224, 0.225])

    if args.data_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            RandomVerticalFlip(),
            transforms.Scale(256),
            transforms.RandomSizedCrop(224),
            ColorJitter(0.5, 0.5, 0.5),
            RandomRotation(30),
            RandomAffine(30),
            transforms.ToTensor(),
            normalize
            ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
        
    #trainset = Dataset(root=args.root, filename=args.train_list, transform=transform_train)

    trainset = ImageWeightSampling(args.root, args.train_list, transform=transform_train)
    testset = Dataset(root=args.root_val, filename=args.val_list, transform=transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
        ]))

    #sampler = torch.utils.data.sampler.WeightedRandomSampler(trainset.sample_weights, len(trainset.sample_weights))

    #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_epoch,sampler=sampler, num_workers=args.workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_epoch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_epoch, shuffle=False, num_workers=args.workers, pin_memory=True)

    #create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        #check if pretrained model exists
        pretrained_filename = os.path.join('./pretrained', args.arch)
        assert os.path.isfile(pretrained_filename), 'Error: no pretrained checkpoint directory found!'
        model = models.__dict__[args.arch](
            pretrained=True,
            num_classes=args.num_classes)
    elif args.arch.startswith('resnext') or args.arch.startswith('se_resnext'):
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](
            baseWidth=args.base_width,
            cardinality=args.cardinality,
            num_classes=args.num_classes,
            attention=args.attention
        )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](
            num_classes=args.num_classes
        )

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        models.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print(' Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if args.weight_loss:
        assert os.path.isfile(args.weight_file), 'weight loss training while weight file not found'
        weights = json.load(open(args.weight_file))
        weight_ = []
        for i in range(len(weights.keys())):
            weight_.append(weights[str(i)])
        weight_ = np.array(weight_, dtype=np.float32)
        weight_ = weight_ / sum(weight_) * (args.num_classes + 0.0) * 0.5 + 0.5
        print(weight_)
        weight_ = torch.from_numpy(weight_)

    if args.weight_loss:
        #criterion = nn.CrossEntropyLoss(weight=weight_,reduce=False).cuda()
        criterion = nn.CrossEntropyLoss(weight=weight_).cuda()
    elif args.mix_loss:
        center = np.loadtxt('center.txt')
        criterion = MixLoss(torch.from_numpy(center).cuda()).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    #Resume
    title = 'Imagenet-' + args.arch
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss: %.8f, Test Acc: %.2f' % (test_loss, test_acc))

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)

        #append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        #save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint(epoch, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc: ')
    print(best_acc)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        if batch_idx % 5000 == 1:
            save_checkpoint('{}-{}'.format(epoch+1, batch_idx),{
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'acc':0.0,
                'best_acc':0.0,
                'optimizer':optimizer.state_dict()},0,checkpoint=args.checkpoint)
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1,5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets, sample_weights) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, torch.max(targets, 1)[1])
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(epoch, state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filename = 'checkpoint-{}.pth.tar'.format(epoch)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()





