# -*- coding:utf-8 -*-
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models

from dataset import NoisyDataset as Dataset

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from torch.utils.data.dataloader import default_collate
from decimal import *
import pandas as pd

# Models
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

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
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
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--num-classes', default=9036, type=int, help='num of classification kinds')

parser.add_argument('--save-dir', default='', type=str, help='directory of saving file')

parser.add_argument('--attention', dest='attention', action='store_true',
                    help='use spatial attention after last conv layer')
parser.add_argument('--output-feature', dest='output_feature', action='store_true',
                    help='output model features for ensemble')
parser.add_argument('--output-attention', dest='output_attention', action='store_true',
                    help='output attention value for visualization')
parser.add_argument('--output-prob', dest='output_prob', action='store_true',
                    help='output probability for ensemble')

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

ave_acc = 0


def main():
    global ave_acc

    if not os.path.isdir(args.checkpoint):
        print('checkpoint dir does not exist.')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    testset = Dataset(root=args.root_val,
                      filename=args.val_list,
                      transform=transforms.Compose([
                          transforms.Scale(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          normalize]))

    image_names = testset.get_ids()

    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_epoch, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](
            pretrained=True,
            num_classes=args.num_classes
        )
    elif args.arch.startswith('resnext') or args.arch.startswith('se_resnext'):
        model = models.__dict__[args.arch](
            baseWidth=args.base_width,
            cardinality=args.cardinality,
            num_classes=args.num_classes,
            attention=args.attention,
            output_feature=args.output_feature,
            output_attention=args.output_attention
        )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=args.num_classes,output_feature=args.output_feature)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found'
    args.checkpoint = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    if args.arch.startswith('mil'):
        checkpoint = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint.items() if k.replace('module.', '') in model_dict}
        pretrained_dict['bag_fc.weight'] = checkpoint['module.fc.weight']
        pretrained_dict['bag_fc.bias'] = checkpoint['module.fc.bias']
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(checkpoint['state_dict'])
    criterion = nn.CrossEntropyLoss().cuda()

    if args.output_feature:
        output_feature(val_loader, model, 0, use_cuda, args.save_dir)
    elif args.output_attention:
        output_attention(val_loader, model, 0, use_cuda, args.save_dir)
    elif args.output_prob:
        output_prob(val_loader, model, 0, use_cuda, args.save_dir,image_names)
    else:
        test_loss, test_acc = test(val_loader, model, criterion, 0, use_cuda, args.save_dir, image_names)
        print('test_loss: {} test_acc: {}'.format(test_loss, test_acc))


def test(val_loader, model, criterion, epoch, use_cuda, save_dir, image_names):
    global ave_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))

    '''
    map_dict = {}
    for line in f:
        info = line.strip().split('\t')
        map_dict[int(info[1])] = info[0]
    f.close()
    '''
    #fw = open(os.path.join(save_dir, 'results.txt'), 'w')
    #submit_id = []
    submit_result = []

    count = 0
    for batch_idx, (inputs, targets, weights) in enumerate(val_loader):
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        outputs = model(inputs)
	
	    #probs = torch.nn.functional.softmax(probs)
	    #topk_probs, topk_preds = probs.topk(1, 1, True, True)
	    #topk_probs = torch.log(topk_probs)
	    #print(topk_probs.size())
        
        #loss = criterion(outputs, torch.max(targets, 1)[1])
        targets = torch.squeeze(targets)
        loss = criterion(outputs, targets)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        # save top1 mapping result
        pred_float, pred = outputs.data.topk(5, 1, True, True)
  
        pred = pred.cpu().numpy()
        pred = pred.reshape((-1))
        targets = targets.cpu().data.numpy()
        targets = targets.reshape((-1))
        
        # print(pred)
        # print(targets)
        
        loss_numpy = loss.data
        #print(loss_numpy)
        for index in range(outputs.data.size(0)):
            #fw.write('{} pred:{} real:{} loss:{}'.format(image_names[count], pred[index][5*index], targets[index], loss.data[index]))
            #fw.write('\n')
            submit_each = []
	        #count += 1
            submit_each.append(image_names[count])
            submit_each.extend(list(pred[index*5:(index+1)*5]))
            submit_result.append(submit_each)
            count += 1
        #print(submit_result)
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg
        )

        bar.next()
    df = pd.DataFrame(submit_result, columns=['image_id','predicted_0','predicted_1','predicted_2','predicted_3','predicted_4'])
    df.to_csv(os.path.join(save_dir, 'submit_cluster_reweight_0.5.csv'),index=False,header=False,sep=' ')

    bar.finish()
    #fw.close()
    return (losses.avg, top1.avg)

def output_feature(val_loader, model, epoch, use_cuda, save_dir):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    #fw = open(os.path.join(save_dir,'val_feature.txt'), 'w')
    fp = open(os.path.join(save_dir,'train_probability.txt'), 'w')
    for batch_idx, (inputs, targets, weights) in enumerate(val_loader):
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        probs, outputs = model(inputs)
        probs = torch.nn.functional.softmax(probs)
        topk_probs, topk_preds = probs.topk(20, 1, True, True)
        output_probs = topk_probs.data.cpu().numpy()
        output_preds = topk_preds.data.cpu().numpy()
        output_feature = outputs.data.cpu().numpy()
        #print(output_feature)
        real_targets = targets.cpu().data.numpy()
        real_targets = real_targets.reshape((-1))

        for index in range(outputs.data.size(0)):
            feature = str(real_targets[index])

            #for i in range(outputs.data.size(1)):
                #print(output_feature[index][i])
                #feature += ' {:.3f}'.format(output_feature[index][i])
            #fw.write(feature)
            #fw.write('\n')
            
            probability = str(real_targets[index])
            for i in range(20):
                probability += ' {}:{:.3f}'.format(output_preds[index][i], output_probs[index][i])
            fp.write(probability)
            fp.write('\n')
            #print(output_feature[index])

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
        )
        bar.next()
    bar.finish()
    fp.close()

def output_attention(val_loader, model, epoch, use_cuda, save_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))

    fw = open(os.path.join(save_dir, 'attention.txt'), 'w')
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        probs, attention = model(inputs)

        bs, c, w, h = attention.size()
        attention = attention.sum(1)
        attention = attention.cpu().data.numpy()
        attention = attention.reshape((bs, -1))
        for index in range(bs):
            hot = ''
            for j in range(w * h):
                hot += '{:.3f} '.format(attention[index][j])
            hot += '\n'
            fw.write(hot)
        prec1, prec5 = accuracy(probs.data, targets.data, topk=(1,5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        bar.shuffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            top1=top1.avg,
            top5=top5.avg
        )
        bar.next()
    bar.finish()
    fw.close()

def output_prob(val_loader, model, epoch, use_cuda, save_dir, image_names):
    global avg_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    #bar = Bar('Processing', max=len(val_loader))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fw = open(os.path.join(save_dir, 'val_single_crop_probability.txt'),'w')
    submit_result = []
    count = 0
    for batch_idx, (inputs, targets, weights) in enumerate(val_loader):
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        outputs = model(inputs)
        targets = torch.squeeze(targets)
        prec1, prec5 = accuracy(outputs.data, targets.data,topk=(1,5))

        probs = torch.nn.functional.softmax(outputs)
        topk_probs, topk_preds = probs.topk(20, 1, True, True)
        pred_float, pred = outputs.data.topk(5,1,True,True)
        pred = pred.cpu().numpy()
        pred = pred.reshape((-1))
        output_probs = topk_probs.data.cpu().numpy()
        output_preds = topk_preds.data.cpu().numpy()
        real_targets = targets.cpu().data.numpy()
        #real_targets = real_targets.reshape(-1)

        for index in range(outputs.data.size(0)):
            probability = str(real_targets[index])
            for i in range(20):
                probability += ' {}:{:.3f}'.format(output_preds[index][i],output_probs[index][i])
            fw.write(probability)
            fw.write('\n')

            submit_each = []
            submit_each.append(image_names[count])
            submit_each.extend(list(pred[index*5:(index+1)*5]))
            submit_result.append(submit_each)
            count += 1

        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        print('Processing: {}/{} Batch: {:.3f} top1: {:.4f} top5: {:.4f}'.format(batch_idx+1, len(val_loader), batch_time.avg, top1.avg, top5.avg)) 
        #bar.shuffix = '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
        #    batch=batch_idx + 1,
        #    size=len(val_loader),
        #    bt=batch_time.avg,
        #    total=bar.elapsed_td,
        #    eta=bar.eta_td,
        #    top1=top1.avg,
        #    top5=top5.avg)
        #bar.next()
    #bar.finish()
    df = pd.DataFrame(submit_result, columns=['image_id','predicted_0','predicted_1','predicted_2','predicted_3','predicted_4'])
    df.to_csv(os.path.join(save_dir, 'val_single_crop.csv'),index=False,header=False,sep=' ')
    
    fw.close()


if __name__ == '__main__':
    main()




