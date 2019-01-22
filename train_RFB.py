from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.modules import MultiBoxLoss
from layers.functions import PriorBox
import time


parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='./weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=32,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument(
    '--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('-max','--max_epoch', default=300,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='./weights/',
                    help='Location to save checkpoint models')
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
# 训练 + 验证集
if args.dataset == 'VOC':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    train_sets = [('2014', 'train'),('2014', 'valminusminival')]
    cfg = (COCO_300, COCO_512)[args.size == '512']


# 基于主干网构建RFBNet
if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = COCO_mobile_300
else:
    print('Unkown version!')

img_dim = (300,512)[args.size=='512']
rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']
p = (0.6,0.2)[args.version == 'RFB_mobile']
num_classes = (21, 81)[args.dataset == 'COCO']
batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9

net = build_net('train', img_dim, num_classes) # RFBNet网络构建
print(net)
if args.resume_net == None:
    base_weights = torch.load(args.basenet) # 相当于加载vgg主干网的模型参数
    print('Loading base network...')
    net.base.load_state_dict(base_weights) # 只灌入了RFBNet中vgg主干网的参数

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        for key in m.state_dict(): # 新增层参数的初始化
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(m.state_dict()[key], mode='fan_out') # conv层参数的初始化,用的kaiming_normal
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias': # bias初始化为0
                m.state_dict()[key][...] = 0

    print('Initializing weights...')
# initialize newly added layers' weights with kaiming_normal method,新增层参数的初始化
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    net.Norm.apply(weights_init)
    if args.version == 'RFB_E_vgg':# 这两层是RFB_E_vgg中特有的，可结合论文中table 4理解
        net.reduce.apply(weights_init)
        net.up_reduce.apply(weights_init)

else:
    # load resume network，相当于已有了检测模型，直接在其之上继续finetune即可，
    # 网络结构都不需要调整，也适合于在现有epoch上继续finetune
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict) # 灌训练好的参数

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.cuda:
    net.cuda()
    cudnn.benchmark = True


# SGD优化策略
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
#optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                      momentum=args.momentum, weight_decay=args.weight_decay)

criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)# 沿用了SSD的MultiBoxLoss，可以参照multibox_loss.py
priorbox = PriorBox(cfg) # 特征金字塔上的先验prior_box，可结合prior_box.py理解
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()



def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0 + args.resume_epoch # finetune方式地训练
    print('Loading Dataset...')
    # 加载训练、验证集，preproc类可以参照data_augment.py函数，与SSD数据增强方式一致
    if args.dataset == 'VOC':
        dataset = VOCDetection(VOCroot, train_sets, preproc(
            img_dim, rgb_means, p), AnnotationTransform())
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, p))
    else:
        print('Only VOC and COCO are supported now!')
        return

    epoch_size = len(dataset) // args.batch_size # 每个epoch内需要处理的iter次数
    max_iter = args.max_epoch * epoch_size # 总iter次数，max_epoch*epoch_size
    # learning rate调整的节点
    stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues = (stepvalues_VOC,stepvalues_COCO)[args.dataset=='COCO']
    print('Training',args.version, 'on', dataset.name)
    step_index = 0
    # 是否需要finetune
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    lr = args.lr
    for iteration in range(start_iter, max_iter):# 共需迭代的次数，是否finetune间有差异，同时也对应到了epoch次数
        if iteration % epoch_size == 0:
            # create batch iterator 新一轮epoch加载数据，把全部数据又重新加载了，下面的next(batch_iterator)再逐batch_size地取数据
            batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate))
            # detection_collate逐batch_size地取出图像 + 标签
            loc_loss = 0
            conf_loss = 0
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 ==0 and epoch > 200):
                torch.save(net.state_dict(), args.save_folder+args.version+'_'+args.dataset + '_epoches_'+
                           repr(epoch) + '.pth') # 模型保存
            epoch += 1

        load_t0 = time.time()

        # 以下操作就是针对lr的调整，warming up操作
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)


        # load train data,batch_iterator一次性加载了数据，next操作就逐个batch_size地取出数据了
        images, targets = next(batch_iterator)# 可以对应到detection_collate函数
        
        #print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))
        # 对应cuda操作
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]
        # forward
        t0 = time.time()
        out = net(images)# batch_size图像批操作，直接forward得到结果
        # backprop
        optimizer.zero_grad() # Clears the gradients of all optimized，本batch_size内来一波
        loss_l, loss_c = criterion(out, priors, targets) # 对应到MultiBoxLoss，可以参照multibox_loss.py
        loss = loss_l + loss_c # 这里设置的loc loss、cls loss权重系数为1:1
        loss.backward() # loss bp反向传播
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()# 累加batch_size内的loss
        conf_loss += loss_c.item()
        load_t1 = time.time()
        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f||' % (
                loss_l.item(),loss_c.item()) + 
                'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))
    # 最终保存的模型
    torch.save(net.state_dict(), args.save_folder +
               'Final_' + args.version +'_' + args.dataset+ '.pth')

# learning rate的warming up操作
def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5) 
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
