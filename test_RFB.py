from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot,COCOroot 
from data import AnnotationTransform, COCODetection, VOCDetection, BaseTransform, VOC_300,VOC_512,COCO_300,COCO_512, COCO_mobile_300

import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer

# 训练好了模型，提供测试数据，来一遍forward操作，就可以直接算出模型在测试数据集上的mAP
parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.') # RFBNet模型结构，主干网是vgg、mobilenet等
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default='weights/RFB300_80_5.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
args = parser.parse_args()

if not os.path.exists(args.save_folder): # 保存计算结果的缓存
    os.mkdir(args.save_folder)

# 对应config.py
if args.dataset == 'VOC':
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    cfg = (COCO_300, COCO_512)[args.size == '512']

# 通过build_net加载模型
if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = COCO_mobile_300
else:
    print('Unkown version!')

# 调用prior_bbox.py内的PriorBox，生成RFBNet上密集采样的先验bbox
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    num_classes = (21, 81)[args.dataset == 'COCO']
    # 这个参数很屌，all_boxes：num_classes×num_images维度，每个元素又是一个5维的numpy，存(x1, x2, y1, y2, score)
    # 一个图像内有多个同类目标，就会对应多个bbox，也会被保存在all_boxes的同一个元素内
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    # 有缓存的话就直接读缓存，没缓存就从零计算
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return


    for i in range(num_images): # 每张图像的处理
        img = testset.pull_image(i) # 读图 + 预处理
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]]) #得到原图大小
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if cuda:
                x = x.cuda()
                scale = scale.cuda()

        _t['im_detect'].tic()
        out = net(x) # forward pass，结合RFBNet的forward，因为是全卷积网络，所以得到的也是feature map，
        # 但包含了loc(其实是offsets) + conf结果
        boxes, scores = detector.forward(out,priors) # 调用了detect,offsets + priors，再辅以decode操作，就可以得到最终的pred bbox + score
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores=scores[0]

        boxes *= scale # bbox float -> int，因为乘原图大小之前的boxer是归一化了的
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image

        _t['misc'].tic()
        # -----1----- NMS
        for j in range(1, num_classes): # 对每个类的pred bbox做nms操作，index从1开始，因为0对应bg类不必nms
            inds = np.where(scores[:, j] > thresh)[0] # 找到该类 j 下，所有cls score大于thresh的bbox，
            # 为什么选择大于thresh的bbox？因为score小于阈值的bbox，直接可以过滤掉，无需劳烦NMS

            if len(inds) == 0: # 没有bbox满足条件，说明图像中没有这个类的目标，或者这个类的目标漏检了
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue # 没有满足条件的bbox，返回空，跳过；
            c_bboxes = boxes[inds] # 筛选过后保存下来的bbox
            c_scores = scores[inds, j] # c_bboxes对应的confidence score
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False) # 类似concate操作组合成c_dets

            keep = nms(c_dets, 0.45, force_cpu=args.cpu) # nms,返回需保存的bbox index：keep
            c_dets = c_dets[keep, :] # 经过nms操作保存下来高置信度bbox
            # i 对应每张图像，j 对应图像中类别 j 的bbox清单
            all_boxes[j][i] = c_dets  #注：图像中同一类可能有多个目标，就会对应多个bbox，但都被保存在all_boxes的一个元素内

        # -----2----- top-k抑制(每张图最多检出top-k个bbox)
        if max_per_image > 0: # [:, -1]对应bbox的score，这个操作就是取出图像中的所有检出bbox的score
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            if len(image_scores) > max_per_image: # 检出bbox太多了，只取top-k
                image_thresh = np.sort(image_scores)[-max_per_image] # 卡住第top-k的bbox，作为thres baseline
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0] # 对应类j下的高score bbox，保留吧
                    all_boxes[j][i] = all_boxes[j][i][keep, :] # keep对应需要保留的bbox对应的index

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f: # 保存整理的缓存结果
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder) # 评估函数，可以结合voc_eval.py看看


if __name__ == '__main__':
    # load net
    img_dim = (300,512)[args.size=='512']
    num_classes = (21, 81)[args.dataset == 'COCO']
    net = build_net('test', img_dim, num_classes) # initialize detector，初始化网络结构，只是建立了网络结构，未灌入模型参数
    state_dict = torch.load(args.trained_model) # 读训练好的模型
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
    net.load_state_dict(new_state_dict) # 将读取的模型参数，灌入net中
    net.eval() # 现在模型参数就有啦,可以进入评估模式了
    print('Finished loading model!')
    print(net)
    # load data，加载测试数据
    if args.dataset == 'VOC':
        testset = VOCDetection(
            VOCroot, [('2007', 'test')], None, AnnotationTransform())
    elif args.dataset == 'COCO':
        testset = COCODetection(
            COCOroot, [('2014', 'minival')], None)
            #COCOroot, [('2015', 'test-dev')], None)
    else:
        print('Only VOC and COCO dataset are supported now!')
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
    # evaluation
    #top_k = (300, 200)[args.dataset == 'COCO']
    top_k = 200 # 每张图像上最多检出top_k个bbox
    detector = Detect(num_classes,0,cfg) # 调用detection.py里的Detect类，完成forward操作的detector
    save_folder = os.path.join(args.save_folder,args.dataset)
    rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']
    test_net(save_folder, net, detector, args.cuda, testset,
             BaseTransform(net.size, rgb_means, (2, 0, 1)), # resize + 减均值 + 通道调换
             top_k, thresh=0.01) # thresh=0.01，为什么这么小？可以结合mAP介绍的笔记

