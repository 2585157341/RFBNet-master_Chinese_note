import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from math import sqrt as sqrt
from itertools import product as product


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.生成feature map上预定义的anchor box
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim'] # 输入RFBNet的图像尺度，这里假设512
        # number of priors for feature map location (300有6个，而512有7个feature map)
        self.num_priors = len(cfg['aspect_ratios']) # 各个feature map上预定义的anchor长宽比清单，与检测分支的数量对应
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps'] # 特征金字塔层上各个feature map尺度
        self.min_sizes = cfg['min_sizes']  # 预定义的anchor尺度的短边，越深层感受野越大故分配的anchor尺度越大
        # SSD中6个default bbox如何定义的？2:1 + 1:2 + 1:3 + 3:1+两个1:1长宽比的anchor，
        # 但SSD定义了一个根号2尺度的anchor，max_sizes类似，但并不是严格对应的
        self.max_sizes = cfg['max_sizes'] # 预定义的anchor尺度的长边
        self.steps = cfg['steps'] # 每个尺度检测特征图分支的stride（即与输入的缩小倍数）
        self.aspect_ratios = cfg['aspect_ratios'] # feature map上每个pix上预定义6/7个anchor
        self.clip = cfg['clip'] # 位置校验
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []  # 用于保存所有feature map上预定义的anchor
        for k, f in enumerate(self.feature_maps): #对特征金字塔的各个检测分支，每个feature map上each-pixel都做密集anchor采样
            for i, j in product(range(f), repeat=2): # 笛卡尔积repeat后的f，组成很多二维元组，可以开始密集anchor采样了
                f_k = self.image_size / self.steps[k] #当前检测分支的特征图大小
                cx = (j + 0.5) / f_k #当前检测分支的归一化后的anchor中心坐标cx
                cy = (i + 0.5) / f_k  # 以上三步操作，就相当于从feature map位置映射至归一化原图，float型


                s_k = self.min_sizes[k]/self.image_size #归一化后的当前检测分支对应的anchor的min_size
                mean += [cx, cy, s_k, s_k]  # 第一个anchor添加，1:1长宽比

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size)) #sqrt(min_sizes[k]*max_sizes[k]/(512*512))
                mean += [cx, cy, s_k_prime, s_k_prime]# 第二个anchor添加，1:1长宽比，尺度与第一个anchor不一样，和SSD对应上了~~~

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:#不管是[2]还是[2,3]都循环当前aspect_ratio内部元素
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]# 如是[2,3]，生成2:1和3:1的anchor，如是[2]则生成2:1的anchor
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]# 如是[2,3]，生成1:2和1:3，如是[2]，生成1:2的anchor

        # 总结：
        # 1 每个检测分支feature map上each-pixel对应6 / 7个anchor，长宽比：2:1 + 1:2 + 1:3 + 3:1 + 1:1 + 1:1，后两个1:1的anchor对应的尺度有差异；
        # 2 跟SSD还是严格对应的，每个feature map上anchor尺度唯一(2:1 + 1:2 + 1:3 + 3:1 + 1:1这五个anchor的尺度还是相等的，面积相等)，仅最后的1:1 anchor尺度大一点；
        # 3 所有feature map上所有预定义的不同尺度、长宽比的anchor保存至mean中；

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4) # 操作类似reshape，规则化输出
        if self.clip:
            output.clamp_(max=1, min=0)# float型坐标校验
        return output
