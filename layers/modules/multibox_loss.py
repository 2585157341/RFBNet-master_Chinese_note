import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
GPU = False
if torch.cuda.is_available():
    GPU = True

# RFBNet复用了SSD的loc + cls loss
class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).----正/负样本cls loss
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.----正样本loc loss
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)---- 难负样本挖掘
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N # x对应的是图像
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """


    def __init__(self, num_classes,overlap_thresh,prior_for_matching,bkg_label,neg_mining,neg_pos,neg_overlap,encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh # 判断正负样本的IoU阈值
        self.background_label = bkg_label
        self.encode_target = encode_target # 这个变量没用到
        self.use_prior_for_matching  = prior_for_matching #用于将anchor匹配groundtruth
        self.do_neg_mining = neg_mining #难分样本挖掘
        self.negpos_ratio = neg_pos # 挖掘后的正负样本比例，1:3
        self.neg_overlap = neg_overlap # 这个变量没用到
        self.variance = [0.1,0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data = predictions # (batch_size,num_priors,4) / (batch_size,num_priors,num_classes)
        priors = priors # (num_priors,4)
        num = loc_data.size(0) # batch_size
        num_priors = (priors.size(0)) # num_priors
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes,
        #将batch_size内每张图像中pred bbox与gt bbox做匹配，进一步计算loss
        loc_t = torch.Tensor(num, num_priors, 4) #gt_loc定义好
        conf_t = torch.LongTensor(num, num_priors) #gt_conf定义好
        for idx in range(num): # 在batch_size中遍历每张图
            truths = targets[idx][:,:-1].data # 前4个参数是gt bbox
            labels = targets[idx][:,-1].data # 最后1个参数是gt label
            defaults = priors.data # 这个defaults定义很好，就是SSD中预定义的default box
            # match函数在box_utils.py里,返回的是conf_t + loc_t，对应gt_cls，gt_offsets
            match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)
            # match这个函数给每个ground truth匹配了最好的priors，给每个priors匹配最好的ground truth
            # 经过encode后的offset([g_cxcy, g_wh])->loc_t,top class label for each prior->conf_t
        if GPU:
            loc_t = loc_t.cuda()#刚刚返回的gt的位置loc_t
            conf_t = conf_t.cuda()#刚刚返回的gt的类别conf_t
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)

        pos = conf_t > 0 # 从gt里取出正样本，正样本才需要计算loc loss

        # Localization Loss (Smooth L1)
        # loc Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data) # 相当于取出所有正样本对应的index位置
        loc_p = loc_data[pos_idx].view(-1,4) #取出预测结果里的正样本pred bbox，结合RFB_Net_vgg.py和detection.py，可以发现其实预测的也是offsets
        # 也就是保留与计算出来的positive的default box所对应的卷积生成的encode offset（相当于预测的）
        loc_t = loc_t[pos_idx].view(-1,4) # gt offsets也reshape
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False) # Localization Loss (Smooth L1)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1,self.num_classes) # batch_size内所有pred bbox进行reshape操作
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1)) #把conf_t变为和batch_conf相同shape，正负样本分类loss

        # Hard Negative Mining，仅筛选难负样本计算loss
        loss_c[pos.view(-1,1)] = 0 # filter out pos boxes for now，OHEM操作不考虑正样本，仅在负样本上操作
        loss_c = loss_c.view(num, -1) # 按图像归类各个负样本
        _,loss_idx = loss_c.sort(1, descending=True) # loss降序排序，那么仅需要选择前面的高loss即可
        _,idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1,keepdim=True)  # 正样本数量
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1) # 由正样本数量按1：3比例得出需保留的难负样本数量
        neg = idx_rank < num_neg.expand_as(idx_rank) # 结合_,idx_rank = loss_idx.sort(1)理解，为了取出难neg pred bbox

        # Confidence Loss Including Positive and Negative Examples最终只有难负样本loss + 正样本loss参与模型参数的更新
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes) # pred bbox的预测结果，经过难负样本挖掘后留存下来的
        targets_weighted = conf_t[(pos+neg).gt(0)] # 剩余需要计算cls gt label，包含了正负样本的gt label
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False) # 分类的交叉熵损失函数

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = max(num_pos.data.sum().float(), 1) # N: number of matched default boxes
        loss_l/=N
        loss_c/=N
        return loss_l,loss_c
