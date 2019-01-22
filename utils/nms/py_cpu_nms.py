# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0] # pred bbox top_x
    y1 = dets[:, 1] # pred bbox top_y
    x2 = dets[:, 2] # pred bbox bottom_x
    y2 = dets[:, 3] # pred bbox bottom_y
    scores = dets[:, 4] # pred bbox cls score
    # pred bbox areas,
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) #注意要加1才是真正的边长像素（如x2=6,x1=2,则长度实际为5 ）
    order = scores.argsort()[::-1] # 对pred bbox按score做降序排序

    keep = [] # NMS后，保留的pred bbox
    while order.size > 0:
        i = order[0] # top-1 score bbox
        keep.append(i) # top-1 score的话，自然就保留了
        xx1 = np.maximum(x1[i], x1[order[1:]]) # top-1 bbox（score最大）与order中剩余bbox计算NMS
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)#计算IOU

        inds = np.where(ovr <= thresh)[0]# 这个操作可以对代码断点调试理解下，我们希望剔除所有与当前top-1 bbox IoU > thresh的冗余bbox，
        # 那么保留下来的bbox，自然就是ovr <= thresh的非冗余bbox，其inds保留下来，作进一步筛选
        order = order[inds + 1]# 保留有效bbox，就是这轮NMS未被抑制掉的幸运儿，为什么+1？因为ind=0就是这轮NMS的top-1，
        # 剩余有效bbox在IoU计算中与top-1做的计算，inds对应回原数组，自然要做 +1 的映射，接下来就是重复迭代的循环


    return keep # 最终NMS结果返回

