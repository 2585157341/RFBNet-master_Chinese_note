import torch
import torch.nn as nn
import math
import numpy as np
if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn

# 主要是对torch.tensor格式的bbox处理

# bbox格式转换：(cx, cy, w, h) -> (x1, y1, x2, y2)
def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

# bbox格式转换：(x1, y1, x2, y2) -> (cx, cy, w, h)
def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    # 注释已经很好地描述了本函数的功能，其实就是计算IoU中的I
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    # 计算I中的bbox长、宽，通过min=0规则化，如果两个bbox没有交集，inter中自然有0，此时I就等于0了
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    # 计算IoU中I的面积
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter # A ∪ B
    return inter / union  # [A,B]，这个就直接是IoU了

# 这个也是计算IoU，用在data_augment.py中，numpy格式的操作
def matrix_iou(a,b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2]) # top-left
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:]) # right-bottom
    #I，后面的all函数就是校验两个bbox是否有交集，若无交集，则对应area为0，与torch.clamp((max_xy - min_xy), min=0)目的相同
    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)

# 相当于anchor与gt bbox的匹配，返回的loc_t(对应offsets)、conf_t(对应cls_id)
def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index，因为每次都是batch size图像输入训练的
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index，直接在gt box和priors(也即anchor)上计算Iou
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)，双路匹配，分别为gt bbox、anchor寻找最佳匹配
    # [1,num_objects] best prior for each ground truth，为gt bbox寻找最佳匹配的anchor
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior，为anchor寻找最佳匹配的gt bbox
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior，确认过眼神，你是对的人
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap，为gt bbox分配anchor
    # 这里稍微有点绕，走了个回路，代码写得多的话，能明白其中含义的，双重校验
    # 最终目的为：确保gt bbox能匹配到max IoU的anchor，gt bbox匹配优先
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    # 分配gt bbox与label
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx]          # Shape: [num_priors]，分配cls label
    conf[best_truth_overlap < threshold] = 0  # label as background，IoU过小，直接分配bg label
    loc = encode(matches, priors, variances) # 计算offsets，也是multibox_loss中所需要的
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior

# 这个操作同样结合RCNN中的supplements，就是decode的反操作，最终计算的是offsets
# priors：预定义的anchor，(cx, cy, w, h)；
# matched：基于IoU与priors匹配的gt bbox；
def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    # 下面两步操作就是对cx, cy计算offsets
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss，与RCNN supplements严格对应
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]，concate

# 与encode操作类似，增加了一个offsets而已
def encode_multi(matched, priors, offsets, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center，对cx, cy计算offsets
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2] - offsets[:,:2]
    # encode variance
    #g_cxcy /= (variances[0] * priors[:, 2:])
    g_cxcy.div_(variances[0] * offsets[:, 2:])
    # match wh / prior wh，对w, h计算log-style的offset
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

# Adapted from https://github.com/Hakuyume/chainer-ssd
# decode函数功能很容易理解，结合RCNN中的supplements，一下就能明白
# 就是我们现在有了在feature map上预定义的密集采样的prior，又有了网络预测输出的loc offset，接下来就是计算bbox reg回归了
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    # priors: (cx, cy, w, h)，loc：pred offsets，结合RCNN中介绍的转换公式就好理解了
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    # bbox格式转换：(cx, cy, w, h) -> (x1, y1, x2, y2)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
# 与decode操作方式类似，只是增加了一个offsets变量而已
def decode_multi(loc, priors, offsets, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + offsets[:,:2]+ loc[:, :2] * variances[0] * offsets[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

# 这个公式有点类似softmax，对应的cls loss，这里只是通过exp、log做了计算，
# 结合softmax公式，只是为了做bp反向传播时计算更简洁
def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max() # 求x中的最大值
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    这里面有一个细节，NMS仅用于测试阶段,因为train阶段只对pos样本+3*num(pos)难负样本训练，做了ohem了...
    所以没有必要再在此基础上进行box的非极大值抑制
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1) # IoU初步准备
    v, idx = scores.sort(0)  # sort in ascending order,不过是升序操作，非降序
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals，依然是升序的结果
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0: #若所有pred bbox都处理完毕，就可以结束循环啦~
        i = idx[-1]  # index of current largest val，top-1 score box，因为是升序的，所有返回index = -1的最后一个元素即可
        # keep.append(i)
        keep[count] = i
        count += 1 # 不仅记数NMS保留的bbox个数，也作为index存储bbox
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view，top-1已保存，不需要了~~~
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i]) # 对应 np.maximum(x1[i], x1[order[1:]])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0) #截取w>0
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        # 以下两步操作做了个优化，area已经计算好了，就可以直接根据idx读取结果了，area[i]同理，避免了不必要的冗余计算
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i] # 就是area(a) + area(b) - i
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)] # 这一轮NMS操作，IoU阈值小于overlap的idx，就是需要保留的bbox，其他的就直接忽略吧，并进行下一轮计算
    return keep, count


