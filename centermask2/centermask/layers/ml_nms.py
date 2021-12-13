import torch

from detectron2.layers import batched_nms
from detectron2.structures import Instances, Boxes


class BatchNMSOp(torch.autograd.Function):
    @staticmethod
    # 添加了labels
    def forward(ctx, bboxes, scores, labels, score_threshold, iou_threshold, max_size_per_class, max_total_size):
        """
        boxes (torch.Tensor): boxes in shape (batch, N, C, 4).
        scores (torch.Tensor): scores in shape (batch, N, C).
        return:
            nmsed_boxes: (1, N, 4)
            nmsed_scores: (1, N)
            nmsed_classes: (1, N)
            nmsed_num: (1,)
        """

        # Phony implementation for onnx export
        nmsed_boxes = bboxes[:, :max_total_size, 0, :]
        nmsed_scores = scores[:, :max_total_size, 0]  # 这里添加的对scores的处理，伪传递应该是可行的
        # nmsed_classes = torch.ones(nmsed_scores.shape[1], dtype=torch.long)
        nmsed_classes = labels[:, :max_total_size, 0]
        nmsed_num = torch.Tensor([nmsed_scores.shape[1]])

        return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num  # 返回顺序是否正确？

    @staticmethod
    # 添加了labels
    def symbolic(g, bboxes, scores, labels, score_thr, iou_thr, max_size_p_class, max_t_size):
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = g.op('BatchMultiClassNMS',
            bboxes, scores, labels, score_threshold_f=score_thr, iou_threshold_f=iou_thr,
            max_size_per_class_i=max_size_p_class, max_total_size_i=max_t_size, outputs=4)
        return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num


def batch_nms_op(bboxes, scores, labels, score_threshold, iou_threshold, max_size_per_class, max_total_size):
    """
    boxes (torch.Tensor): boxes in shape (N, 4).
    scores (torch.Tensor): scores in shape (N, ).
    """

    if bboxes.dtype == torch.float32:
        bboxes = bboxes.reshape(1, bboxes.shape[0], -1, 4).half()
        scores = scores.reshape(1, scores.shape[0], -1).half()
        labels = labels.reshape(1, labels.shape[0], -1).half()
    else:
        bboxes = bboxes.reshape(1, bboxes.shape[0].numpy(), -1, 4)
        scores = scores.reshape(1, scores.shape[0].numpy(), -1)
        labels = labels.reshape(1, labels.shape[0].numpy(), -1)

    nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = BatchNMSOp.apply(bboxes, scores, labels,
        score_threshold, iou_threshold, max_size_per_class, max_total_size)

    fst_size = nmsed_boxes.shape[1]
    nmsed_boxes = nmsed_boxes.float().reshape((fst_size, 4))
    nmsed_scores = nmsed_scores.float().reshape((fst_size,))
    nmsed_classes = nmsed_classes.long().reshape((fst_size, ))

    return nmsed_boxes, nmsed_scores, nmsed_classes


def ml_nms(boxlist, nms_thresh, max_proposals=-1,
           score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.
    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.pred_boxes.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes

    if torch.onnx.is_in_onnx_export():  # 导出onnx时替换自定义算子
        # 第三个参数如何确定，原本是没有这个scores的，这个参数到底如何传进去
        boxes, scores, labels = batch_nms_op(boxes, scores, labels, 0, nms_thresh, 100, 100)
        result = Instances(boxlist.image_size)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = labels
        result.locations = boxlist.locations[:labels.shape[0]]
        return result

    keep = batched_nms(boxes, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]

    return boxlist
