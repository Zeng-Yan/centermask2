from detectron2.layers import batched_nms
from detectron2.structures import Instances, Boxes
import torch


class BatchNMSOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bboxes, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size):
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
        nmsed_scores = scores[:, :max_total_size, 0]
        nmsed_classes = torch.ones(max_total_size, dtype=torch.long)
        nmsed_num = torch.Tensor([max_total_size])

        return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num

    @staticmethod
    def symbolic(g, bboxes, scores, score_thr, iou_thr, max_size_p_class, max_t_size):
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = g.op('BatchMultiClassNMS',
            bboxes, scores, score_threshold_f=score_thr, iou_threshold_f=iou_thr,
            max_size_per_class_i=max_size_p_class, max_total_size_i=max_t_size, outputs=4)
        return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num


def batch_nms_op(bboxes, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size):
    """
    boxes (torch.Tensor): boxes in shape (N, 4).
    scores (torch.Tensor): scores in shape (N, ).
    """

    if bboxes.dtype == torch.float32:
        bboxes = bboxes.reshape(1, bboxes.shape[0], -1, 4).half()
        scores = scores.reshape(1, scores.shape[0], -1).half()
    else:
        bboxes = bboxes.reshape(1, bboxes.shape[0].numpy(), -1, 4)
        scores = scores.reshape(1, scores.shape[0].numpy(), -1)

    nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = BatchNMSOp.apply(bboxes, scores,
        score_threshold, iou_threshold, max_size_per_class, max_total_size)
    nmsed_boxes = nmsed_boxes.float()
    nmsed_scores = nmsed_scores.float()
    nmsed_classes = nmsed_classes.long()
    dets = torch.cat((nmsed_boxes.reshape((max_total_size, 4)), nmsed_scores.reshape((max_total_size, 1))), -1)
    labels = nmsed_classes.reshape((max_total_size, ))
    return dets, labels


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

    if torch.onnx.is_in_onnx_export():
        dets, labels = batch_nms_op(boxes, scores, 0, nms_thresh, 100, 100)  # 第三个参数如何确定
        result = Instances(boxlist.image_size)
        result.pred_boxes = Boxes(dets[:, :4])
        result.scores = dets.permute(1, 0)[4, :]
        result.pred_classes = labels
        result.locations = boxlist.locations[:labels.shape[0]]
        return result

    keep = batched_nms(boxes, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist
