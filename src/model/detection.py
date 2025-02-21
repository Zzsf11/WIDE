import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.jit.annotations import Dict

from model.Backbone import backbone_mtf_fpn, unibackbone_fpn, mtf_backbone_msf, backbone_msf_mtf, backbone_msf
from model.Backbone import MTF


# Example function to create a Faster R-CNN model with a ResNet-18 backbone
def resnet50_mtf_msf_fasterrcnn(args):
    backbone = backbone_mtf_fpn('resnet50', fpn_num=args.msf, mode=args.mtf)
    backbone.out_channels = 256
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),  # 每个特征图对应一个尺度
        aspect_ratios=((0.5, 1.0, 2.0),) * 4         # 对所有特征图应用相同的宽高比
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
    # TODO: NMS 不能作用在不同类别之间
    model = FasterRCNN(backbone,
                       num_classes=args.num_classes+1, 
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,
                       min_size=1024, max_size=1333,
                    #    min_size=1200, max_size=2000,
                       box_nms_thresh=0.5,
                       box_score_thresh=0.05,
                    #    box_detections_per_img=2
                       )
    return model
