from ..dense_heads import SMOKEMono3DHead
from ..dense_heads.anchor_free_mono3d_head import AnchorFreeMono3DHead
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from .single_stage_mono3d import SingleStageMono3DDetector

from ..builder import DETECTORS
from .. import builder
from mmdet.core import multi_apply
from torch.nn import functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from torch import nn as nn


@DETECTORS.register_module()
class MY_IMG_MODEL(SingleStageMono3DDetector):


    def __init__(self,
                 img_backbone,
                 img_neck,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 camera_stream=True,
                 pretrained=None,
                 **kwargs
                 ):
        super(MY_IMG_MODEL, self).__init__(img_backbone, img_neck, bbox_head, train_cfg,
                                          test_cfg, pretrained,**kwargs)



    def __repr__(self):
        s = self.__class__.__name__ + '('
        # s += 'num_input_features=' + str(self.num_input_features)
        s += ')'
        return s
