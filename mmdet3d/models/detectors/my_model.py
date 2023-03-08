from mmcv.ops import Voxelization
from .. import builder
from ..builder import DETECTORS


@DETECTORS.register_module()
class MY_MODEL():
    def __init__(self, voxel_layer, voxel_encoder, train_cfg, test_cfg):
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
