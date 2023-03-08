_base_ = [
    '../_base_/datasets/nus-3d-cam.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]
voxel_size = [0.1, 0.1, 0.1]
norm_cfg = None
DOUBLE_FLIP = False
# dataset_type = 'NuScenesDataset'
# data_root = '/home/wistful/work/mmdetection3d/data/nuscenes/'
# input_modality = dict(
#     use_lidar=True,
#     use_camera=True,
#     use_radar=False,
#     use_map=False,
#     use_external=False)
model = dict(
    type="MY_MODEL",
    voxel_layer=dict(
        max_num_points=32,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),
    voxel_encoder=dict(
        type='VoxelFeatureExtractorV3',
        num_input_features=4
    ),
    train_cfg=dict(),
    test_cfg=dict())

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4
)
