_base_ = [
    '../_base_/default_runtime.py', '../_base_/datasets/sisr_x2_test_config.py'
]

experiment_name = 'llan_x2c48n5_1xb32-250k_div2k'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

load_from = None  # based on pre-trained x2 model
scale = 2
# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='LLANNet',
        in_channels=3,
        out_channels=3,
        mid_channels=48,
        up_channels=24,
        num_blocks=5,
        num_groups=4,
        upscale_factor=scale,
    ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    init_cfg=dict(
            # type='Pretrained',
            # checkpoint=pretrain_generator_url,
            # prefix='generator.'
        ),
    train_cfg=dict(),
    test_cfg=dict(metrics=['PSNR'], crop_border=scale),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=252),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='PackEditInputs')
]

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='PackEditInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data'

train_dataloader = dict(
    num_workers=8,
    batch_size=32,
    drop_last=True,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='meta_info_DIV2K800sub_GT.txt',
        metainfo=dict(dataset_type='div2k', task_name='sisr'),
        data_root=data_root + '/DIV2K/',
        data_prefix=dict(
            img='DIV2K_train_LR_bicubic/X2_sub', gt='DIV2K_train_HR_sub'),
        filename_tmpl=dict(img='{}', gt='{}'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=1,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='div2k10', task_name='sisr'),
        data_root=data_root + '/DIV2K/',
        data_prefix=dict(img='DIV2K_valid10_LR_bicubic/X2', gt='DIV2K_valid10_HR'),
        pipeline=val_pipeline))

val_evaluator = dict(
    type='EditEvaluator',
    metrics=[
        dict(type='MAE'),
        dict(type='PSNR', crop_border=scale),
        dict(type='SSIM', crop_border=scale),
    ])

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=250000, val_interval=1000)
val_cfg = dict(type='EditValLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-3, betas=(0.9, 0.999)))

# learning policy
param_scheduler = dict(
    type='CosineRestartLR', by_epoch=False, end=250_000, eta_min=1e-6, periods=[250000],
    restart_weights=[1])

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1000),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
