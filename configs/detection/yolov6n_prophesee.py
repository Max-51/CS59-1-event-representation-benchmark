model = dict(
    type="YOLOv6n",
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.25,
    backbone=dict(
        type="EfficientRep",
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        csp_e=0.5,
        fuse_P2=False,
        cspsppf=True,
    ),
    neck=dict(
        type="RepPANNeck",
        num_repeats=[12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512],
        csp_e=0.5,
    ),
    head=dict(
        type="EffiDeHead",
        in_channels=[128, 256, 512],
        num_layers=3,
        anchors=1,
        strides=[8, 16, 32],
        atss_warmup_epoch=0,
        iou_type="giou",
        use_dfl=True,
        reg_max=16,
        distill_weight={"class": 1.0, "dfl": 1.0},
    ),
)

solver = dict(
    optim="SGD",
    lr_scheduler="Cosine",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=0.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
)

data_aug = dict(
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,
)

training_mode = "conv_silu"
