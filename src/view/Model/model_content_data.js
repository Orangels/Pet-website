export let model_content_data = {
    detection:{
        coco:{
            Model_title_data:{
                title:'Detection',
                text:'For COCO dataset, training imageset is train2017 and validation imageset is val2017. The COCO metric, Average Precision (AP) with IoU threshold 0.5:0.95 (averaged 10 values, AP 0.5:0.95), 0.5 (AP 0.5) and 0.75 (AP 0.75) are reported together in the format (AP 0.5:0.95)/(AP 0.5)/(AP 0.75). For object detection task, only box overlap based AP is evaluated and reported.for instence segmentation both box overlap and mask   overlap AP are evaluated and reported.',
                title_1:'Common Settings and Notes',
                block:[
                    {
                        title:'Hardware:',
                        text:['8 NVIDIA GTX Titan Xp GPUs ',
                            'Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz']
                    },
                    {
                        title:'Software environment: ',
                        text:['Python 3.5            Pet 1.0',
                        'CUDA 9.0.176      CUDNN7.0.4    NCCL 2.1.15']
                    },
                    {
                        title:'Train stage : ',
                        text:['Batchsize is 16 (2img/gpu), data',
                        'enhancement only uses horizontal flip.']
                    },
                ]
            },
            detail_content:[
                [
                    ['faster-rcnn:', 'mask-rcnn:', 'R-50:', 'R50-FPN:', 'R50-C4:', 'se-R50:', 'X-101-32x8d:', 'A-101-32x8d:',
                        'AR-101-32x8d:', 'AX-101-32x8d:',],
                    ['end-to-end Faster-rcnn structure', 'Mask-rcnn structure', 'Resnet 50  (R-18 R -29 R101 the same)', 'FPN structure of Resnet 50',
                        'C4 structure of Resnet 50', 'Resnet 50 with Se module', 'Resnext101-32x8', 'AirNet-101-32x8d', 'Alignresnet101-32x8',
                        'AlignresneXt101-32x8',]
                ],
                [
                    ['ms:', 'GN:', 'MV1:', 'DCN:', 'MDCN:', '1x  / 2x:', 'MDCN@C3:', 'MDCN@C345:'],
                    ['multi-scale', 'GN', 'MobilenetV1', 'Deformable', 'Modulated Deformable', 'learn rate 1x /2x',
                         'Add deformable convolution in the resnet stage 3',
                        'Add the Modulated Deformable convolution in the resnet stage 3 4 5',]
                ]
            ]
        }
    },
    classification:{
        cifar:{
            Model_title_data:{
                title:'Classification',
                text:'For Imagenet dataset ,we use Top1 and Top5 to evaluate the accuracy of the model, and give the running speed and memory occupancy of each model.',
                title_1:'Common Settings and Notes',
                block:[
                    {
                        title:'Hardware:',
                        text:['8 NVIDIA GTX Titan Xp GPUs ',
                        // 'Intel Xeon 4114 CPU @ 2.20GHz',
                            'Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz'
                        ]
                    },
                    {
                        title:'Software environment: ',
                        text:['Python 3.5            Pet 1.0',
                        'CUDA 9.0.176      CUDNN7.0.4    NCCL 2.1.15']
                    },
                    {
                        title:'Test  stage : ',
                        text:['Throughputs are measured with single TITAN Xp ',
                            'and batch size 1.']
                    },
                ]
            },
            detail_content:[
                [
                    ['faster-rcnn:', 'mask-rcnn:', 'R-50:', 'R50-FPN:', 'R50-C4:', 'se-R50:', 'X-101-32x8d:', 'A-101-32x8d:',
                        'AR-101-32x8d:', 'AX-101-32x8d:',],
                    ['end-to-end Faster-rcnn 结构', 'Mask-rcnn结构', 'Resnet 50  (R-18 R -29 R101 相同)', 'Resnet 50的FPN结构',
                        'Resnet 50 C4结构', 'Resnet 50 加Se模块', 'Resnext101-32x8', 'AirNet-101-32x8d', 'Alignresnet101-32x8',
                        'AlignresneXt101-32x8',]
                ],
                [
                    ['ms:', 'GN:', 'MV1:', 'DCN:', 'MDCN:', '1x  / 2x:',  'MDCN@C3:', 'MDCN@C345:'],
                    ['multi-scale', 'GN', 'MobilenetV1', 'Deformable', 'Modulated Deformable', '学习率策略 1x /2x',
                         '在 resnet 阶段 3 加入 deformable convolution',
                        '在 resnet 阶段 3 4 5  加入 Modulated Deformable convolution',]
                ]
            ]
        },
        imageNet:{
            Model_title_data:{
                title:'Classification',
                text:'For Imagenet dataset ,we use Top1 and Top5 to evaluate the accuracy of the model, and give the running speed and memory occupancy of each model.',
                title_1:'Common Settings and Notes',
                block:[
                    {
                        title:'Hardware:',
                        text:['8 NVIDIA GTX Titan Xp GPUs ',
                        // 'Intel Xeon 4114 CPU @ 2.20GHz',
                            'Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz'
                        ]
                    },
                    {
                        title:'Software environment: ',
                        text:['Python 3.5            Pet 1.0',
                        'CUDA 9.0.176      CUDNN7.0.4    NCCL 2.1.15']
                    },
                    {
                        title:'Test  stage : ',
                        text:['Throughputs are measured with single TITAN Xp',
                            'and batch size 1.']
                    },
                ]
            },
            detail_content:[
                [
                    ['faster-rcnn:', 'mask-rcnn:', 'R-50:', 'R50-FPN:', 'R50-C4:', 'se-R50:', 'X-101-32x8d:', 'A-101-32x8d:',
                        'AR-101-32x8d:', 'AX-101-32x8d:',],
                    ['end-to-end Faster-rcnn 结构', 'Mask-rcnn结构', 'Resnet 50  (R-18 R -29 R101 相同)', 'Resnet 50的FPN结构',
                        'Resnet 50 C4结构', 'Resnet 50 加Se模块', 'Resnext101-32x8', 'AirNet-101-32x8d', 'Alignresnet101-32x8',
                        'AlignresneXt101-32x8',]
                ],
                [
                    ['ms:', 'GN:', 'MV1:', 'DCN:', 'MDCN:', '1x  / 2x:',  'MDCN@C3:', 'MDCN@C345:'],
                    ['multi-scale', 'GN', 'MobilenetV1', 'Deformable', 'Modulated Deformable', '学习率策略 1x /2x',
                         '在 resnet 阶段 3 加入 deformable convolution',
                        '在 resnet 阶段 3 4 5  加入 Modulated Deformable convolution',]
                ]
            ]
        }
    }
}


export let model_content_data_CN = {
    detection:{
        coco:{
            Model_title_data:{
                title:'检测',
                text:'对于COCO数据集，training imageset为train2017, validation imageset为val2017。COCO指标，IoU阈值为0.5:0.95(平均10个值，AP 0.5:0.95)、0.5 (AP 0.5)和0.75 (AP 0.75)的平均精度(AP)以格式(AP 0.5:0.95)/(AP 0.5)/(AP 0.75)同时报告。对于目标检测任务，只评估和报告基于框重叠的AP。对于实例分割，分别对box重叠和mask重叠AP进行了评估和报告。',
                title_1:'常见设置及注意事项',
                block:[
                    {
                        title:'硬件:',
                        text:['8 NVIDIA GTX Titan Xp GPUs ',
                        '英特尔 Xeon E5-2630 CPU @ 2.20GHz']
                    },
                    {
                        title:'软件环境: ',
                        text:['Python 3.5     Pet 1.0',
                        'CUDA 9.0.176 CUDNN7.0.4 NCCL 2.1.15']
                    },
                    {
                        title:'训练阶段 : ',
                        text:['批处理大小为16 (2img/gpu)，数据增强只',
                        '使用水平翻转']
                    },
                ]
            },
            detail_content:[
                [
                    ['faster-rcnn:', 'mask-rcnn:', 'R-50:', 'R50-FPN:', 'R50-C4:', 'se-R50:', 'X-101-32x8d:', 'A-101-32x8d:',
                        'AR-101-32x8d:', 'AX-101-32x8d:',],
                    ['end-to-end Faster-rcnn 结构', 'Mask-rcnn结构', 'Resnet 50  (R-18 R -29 R101 相同)', 'Resnet 50的FPN结构',
                        'Resnet 50 C4结构', 'Resnet 50 加Se模块', 'Resnext101-32x8', 'AirNet-101-32x8d', 'Alignresnet101-32x8',
                        'AlignresneXt101-32x8',]
                ],
                [
                    ['ms:', 'GN:', 'MV1:', 'DCN:', 'MDCN:', '1x  / 2x:', 'MDCN@C3:', 'MDCN@C345:'],
                    ['multi-scale', 'GN', 'MobilenetV1', 'Deformable', 'Modulated Deformable', '学习率策略 1x /2x',
                         '在 resnet 阶段 3 加入 deformable convolution',
                        '在 resnet 阶段 3 4 5  加入 Modulated Deformable convolution',]
                ]
            ]
        }
    },
    classification:{
        cifar:{
            Model_title_data:{
                title:'分类',
                text:'对于ImageNet分类数据集，我们分别使用Top1和top5对模型的精度尽行评估，同时给出每个模型的运行速度和显存占用。',
                title_1:'常见设置及注意事项',
                block:[
                    {
                        title:'硬件:',
                        text:['8 NVIDIA GTX Titan Xp GPUs ',
                            '英特尔 Xeon E5-2630 CPU @ 2.20GHz']
                    },
                    {
                        title:'软件环境: ',
                        text:['Python 3.5     Pet 1.0',
                            'CUDA 9.0.176 CUDNN7.0.4 NCCL 2.1.15']
                    },
                    {
                        title:'测试阶段 : ',
                        text:['详细参数相见对应的yaml。']
                    },
                ]
            },
            detail_content:[
                [
                    ['faster-rcnn:', 'mask-rcnn:', 'R-50:', 'R50-FPN:', 'R50-C4:', 'se-R50:', 'X-101-32x8d:', 'A-101-32x8d:',
                        'AR-101-32x8d:', 'AX-101-32x8d:',],
                    ['end-to-end Faster-rcnn 结构', 'Mask-rcnn结构', 'Resnet 50  (R-18 R -29 R101 相同)', 'Resnet 50的FPN结构',
                        'Resnet 50 C4结构', 'Resnet 50 加Se模块', 'Resnext101-32x8', 'AirNet-101-32x8d', 'Alignresnet101-32x8',
                        'AlignresneXt101-32x8',]
                ],
                [
                    ['ms:', 'GN:', 'MV1:', 'DCN:', 'MDCN:', '1x  / 2x:',  'MDCN@C3:', 'MDCN@C345:'],
                    ['multi-scale', 'GN', 'MobilenetV1', 'Deformable', 'Modulated Deformable', '学习率策略 1x /2x',
                         '在 resnet 阶段 3 加入 deformable convolution',
                        '在 resnet 阶段 3 4 5  加入 Modulated Deformable convolution',]
                ]
            ]
        },
        imageNet:{
            Model_title_data:{
                title:'分类',
                text:'对于ImageNet分类数据集，我们分别使用Top1和top5对模型的精度尽行评估，同时给出每个模型的运行速度和显存占用。',
                title_1:'常见设置及注意事项',
                block:[
                    {
                        title:'硬件:',
                        text:['8 NVIDIA GTX Titan Xp GPUs ',
                            '英特尔 Xeon E5-2630 CPU @ 2.20GHz']
                    },
                    {
                        title:'软件环境: ',
                        text:['Python 3.5     Pet 1.0',
                            'CUDA 9.0.176 CUDNN7.0.4 NCCL 2.1.15']
                    },
                    {
                        title:'测试阶段 : ',
                        text:['模型的运行速度测试时，我们使用1 块TITAN Xp，',
                        'batch 为1 进行测试。']
                    },
                ]
            },
            detail_content:[
                [
                    ['faster-rcnn:', 'mask-rcnn:', 'R-50:', 'R50-FPN:', 'R50-C4:', 'se-R50:', 'X-101-32x8d:', 'A-101-32x8d:',
                        'AR-101-32x8d:', 'AX-101-32x8d:',],
                    ['end-to-end Faster-rcnn 结构', 'Mask-rcnn结构', 'Resnet 50  (R-18 R -29 R101 相同)', 'Resnet 50的FPN结构',
                        'Resnet 50 C4结构', 'Resnet 50 加Se模块', 'Resnext101-32x8', 'AirNet-101-32x8d', 'Alignresnet101-32x8',
                        'AlignresneXt101-32x8',]
                ],
                [
                    ['ms:', 'GN:', 'MV1:', 'DCN:', 'MDCN:', '1x  / 2x:',  'MDCN@C3:', 'MDCN@C345:'],
                    ['multi-scale', 'GN', 'MobilenetV1', 'Deformable', 'Modulated Deformable', '学习率策略 1x /2x',
                         '在 resnet 阶段 3 加入 deformable convolution',
                        '在 resnet 阶段 3 4 5  加入 Modulated Deformable convolution',]
                ]
            ]
        },
    }
}