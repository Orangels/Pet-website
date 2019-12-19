export let CIHP_Parsing_R_CNN = {
    key: 'CIHP_Parsing_R_CNN',
    dataSource: [
        {title:'Train Parsing R-CNN on CIHP'},
        {text:'This tutorial will introduce the main steps of using Pet to train and test Parsing R-CNN model for multi-human parsing. Here we will show you how to construct Parsing R-CNN model by combining various functional modules provided by Pet with only explaining using of components. For some details, please refer to the corresponding parts of system components. Before reading this tutorial, we strongly recommend that you read the original papers [Parsing R-CNN](https://arxiv.org/abs/1811.12596)\\[1\\], [FPN](https://arxiv.org/pdf/1808.00157v1)\\[2\\] to learn more about Parsing R-CNN.\n'},
        {
            note:[
                {text:'First turn to the tutorial [CIHP](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87/Prepare%20Datasets.md#cihp) and prepare the CIHP dataset on the disk.\n'}
            ]
        },
        {text:'If you have rich experience in multi-human parsing algorithms, you can also run `$Pet/tools/rcnn/train_net.py` script in Pet to start training your Parsing R-CNN model.\n'},
        {text:'Use case：'},
        {ul:'Train end to end Parsing R-CNN model on `CIHP_train` with 8 gpus:'},
        {shell:'```\n' +
                '    cd $Pet\n' +
                '\n' +
                '    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 tools/rcnn/train_net.py \n' +
                '    --cfg cfgs/rcnn/CIHP/e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml\n' +
                '```\n'},
        {ul:'Train Parsing R-CNN model on `CIHP_train` with 8 gpus：'},
        {shell:'```\n' +
                '    cd $Pet\n' +
                '\n' +
                '    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/rcnn/test_net.py \n' +
                '    --cfg cfgs/rcnn/CIHP/e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml\n' +
                '```\n'},
        {text:'Before performing any operations related to model training and testing, a specified `yaml` file should be selected to specify the requirements and settings for dataset, model structure, optimization strategy and other important parameters during training. This tutorial takes `$Pet/cfgs/rcnn/CIHP/e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml` as an example to explain the training. The key configurations needed in the training process will guide all the steps and details of the Parsing R-CNN model training and testing. For all parameters, see [$Pet/cfgs/rcnn/CIHP/e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml](https://github.com/BUPT-PRIV/Pet-dev/blob/master/cfgs/rcnn/CIHP/e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml).\n'},
        {part_title:'Data Loading'},
        {text:'Make sure that the CIHP dataset is stored on your disk and the file structure of the CIHP dataset is organized according to the file structure in [Data Preparation](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%9E%B6%E6%9E%84/component-collects/%E6%95%B0%E6%8D%AE%E5%88%B6%E5%A4%87/Data%20Preparation.md). Then we can start loading the `CIHP_train` training set.\n' +
                '\n' +
                '```Python\n' +
                '    # Create data loder\n' +
                '    datasets = build_dataset(cfg.TRAIN.DATASETS, is_train=True)\n' +
                '    train_loader = make_train_data_loader(\n' +
                '        datasets,\n' +
                '        is_distributed=args.distributed,\n' +
                '        start_iter=scheduler.iteration,\n' +
                '    )\n' +
                '```\n'},
        {
            ul:[
                'At present, training the region-based two-stage instance analysis models needs to scale the length and width of the image in equal proportion. In Parsing R-CNN, the short edge of the image is scaled to a series of basic sizes to enrich the scale of human instances in CIHP data. In the YAML file, `TRAIN.SCALES` is set to `(512, 640, 704, 768, 800, 864)`, `TRAIN.MAX_SIZE` is set to 1400 to limit the maximum size of the image. ',
                'Parsing R-CNN also randomly flips the training data horizontally to augment the data and enhance the generalization of the model. The visualization results of the transformed images and their annotations are as follows:'
            ]
        },
        {img:'Parsing'},
        {text:'Visualization of pictures, detection boxes, keypoints and human part segmentation masks in Dataloader before and after transformation', type:'center'},
        {ul:'Data loading component not only completes the reading of image data and annotation information, but also generates the training label of RPN network while collecting data of each batch. Data loading component outputs data of each batch, including image data, the category of objects in the picture, the bounding box of objects, and the segmentation masks as the number of objects(each mask contains only one foreground target\'s mask).'},
        {yaml:'```\n' +
                '    data: (1, 3, 800, 1196)\n' +
                '    label: (1, 6)\n' +
                '    box: (1, 6, 4)\n' +
                '    mask: (1, 6, 800, 1196)\n' +
                '```\n'},
        {part_title: 'Parsing R-CNN Network'},
        {text:'In Pet, model builder `Generalized_RCNN` is used to build the Parsing R-CNN network, It is only necessary to add `PRCNN` parameters to the YAML file to construct a branch network of Parsing R-CNN for human body part analysis:\n' +
                '\n' +
                '```\n' +
                '    PRCNN:\n' +
                '      ROI_XFORM_RESOLUTION: 32\n' +
                '      ROI_XFORM_SAMPLING_RATIO: 2\n' +
                '      RESOLUTION: 128\n' +
                '      NUM_PARSING: 20\n' +
                '      ROI_BATCH_SIZE: 32\n' +
                '      FINEST_LEVEL_ROI: True\n' +
                '      ROI_PARSING_HEAD: "roi_asppv3_convX_head"\n' +
                '      GCE_HEAD:\n' +
                '        NUM_CONVS_BEFORE_ASPPV3: 0\n' +
                '        NUM_CONVS_AFTER_ASPPV3: 4\n' +
                '```\n'},
        {text:'In `Generalized_RCNN`, you only need to add `Parsing_RCNN` branch network construction code after `Fast_RCNN`, and the rest of it same as the model construction in [Mask R-CNN](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E5%9C%A8MSCOCO2017%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E8%AE%AD%E7%BB%83Mask%20R-CNN%E6%A8%A1%E5%9E%8B.md#mask-r-cnn%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84).\n' +
                '\n' +
                '```Python\n' +
                '    class Generalized_RCNN(nn.Module):\n' +
                '        def __init__(self, is_train=True):\n' +
                '            super().__init__()\n' +
                '\n' +
                '            ...\n' +
                '\n' +
                '            if not cfg.MODEL.RPN_ONLY:\n' +
                '                self.Fast_RCNN = FastRCNN(self.dim_in, self.spatial_scale)\n' +
                '\n' +
                '                if cfg.MODEL.PARSING_ON:\n' +
                '                    self.Parsing_RCNN = ParsingRCNN(self.dim_in, self.spatial_scale)\n' +
                '```\n' +
                '\n' +
                'Parsing R-CNN network includes regional proposal network(RPN), feature pyramid network(FPN) and object detection sub network (FastRCNN). The most important sub network of Parsing R-CNN is **human parts parsing branch** (ParsingRCNN). The main structure of `ParsingRCNN` network is `roi_gce_head`, which is mainly composed of **Geometric and Context Encoding(GCE) and **feature conversion module**.\n' +
                '\n' +
                '```Python\n' +
                '    @registry.ROI_PARSING_HEADS.register("roi_gce_head")\n' +
                '    class roi_gce_head(nn.Module):\n' +
                '        def __init__(self, dim_in, spatial_scale):\n' +
                '            super(roi_gce_head, self).__init__()\n' +
                '\n' +
                '            resolution = cfg.PRCNN.ROI_XFORM_RESOLUTION\n' +
                '            sampling_ratio = cfg.PRCNN.ROI_XFORM_SAMPLING_RATIO\n' +
                '            pooler = Pooler(\n' +
                '                output_size=(resolution, resolution),\n' +
                '                scales=spatial_scale,\n' +
                '                sampling_ratio=sampling_ratio,\n' +
                '            )\n' +
                '            self.pooler = pooler\n' +
                '            self.dim_in = dim_in\n' +
                '\n' +
                '            use_nl = cfg.PRCNN.GCE_HEAD.USE_NL\n' +
                '            use_bn = cfg.PRCNN.GCE_HEAD.USE_BN\n' +
                '            use_gn = cfg.PRCNN.GCE_HEAD.USE_GN\n' +
                '            conv_dim = cfg.PRCNN.GCE_HEAD.CONV_DIM\n' +
                '            asppv3_dim = cfg.PRCNN.GCE_HEAD.ASPPV3_DIM\n' +
                '            num_convs_before_asppv3 = cfg.PRCNN.GCE_HEAD.NUM_CONVS_BEFORE_ASPPV3\n' +
                '            asppv3_dilation = cfg.PRCNN.GCE_HEAD.ASPPV3_DILATION\n' +
                '            num_convs_after_asppv3 = cfg.PRCNN.GCE_HEAD.NUM_CONVS_AFTER_ASPPV3\n' +
                '\n' +
                '            # convx before asppv3 module\n' +
                '            before_asppv3_list = []\n' +
                '            for _ in range(num_convs_before_asppv3):\n' +
                '                before_asppv3_list.append(\n' +
                '                    make_conv(self.dim_in, conv_dim, kernel=3, stride=1, use_bn=use_bn, use_gn=use_gn, use_relu=True)\n' +
                '                )\n' +
                '                self.dim_in = conv_dim\n' +
                '            self.conv_before_asppv3 = nn.Sequential(*before_asppv3_list) if len(before_asppv3_list) else None\n' +
                '\n' +
                '            # asppv3 module\n' +
                '            self.asppv3 = []\n' +
                '            self.asppv3.append(\n' +
                '                make_conv(self.dim_in, asppv3_dim, kernel=1, use_bn=use_bn, use_gn=use_gn, use_relu=True)\n' +
                '            )\n' +
                '            for dilation in asppv3_dilation:\n' +
                '                self.asppv3.append(\n' +
                '                    make_conv(self.dim_in, asppv3_dim, kernel=3, dilation=dilation, use_bn=use_bn, use_gn=use_gn,\n' +
                '                              use_relu=True)\n' +
                '                )\n' +
                '            self.asppv3 = nn.ModuleList(self.asppv3)\n' +
                '            self.im_pool = nn.Sequential(\n' +
                '                nn.AdaptiveAvgPool2d(1),\n' +
                '                make_conv(self.dim_in, asppv3_dim, kernel=1, use_bn=use_bn, use_gn=use_gn, use_relu=True)\n' +
                '            )\n' +
                '            self.dim_in = (len(asppv3_dilation) + 2) * asppv3_dim\n' +
                '\n' +
                '            feat_list = []\n' +
                '            feat_list.append(\n' +
                '                make_conv(self.dim_in, conv_dim, kernel=1, use_bn=use_bn, use_gn=use_gn, use_relu=True)\n' +
                '            )\n' +
                '            if use_nl:\n' +
                '                feat_list.append(\n' +
                '                    NonLocal2d(conv_dim, int(conv_dim * cfg.PRCNN.GCE_HEAD.NL_RATIO), conv_dim, use_gn=True)\n' +
                '                )\n' +
                '            self.feat = nn.Sequential(*feat_list)\n' +
                '            self.dim_in = conv_dim\n' +
                '\n' +
                '            # convx after asppv3 module\n' +
                '            assert num_convs_after_asppv3 >= 1\n' +
                '            after_asppv3_list = []\n' +
                '            for _ in range(num_convs_after_asppv3):\n' +
                '                after_asppv3_list.append(\n' +
                '                    make_conv(self.dim_in, conv_dim, kernel=3, use_bn=use_bn, use_gn=use_gn, use_relu=True)\n' +
                '                )\n' +
                '                self.dim_in = conv_dim\n' +
                '            self.conv_after_asppv3 = nn.Sequential(*after_asppv3_list) if len(after_asppv3_list) else None\n' +
                '            self.dim_out = self.dim_in\n' +
                '\n' +
                '        def forward(self, x, proposals):\n' +
                '            resolution = cfg.PRCNN.ROI_XFORM_RESOLUTION\n' +
                '            x = self.pooler(x, proposals)\n' +
                '\n' +
                '            if self.conv_before_asppv3 is not None:\n' +
                '                x = self.conv_before_asppv3(x)\n' +
                '\n' +
                '            asppv3_out = [interpolate(self.im_pool(x), scale_factor=resolution, \n' +
                '                                      mode="bilinear", align_corners=False)]\n' +
                '            for i in range(len(self.asppv3)):\n' +
                '                asppv3_out.append(self.asppv3[i](x))\n' +
                '            asppv3_out = torch.cat(asppv3_out, 1)\n' +
                '            asppv3_out = self.feat(asppv3_out)\n' +
                '\n' +
                '            if self.conv_after_asppv3 is not None:\n' +
                '                x = self.conv_after_asppv3(asppv3_out)\n' +
                '            return x\n' +
                '```\n'},
        {ul:'The function of GCE module is to enrich the receptive field of sub network\'s feature map and encode geometric and context information between human parts.'},
        {
            block:{
                title:'GCE module',
                children:[
                    {text:'The relationship between human parts is very important information in the task of human part analysis. Simply using the conventional convolution stacking method to build ParsingRCNN sub network can not capture the relationship between the left/right hands, left/right foots and the limbs of different people. On one hand, On the one hand, it is due to the deficiency of the receptive field, and on the other hand, it is due to the fact that ordinary convolution generally extracts the semantic imformation of objects without paying more attention to the relationship of human parts. GCE module utilize  non-local structure from [Non-local Neural Networks](https://arxiv.org/pdf/1711.07971.pdf)\\[3\\] to enhance the capture of this geometric and contextual relationship between human parts.\n'},
                    {img:'GCE'},
                    {text:'GCE module', type:'center'},
                    {text:'According to the ablation experiments of non-local and ASPP structures in GCE module, the non-local module can only play its role if the feature map has rich receptive fields, using the non-local module alone can not be helpful to human body parts analysis. ASPP module is used to enrich the receptive field of feature maps, and the geometric and contextual relationship between human parts can be better captured by using ASPP alone to increase the receptive field. Refer to [Deeplab-v3](https://arxiv.org/pdf/1706.05587)\\[4\\] for more information about the ASPP structure.\n'},
                    {
                        table:{
                            titles:['-','mIoU', 'AP@50',  'AP@vol', 'PCP@50'],
                            data:[
                                ['baseline',50.7 , 47.9 , 47.6 , 49.7],
                                ['ASPP only', 51.9 , 51.1 , 48.3 , 51.4],
                                ['Non-local only' , 50.5 , 47.0 , 47.6 , 48.9],
                                ['GCE' , 52.7 , 53.2 , 49.7 , 52.6]
                            ]
                        }
                    },
                    {text:'PS：Baseline network comes from [Mask R-CNN](https://arxiv.org/abs/1703.06870)\\[5\\]\n'},
                    {img:'gce_effect'}
                ]
            }
        },
        {ul:'The feature transformation module is used to transform the features captured from the backbone feature map into task features, which can be implemented by several stacked convolution layers;'},
        {
            block:{
                title:'Parsing branch decoupling',
                children:[
                    {text:'In the anchor-based instance analysis method, each task branch such as Mask R-CNN, Keypoint R-CNN, Parsing R-CNN, can be understood as an independent neural network for its task. However, most of the work has not carried out detailed research and design on the structure of the task branch. Based on the GCE module, Parsing R-CNN decoupled the main structure of the parsing branch (PBD). The structure of the parsing branch was decoupled into **pre-GCE structure**,**GCE structure** and **post-GCE structure**, and their effects were analyzed experimentally.\n'},
                    {
                        table:{
                            titles:['-','mIoU', 'AP@50',  'AP@vol', 'PCP@50'],
                            data:[
                                ['baseline',52.7 , 53.2 , 49.7 , 52.6],
                                ['4conv + GCE' , 52.8 , 54.9 , 50.5 , 54.2 ],
                                ['GCE + 4conv (PBD)' , 53.5 , 58.5 , 51.7 , 56.5],
                                ['4conv + GCE + 4conv' , 53.1 , 58.8 , 51.6 , 56.7]
                            ]
                        }
                    },
                    {text:'PS：Baseline network comes from [Mask R-CNN](https://arxiv.org/abs/1703.06870)，the convolution kernel is 3 of all ablation experiments.\n'}
                ]
            }
        },
        {ul:'Parsing R-CNN also utilizes **Proposals Separation Sampling**(PSS) and **Enlarging RoI Resolution**(ERR) to improve the resolution of the RoI feature map to improve the performance of Parsing R-CNN.'},
        {
            block:{
                title:'PSS & ERR',
                children:[
                    {text:'In object detection tasks, FPN is used to extract RoI from feature maps of different resolutions. However, parsing task is similar to image segmentation which requires a larger resolution RoI feature map to retain detailed information of human body parts, the feature maps with too small resolution do not have such information.\n'},
                    {text:'PSS, that is, the parsing branch perform RoIAlign on the feature map with 4 times downsample ratio in FPN output feature maps to ensure that the RoI feature maps have sufficient detail information.\n'},
                    {
                        table:{
                            titles:['-','boxAP','mIoU', 'AP@50',  'AP@vol', 'PCP@50'],
                            data:[
                                ['baseline',67.7 , 47.2 , 41.4 , 45.4 , 44.3],
                                ['P2 only' , 66.4 , 47.7 , 42.6 , 45.8 , 45.1 ],
                                ['PSS' , 67.5 , 48.2 , 42.9 , 46.0 , 45.5 ],
                            ]
                        }
                    },
                    {text:'ERR enlarges RoI feature map of 14x14 pixels output by RoIAlign to 32x32 pixels, which further enriches the detail information needed for segmentation. However, with the increase of RoI feature map resolution, the computing speed of the network decreases. Therefore, it is very important to balance the accuracy and speed in algorithm research and practical application.\n'},
                    {
                        table:{
                            titles:['-','fps','mIoU', 'AP@50',  'AP@vol', 'PCP@50'],
                            data:[
                                ['baseline (14×14)' , 10.4 , 48.2 , 42.9 , 46.0 , 45.5],
                                ['ERR (32×32)' , 9.1 , 50.7 , 47.9 , 47.6 , 49.7 ],
                                ['ERR (32×32)，100 RoIs' , 11.5 , 50.5 , 47.5 , 47.3 , 49.0 ],
                                ['ERR (64×64)' , 5.6 , 51.5 , 49.0 , 47.9 , 50.8 ]
                            ]
                        }
                    },
                ]
            }
        },
        {part_title:'Training'},
        {text:'After completing data loading and model building, we need to choose the optimization strategy of training Parsing R-CNN model before training, follow the settings of Mask R-CNN, set the base learning rate of 0.02, train 45000 iterations with the batch size of 16, and use the learning rate warming up and step declining strategy in combination. At 30000 and 40000 iterations, the learning rate is reduced ten times.\n' +
                '\n' +
                '```Python\n' +
                '    # Train\n' +
                '    logging_rank(\'Training starts !\', distributed=args.distributed, local_rank=args.local_rank)\n' +
                '    train(model, train_loader, optimizer, scheduler, checkpointer, logger)\n' +
                '    logging_rank(\'Training done !\', distributed=args.distributed, local_rank=args.local_rank)\n' +
                '```\n' +
                '\n' +
                'During the training process, the log system records the training status such as the number of current iterations and the losses for every iteration. The `Checkpointer` periodically saves the model parameters to the path set by `cfg.CKPT` in the config system.\n'},
        {text:'According to the log interval set by `cfg.DISPLAY_ITER`, the log system records the training status of the model in the terminal every 20 iterations during the training process.\n'},
        {shell:'```\n' +
                '[Training][e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml][iter: 200/45000][lr: 0.009200][eta: 21:23:30]\n' +
                '\t  total_loss: 1.690106 (1.417845), iter_time: 1.8643 (1.7190), data_time: 0.1549 (0.1443)\n' +
                '\t  loss_parsing: 0.395894 (0.365891), loss_objectness: 0.252050 (0.210352), loss_classifier: 0.161344 (0.199260), loss_box_reg: 0.228464 (0.202087), loss_rpn_box_reg: 0.431002 (0.427271)\n' +
                '[Training][e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml][iter: 220/45000][lr: 0.009920][eta: 21:29:40]\n' +
                '\t  total_loss: 1.188639 (1.316550), iter_time: 2.0313 (1.7280), data_time: 0.1353 (0.1444)\n' +
                '\t  loss_parsing: 0.395576 (0.342062), loss_objectness: 0.205645 (0.191415), loss_classifier: 0.199962 (0.190168), loss_box_reg: 0.156144 (0.187377), loss_rpn_box_reg: 0.411209 (0.438963)\n' +
                '[Training][e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml][iter: 240/45000][lr: 0.010640][eta: 21:37:11]\n' +
                '\t  total_loss: 1.737057 (1.387051), iter_time: 1.8072 (1.7389), data_time: 0.1581 (0.1447)\n' +
                '\t  loss_parsing: 0.347431 (0.351932), loss_objectness: 0.299453 (0.190103), loss_classifier: 0.196695 (0.190588), loss_box_reg: 0.149391 (0.185793), loss_rpn_box_reg: 0.479773 (0.427392)\n' +
                '```\n'},
        {part_title:'Testing'},
        {text:'After training Parsing R-CNN model, we used [$Pet/tools/rcnn/test_net.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/tools/rcnn/test_net.py) to evaluate the precision on CIHP_val. It is also necessary to use `Dataloader` to load the test dataset, scale the short side of the image to 800 pixels and scale the long side with the same factor(limiting under 1333 pixels).\n'},
        {text:'By loading the model `$Pet/ckpts/rcnn/mscoco/e2e_mask_rcnn_R-50-FPN-600_0.5x/model_latest.pth` after the maximum number of  training iterations, the following command are executed to test the Mask R-CNN model, and the test log of Mask R-CNN will also be recorded by `Logger`.\n'},
        {shell:'```\n' +
                '    cd $Pet\n' +
                '\n' +
                '    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/rcnn/test_net.py\n' +
                '    --cfg cfgs/rcnn/CIHP/e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml\n' +
                '```\n'},
        {
            block:{
                title:'Metrics for multi-human parsing',
                children:[
                    {text:'In human parsing task, model precision is evaluated via mIOU、AP<sup>p</sup>（AP@50）、AP<sup>p</sup><sub>vol</sub>（AP@vol) and PCP from [MHP-v1](https://arxiv.org/pdf/1705.07206)\\[6\\].\n'},
                    {text:'mean Intersection Over Union(mIOU），averaged pixel IOU of all human parts between the predicted mask and ground-truth mask.\n'},
                    {text:'Average Precision based on Part(AP<sub>p</sub>)，different from the average precision of the whole region. AP<sub>p</sub> uses the mIOU between the prediction mask and the ground-truth mask in different parts of a human body instance to judge whether a prediction instance is correct. AP@50 represents AP with the threshold of mIOU is 0.5, AP@vol represents averaged AP<sub>p</sub> with mIOU threshold range from 0.1 to 0.9 (gap is 0.1).'},
                    {text:'Percentage of Correctly Parsed Body Parts PCP)，AP<sub>p</sub> averages the Precision of all human body parts， thus it does not really reflect how many parts of the human body have been correctly predicted. Therefore, in each human instance, each human part whose IOU is higher than a certain threshold are considered to be correctly predicted. Each human instance calculates a PCP, and the overall CPP is the average of all the human instances. PCP@50 represents the PCP with IOU threshold of 0.5.\n'},

                ]
            }
        },
        {part_title:'Visualization of Reasoning Results'},
        {text:'In Pet, Parsing R-CNN returns the confidence scores, bounding box coordinates and parsing masks for each human instance。 Visualization of the inference result of a picture in CIHP_val as follows.\n'},
        {img:'demo_parsing_0000004'},
        {part_title:'Reference'},
        {text:'\\[1\\] Lu Yang, Qing Song, Zhihui Wang and Ming Jiang. Parsing R-CNN for Instance-Level Human Analysis. CVPR 2019.\n' +
                '\n' +
                '\\[2\\] K. Gong, X. Liang, Y. Li, Y. Chen, and L. Lin. Instance-level human parsing via part grouping network. ECCV, 2018.\n' +
                '\n' +
                '\\[3\\] Xiaolong Wang, Ross Girshick, Abhinav Gupta1, and Kaiming He. Non-localneural networks. In CVPR, 2018.\n' +
                '\n' +
                '\\[4\\] L. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam. Encoder-decoder with atrous separable convolution for se- mantic image segmentation. In ECCV, 2018.\n' +
                '\n' +
                '\\[5\\] Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam. Rethinking Atrous Convolution for Semantic Image Segmentation. arXiv:1706.05587, 2017.\n' +
                '\n' +
                '\\[6\\] Jianshu Li, Jian Zhao, Yunchao Wei, Congyan Lang, Yidong Li, Terence Sim, Shuicheng Yan, Jiashi Feng. Multi-human parsing in the wild. arXiv preprint arXiv:1705.07206, 2017.\n'},

    ]
}