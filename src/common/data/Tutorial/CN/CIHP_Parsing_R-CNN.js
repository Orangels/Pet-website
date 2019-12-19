export let CIHP_Parsing_R_CNN = {
    key: 'CIHP_Parsing_R_CNN',
    dataSource: [
        {title:'在CIHP数据集上训练Parsing R-CNN模型'},
        {text:'本教程将介绍使用Pet训练以及测试Parsing R-CNN模型进行人体部位分割的主要步骤，在此我们会指导您如何通过组合Pet的提供的各个功能模块来构建Parsing R-CNN模型，在此我们仅讲解组件的调用，部分实现细节请查阅系统组件的相应部分。在阅读本教程的之前我们强烈建议您阅读原始论文[Parsing R-CNN](https://arxiv.org/abs/1811.12596)\\[1\\]、[PGN](https://arxiv.org/pdf/1808.00157v1)\\[2\\]以了解更多关于Parsing R-CNN的算法原理。\n'},
        {
            note:[
                {text:'首先参阅[CIHP数据准备](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87.md#cihp%E6%95%B0%E6%8D%AE%E9%9B%86)教程并在硬盘上准备好CIHP数据集。\n'}
            ]
        },
        {text:'如果您具有丰富的人体部位分析算法的研究经验，您也可以直接在Pet中运行`$Pet/tools/rcnn/train_net.py`脚本利己开始训练您的Parsing R-CNN模型.\n'},
        {text:'用法示例：\n'},
        {ul:'在8块GPU上使用`CIHP_train`训练一个端到端的Parsing R-CNN模型：'},
        {shell:'```\n' +
                '    cd $Pet\n' +
                '\n' +
                '    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 tools/rcnn/train_net.py --cfg cfgs/rcnn/CIHP/e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml\n' +
                '```\n'},
        {ul:'在8块GPU上使用`CIHP_val`数据集上测试训练的Parsing R-CNN模型：'},
        {shell:'```\n' +
                '    cd $Pet\n' +
                '\n' +
                '    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/rcnn/test_net.py --cfg cfgs/rcnn/CIHP/e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml\n' +
                '```\n'},
        {text:'在进行任何与模型训练和测试有关的操作之前，需要先选择一个指定的`yaml`文件，明确在训练时候对数据集、模型结构、优化策略以及其他重要参数的需求与设置，本教程以`$Pet/cfgs/rcnn/CIHP/e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml`为例，讲解训练过程中所需要的关键配置，该套配置将指导此Parsing R-CNN模型训练以及测试的全部步骤和细节，全部参数设置请见[配置文件](https://github.com/BUPT-PRIV/Pet-dev/blob/master/cfgs/rcnn/CIHP/e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml)。\n'},
        {part_title:'数据载入'},
        {text:'确保CIHP数据集已经存放在您的硬盘中，并按照[数据制备](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%9E%B6%E6%9E%84/component-collects/%E6%95%B0%E6%8D%AE%E5%88%B6%E5%A4%87/%E6%95%B0%E6%8D%AE%E5%88%B6%E5%A4%87.md)中的文件结构整理好CIHP数据集的文件结构，接下来开始加载`CIHP_train`训练集。\n' +
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
                '目前训练基于区域两阶段实例分析模型都需要将图像的长和宽按照等比例缩放，Parsing R-CNN中将图像短边缩放到一系列的基本尺寸来丰富CIHP数据中的人体样本实例的尺度，在配置文件中将`TRAIN.SCALES`设置为`(512, 640, 704, 768, 800, 864)`，同时将`TRAIN.MAX_SIZE`设置为`1400`限制图像的最大尺寸。',
                'Parsing R-CNN还对训练数据进行了随机水平翻转来进行数据增广，提升模型的泛化性，经过变换的图像以及其标注的可视化结果如下图：'
            ]
        },
        {img:'Parsing'},
        {text:'Dataloader中的图片、检测框与分割掩模的transform前后的可视化', type:'center'},
        {ul:'数据载入组件不只是完成了图像数据以及标注信息的读取，还在采集每个批次的数据的同时生成了RPN网络的训练标签，数据载入组件输出的每个批次的数据中包含图片数据、图片中物体的类别，物体的包围框、以及与物体数量相同的人体部位分割掩模（每个掩模只包含一个目标的前景掩码）。'},
        {yaml:'```\n' +
                '    data: (1, 3, 800, 1196)\n' +
                '    label: (1, 6)\n' +
                '    box: (1, 6, 4)\n' +
                '    mask: (1, 6, 800, 1196)\n' +
                '```\n'},
        {part_title: 'Parsing R-CNN网络结构'},
        {text:'在Pet中，Parsing R-CNN网络使用`Generalized_RCNN`模型构建器来搭建。只需要在YAML文件中添加\'PRCNN\'参数来构建Parsing R-CNN网络的人体部位分析的分支网络：\n' +
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
        {text:'在`Generalized_RCNN`中只需要在`Fast_RCNN`之后添加`Parsing_RCNN`分支网络的构建代码即可，其他部分与[Mask R-CNN](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E5%9C%A8MSCOCO2017%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E8%AE%AD%E7%BB%83Mask%20R-CNN%E6%A8%A1%E5%9E%8B.md#mask-r-cnn%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84)网络的构建方式类似。\n' +
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
                'Parsing R-CNN网络在除了包含区域建议网络（RPN)、特征金字塔网络（FPN）以及目标检测分支网络（FastRCNN）之外，最重要的分支网络就是**人体部位分析分支网络**（ParsingRCNN)。`ParsingRCNN`网络的主体结构是`roi_gce_head`，`roi_gce_head`主要由**几何与周边信息编码模块**（Geometric and Context Encoding， GCE）与**特征转换模块**组成。\n' +
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
        {ul:'GCE模块的作用是丰富分支网络特征图的感受野并编码人体部位之间的几何与周边信息'},
        {
            block:{
                title:'GCE模块',
                children:[
                    {text:'人体部位相互之间的关系在人体部位分析任务中是十分重要的信息，简单地使用常规的卷积层堆叠的方式来构建ParsingRCNN分支网络并不能很好地捕获如左、右手，左、右脚以及不同人物的肢体之间的关系，一方面是由于感受野的不足，另一方面是因为普通卷积普遍地提取目标的语义特征而没有将更多的注意力放在人体部位的关系捕获上。GCE模块使用[Non-local Neural Networks](https://arxiv.org/pdf/1711.07971.pdf)\\[3\\]中提出的non-local结构来加强捕获这种属于人体部位之间的几何与上下文关系。\n'},
                    {img:'GCE'},
                    {text:'GCE模块', type:'center'},
                    {text:'根据Parsing R-CNN论文中也对GCE模块中Non-local与ASPP结构的退化实验，Non-local模块只有在特征图具有丰富感受野的情况下才能发挥作用，单独使用Non-local模块并不能对人体部位分析有所帮助。ASPP模块被用来丰富特征图的感受野，单独使用ASPP没开来增大感受野也可以更好地捕获人体部位之间的几何与上下文关系。关于ASPP结构的详细信息请参考[Deeplab-v3](https://arxiv.org/pdf/1706.05587)\\[4\\]。\n'},
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
                    {text:'注：基准实验采用[Mask R-CNN](https://arxiv.org/abs/1703.06870)\\[5\\]的网络结构\n'},
                    {img:'gce_effect'}
                ]
            }
        },
        {ul:'特征转换模块被用来将从主干网络特征图上截取出来的特征向任务特征进行转换，由几个堆叠的卷积层即可完成；'},
        {
            block:{
                title:'特征转换与分支网络解耦(PBD)',
                children:[
                    {text:'在基于区域的实例分析方法中，每个分支网络诸如Mask R-CNN、Keypoint R-CNN、Parsing R-CNN都可以被理解为一个针对所属任务的独立神经网络，然而大多数工作并未对分支网络的结构进行详细的研究与设计。基于人体部位分析分支的主体结构GCE模块，Parsing R-CNN进行了人体部位分析分支解耦（Parsing Branch Decoupling，PBD）将分支网络的结构分为**前GCE结构**、**GCE结构**和**后GCE结构**，并对他们的作用进行了实验分析。\n'},
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
                    {text:'注：基准实验采用[Mask R-CNN](https://arxiv.org/abs/1703.06870)的网络结构，所有退化实验的卷积均使用大小为3的卷积核。\n'}
                ]
            }
        },
        {ul:'Parsing R-CNN还使用了区域分离采样（Proposals Separation Sampling，PSS）与扩大区域特征图分辨率（Enlarging RoI Resolution，ERR）的方法来提升建议区域特征图的分辨率，为Parsing R-CNN提升性能。'},
        {
            block:{
                title:'PSS与ERR',
                children:[
                    {text:'在目标检测任务中，FPN被用来从不同分辨率的特征图上提取RoI，但人体部位分析任务类似于图像分割，需要较大分辨率的RoI的特征图来保留人体部位比较详细的细节信息，分辨率过小的特征图不具备这些信息。\n'},
                    {text:'PSS即人体部位分析分支只在FPN中4倍下采样的特征图上对RoI进行RoIAlign，保证RoI特征图具有较充分的细节信息。\n'},
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
                    {text:'ERR将RoIAlign输出的14x14像素的RoI特征图扩大到32x32像素，进一步丰富分割所需的细节信息，但随着RoI特征图分辨率的增大，网络的计算速度随之下降，由此可见做好精度与速度的平衡在算法研究与实际应用中都很重要。\n'},
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
        {part_title:'训练'},
        {text:'完成了数据载入以及模型构建之后，我们需要在开始训练之前选择模型的优化策略，遵循Mask R-CNN的设置，在批次大小为16的情况下，设置初始学习率为0.02，训练45000次迭代，组合使用了学习率预热与阶段下降策略，分别在30000与40000次迭代时将学习率减小十倍。\n' +
                '\n' +
                '```Python\n' +
                '    # Train\n' +
                '    logging_rank(\'Training starts !\', distributed=args.distributed, local_rank=args.local_rank)\n' +
                '    train(model, train_loader, optimizer, scheduler, checkpointer, logger)\n' +
                '    logging_rank(\'Training done !\', distributed=args.distributed, local_rank=args.local_rank)\n' +
                '```\n' +
                '\n' +
                '在训练过程中，日志记录仪会在每若干次迭代后记录当前网络训练的迭代数、各项偏差数值等训练信息，检查点组件会定期保存网络模型到配置系统中`cfg.CKPT`所设置的路径下。\n'},
        {text:'根据`cfg.DISPLAY_ITER`设置的日志记录间隔，在训练过程中每经过20次迭代，日志记录仪会在终端中记录模型的训练状态。\n'},
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
        {part_title:'测试'},
        {text:'在完成Parsing R-CNN模型的训练之后，我们使用[$Pet/tools/rcnn/test_net.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/tools/rcnn/test_net.py)在CIHP_val上评估模型的精度。同样需需要使用`Dataloader`来加载测试数据集，将图像的短边缩放到800像素，长边做同样尺度的缩放（长边最大1333像素）。\n'},
        {text:'通过加载训练最大迭代数之后的模型`$Pet/ckpts/rcnn/CIHP/e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms/model_latest.pth`，执行下面的命令进行Parsing R-CNN模型的测试，测试日志同样会被`Logger`所记录。\n'},
        {shell:'```\n' +
                '    cd $Pet\n' +
                '\n' +
                '    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/rcnn/test_net.py --cfg cfgs/rcnn/CIHP/e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml\n' +
                '```\n'},
        {
            block:{
                title:'人体部位分析任务的评价指标',
                children:[
                    {text:'在人体部位分析任务中，模型精度使用[MHP-v1](https://arxiv.org/pdf/1705.07206)\\[6\\]中提出的mIOU、AP<sup>p</sup>（AP@50）、AP<sup>p</sup>,<sub>vol</sub>（AP@vol)与PCP来评价模型精度。\n'},
                    {text:'平均交并比（mean Intersection Over Union，mIOU），在一张图片中所有人体部位类别的预测掩码与标准掩码的像素交并比的平均值。\n'},
                    {text:'基于人体部位的平均准确率(Average Precision based on Part，AP<sup>p</sup>)，与实例分割任务基于整个区域的平均准确率不同，AP使用的是一个人体实例内不同部位的预测掩码与标注掩码之间的mIOU来判断一个预测的实例是否正确，AP@50代表当mIOU的阈值为0.5，AP@vol代表mIOU的阈值分别为0.1~0.9（间隔为0.1）时的平均值。\n'},
                    {text:'正确分析的人体部位百分比（Percentage of Correctly Parsed Body Parts，PCP)，AP<sup>p</sup>将所有人体部位的准确率进行了均值计算，不能真正反映有多少人体部位被正确的预测。因此在每一个人物实例中，所有与标注掩码的像素IOU高于一定阈值的人体部位被认为是被正确预测，每一个人物实例计算一个PCP，整体的PCP是有所有人物实例的PCP的平均值。PCP@50代表IOU的阈值为0.5。\n'},

                ]
            }
        },
        {part_title:'推理结果可视化'},
        {text:'在Pet中Parsing R-CNN返回每一个人物实例的置信度分数，边界框坐标和分割掩码。将CIHP_val中的一张图片的推理结果进行可视化如下图。\n'},
        {img:'demo_parsing_0000004'},
        {part_title:'参考文献'},
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