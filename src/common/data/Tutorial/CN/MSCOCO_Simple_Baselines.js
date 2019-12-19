export let MSCOCO_Simple_Baselines = {
    key: 'MSCOCO_Simple_Baselines',
    dataSource: [
        {title:'在MSCOCO2017数据集上训练Simple Baselines模型'},
        {text:'本教程将介绍使用Pet在MSCOCO2017人体关键点子集上训练以及测试Simple Baselines模型用于单人体姿态估计的主要步骤，在此我们会指导您如何通过组合Pet的提供的各个功能模块来构建Simple Baselines模型，在此我们仅讲解组件的调用，部分实现细节请查阅系统组件的相应部分。在阅读本教程的之前我们强烈建议您阅读原始论文[Simple Baselines](https://arxiv.org/abs/1804.06208v2)\\[1\\]以了解更多关于Simple Baselines的算法原理。\n'},
        {
            note:[
                {text:'虽然Pet下pose任务仅实现了在MSCOCO2017的关键点子数据集上单人体关键点检测，但为了数据源文件的一致性，pose任务对数据源文件的格式不做要求，使用与rcnn任务相同的数据格式，点击[此处](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87.md#mscoco%E6%95%B0%E6%8D%AE%E9%9B%86)了解MSCOCO关键点数据集的准备。\n'}
            ]
        },
        {text:'如果您具有丰富的目标检测算法的研究经验，您也可以直接在Pet中运行 [$Pet/tools/pose/train_net.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/tools/pose/train_net.py) 脚本利己开始训练您的Simple Baselines模型。\n'},
        {text:'用法示例：\n'},
        {ul:'在4块GPU上使用`keypoints_coco_2017_train`训练一个端到端的Simple Baselines模型，使用两个全连接层作为`RCNN`的功能网络：\n'},
        {shell:'```\n' +
                '  cd $Pet\n' +
                '\n' +
                '  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch tools/pose/train_net.py --cfg cfgs/pose/mscoco/simple_R-50c-D3K4C256_256x192_adam_1x.yaml\n' +
                '```\n'},
        {ul:'在4块GPU上使用`keypoints_coco_2017_val`数据集上测试训练的Simple Baselines模型：'},
        {shell:'```\n' +
                '  cd $Pet\n' +
                '\n' +
                '  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/pose/test_net.py --cfg cfgs/pose/mscoco/simple_R-50c-D3K4C256_256x192_adam_1x.yaml\n' +
                '```\n'},
        {text:'本教程使用[$Pet/cfgs/pose/mscoco/simple_R-50c-D3K4C256_256x192_adam_1x.yaml](https://github.com/BUPT-PRIV/Pet-dev/blob/master/cfgs/pose/mscoco/simple_R-50c-D3K4C256_256x192_adam_1x.yaml)中的设置来明确在训练时候对数据集、模型结构、优化策略的配置，该套配置将指导此Simple Baselines模型训练以及测试的全部步骤和细节。\n'},
        {part_title:'数据载入'},
        {text:'确保MSCOCO2017关键点子集的源文件与关键点标注文件已经按照标准存放在您的硬盘中，接下来我们可以开始加载`keypoints_coco_2017_train`训练集。\n' +
                '\n' +
                '```Python\n' +
                '  # Create training dataset and loader\n' +
                '  train_set = build_dataset(cfg.TRAIN.DATASETS, is_train=True, local_rank=args.local_rank)\n' +
                '  train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if args.distributed else None\n' +
                '  ims_per_gpu = int(cfg.TRAIN.BATCH_SIZE / args.world_size)\n' +
                '  train_loader = make_train_data_loader(train_set, ims_per_gpu, train_sampler)\n' +
                '  cfg.TRAIN.ITER_PER_EPOCH = len(train_loader)\n' +
                '```\n'},
        {ul:[
            '在单人体姿态估计任务中，通过[COCOInstanceDataset]()，中的MSCOCO_2017数据集中的每一个人物实例被从包含多个人物的图片中截取出来，`train_loader`输出由多个人物实例的图片组成的一个批次，训练时每一个人物示例图像被缩放到192像素x256像素。设置`cfg.POSE.HEATMAP_SIZE`为48像素x64像素，尺寸是输入图像的四分之一，在载入数据是用于生成关键点的训练标签。',
                'Simple Baselines在数据加载时还对每个人物实例进行了颜色抖动以及旋转来进行数据增广，提升模型的泛化性，数据集中原始图像与`train_loader`输出的图像如下图：'
            ]},
        {img:'pose_ori'},
        {img:'pose_ins_gaussian'},
        {text:'Dataloader中的图片、人体关键点的transform前后的可视化',type:'center'},
        {
            block:{
                title:'姿态估计任务的训练标签',
                children:[
                    {text:'语义分割任务需要对图像中所有像素的类别进行预测，在训练语义分割模型时与图像一起载入的训练标签是与图像同尺寸的分割掩模。姿态估计任务与语义分割任务相似，模型需要预测出图像中哪些像素点是关键点，因此在训练时同样需要一个承载关键点的掩模图像作为训练标签。\n'},
                    {text:'如果仅将一副图像中的有限数量关键点赋值到掩模标签上，那么在训练过程中将会导致损失函数居高不下，因此在姿态估计任务的训练标签时，以每一个人体关键点为中心，在其周围生成一个高斯分布，每个关键点的高斯分布，作为一个标签热力图，不可见的关键点的高斯热力图的值全部为零。\n'}
                ]
            }
        },
        {text:'输出：\n'},
        {shell:'```\n' +
                '  images：(1, (3, 256, 192))\n' +
                '  target：(1, (17, 64, 48))  # training heatmaps of keypoints\n' +
                '  target_weight：(1, 17, 1))\n' +
                '```\n'},
        {part_title:'Simple Baselines网络结构'},
        {text:'在Pet中，Simple Baselines网络使用`Generalized_CNN`模型构建器来搭建，详情见[模型构建](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%9E%B6%E6%9E%84/component-collects/%E6%A8%A1%E5%9E%8B%E6%9E%84%E5%BB%BA/%E6%A8%A1%E5%9E%8B%E6%9E%84%E5%BB%BA.md)。我们在`yaml`文件中设置如下参数来使用配置系统构建Simple Baselines网络中特征提取网络的构建：\n' },
        {yaml:'```\n' +
                '  MODEL:\n' +
                '    TYPE: \'generalized_cnn\'\n' +
                '    POSE_ON: True\n' +
                '  BACKBONE:\n' +
                '    CONV_BODY: "resnet"\n' +
                '    RESNET:\n' +
                '      LAYERS: (3, 4, 6, 3)\n' +
                '      STRIDE_3X3: True\n' +
                '      USE_3x3x3HEAD: True\n' +
                '```\n'},

        {text:'根据`cfg.BACKBONE.RESNET.LAYERS`的设置来选取ResNet50作为特征提取网络，并且设置`cfg.BACKBONE.RESNET.USE_3x3x3HEAD`为True标志ResNet50使用三组“卷积+批次归一化”堆叠的形式作为第一阶段的网络结构。除了ResNet之外，Pet还提供了如[MobileNet-v1](https://arxiv.org/abs/1704.04861v1)\\[2\\]、[EfficientNet](https://arxiv.org/abs/1905.11946v2)\\[3\\]、[HRNet](https://arxiv.org/abs/1902.09212)\\[4\\]等特征提取网络结构。\n' +
                '\n' +
                '在构建完特征提取网络之后，通过如下参数进行功能网络、任务输出以及损失函数等网络模块的构建：\n'},

        {yaml:'```\n' +
                'KEYPOINT:\n' +
                '  NUM_JOINTS: 17\n' +
                '  HEATMAP_SIZE: (48, 64)\n' +
                '```\n'},
        {ul:[
            '`cfg.KEYPOINT`字典中的`POSE_HEAD`指定了功能网络，在[$Pet/blob/master/pet/instance/core/config.py](https://github.com/BUPT-PRIV/Pet/blob/master/pet/instance/core/config.py)中默认设置为`simple_xdeconv_head`, `simple_xdeconv_head`由3个“反卷积（ConvTranspose2d）+批次归一化（BatchNormlization）+线性激活函数（ReLU）”模块顺序组成。',
                '`NUM_JOINTS`用于确定任务输出的通道数，其数量取决于数据集的人体关键点数量，MSCOCO2017数据集人体关键点子集对每个人物实例标注了17个关键点。',
                '`HEATMAP_SIZE`确定最终用于关键点预测的特征图的尺寸。'
            ]},
        {
            block:{
                title:'自底向下与自顶向下的多人姿态估计',
                children:[
                    {text:'多人体姿态估计任务目前主流有两种实现方法，一种是自底向上（bottom-up），bottom-up方法是同时预测所有人物的关键点与关键点之间的连接关系，连接关系的形式可以是关键点之间的嵌入向量（embedding vector）或者是连接强度场（也是一种形式的热图）。具有代表性的工作如[Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1611.08050)[5]，对应的开源工程就是[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)。\n'},
                    {img:'bottom_up'},
                    {text:'OpenPose的算法流程是将图片中所有人体的不同关键点检测出来，同时预测出同一段肢体上任意两个关键点之间的关联热图，关联热图的数量等于人体相邻关键点对数，再通过对两个关键点对应的的关联热图的积分计算，来连接属于同一个人体的不同关键点。\n'},
                    {img:'apf'},
                    {text:'Simple Baselines多人体姿态估计任务中常常被用于自顶向下（top-down）的方法的第二阶段，其算法流程是第一阶段先使用检测器检测出图片中多个人体的包围框，然后再将人体的部分从原图中截取出来，然后使用关键点检测网络进行单人体的姿态估计。在单人体包围框存在的前提下，关键点检测问题主要着重于网络的设计、训练的策略等方面上，在Simple Baselines之前，[HourGlass](https://arxiv.org/abs/1603.06937)[6]和[CPN](https://arxiv.org/abs/1711.07319)[7]分别针对关键点检测问题的网络结构和关键点遮挡问题做出了一定的贡献。'},
                    {img:'two_stage_top_down'},
                    {text:'自顶向下的两阶段姿态估计算法\n'},
                    {text:'自顶向下方法还存在一种端到端的解决方案，即[Mask R-CNN](https://arxiv.org/abs/1703.06870)[8]，在检测分支预测包围框的同时，并行一个关键点检测分支，在包围框之内直接预测人体实例的关键点。与两阶段方法不同的是，关键点检测网络的输入是进入RPN网络的特征图，较低的分辨率和功能性欠强的特征图常常导致其关键点检测精度不如simple Baselines等两阶段的自顶向下的方法。\n'},
                ]
            }
        },
        {part_title:'训练'},
        {text:'完成了数据载入以及模型构建之后，我们需要在开始训练之前选择训练Simple Baselines模型的优化策略，遵循论文的思想，在批次大小为256的情况下，设置初始学习率为0.002，训练140次全数据迭代，组合使用了学习率预热与阶段下降策略，分别在90与120次全数据迭代时将学习率减小十倍。\n' +
                '\n' +
                '```Python\n' +
                '  def train(model, loader, optimizer, scheduler, logger):\n' +
                '      # switch to train mode\n' +
                '      model.train()\n' +
                '\n' +
                '      # main loop\n' +
                '      logger.iter_tic()\n' +
                '      logger.data_tic()\n' +
                '      for i, (inputs, targets, target_weight, meta) in enumerate(loader):\n' +
                '          scheduler.step()  # adjust learning rate\n' +
                '          optimizer.zero_grad()\n' +
                '\n' +
                '          inputs = inputs.to(args.device)\n' +
                '          targets = targets.to(args.device)\n' +
                '          target_weight = target_weight.to(args.device)\n' +
                '          logger.data_toc()\n' +
                '\n' +
                '          outputs = model(inputs, targets, target_weight)\n' +
                '          logger.update_stats(outputs, args.distributed, args.world_size)\n' +
                '          loss = outputs[\'total_loss\']\n' +
                '          if cfg.SOLVER.AMP.ENABLED:\n' +
                '              with amp.scale_loss(loss, optimizer) as scaled_loss:\n' +
                '                  scaled_loss.backward()\n' +
                '          else:\n' +
                '              loss.backward()\n' +
                '          optimizer.step()\n' +
                '\n' +
                '          if args.local_rank == 0:\n' +
                '              logger.log_stats(scheduler.iteration, scheduler.new_lr, skip_losses=True)\n' +
                '\n' +
                '          logger.iter_toc()\n' +
                '          logger.iter_tic()\n' +
                '          logger.data_tic()\n' +
                '      return None\n' +
                '\n' +
                '  # Train\n' +
                '  logging_rank(\'Training starts !\', distributed=args.distributed, local_rank=args.local_rank)\n' +
                '  train(model, train_loader, optimizer, scheduler, checkpointer, logger)\n' +
                '  logging_rank(\'Training done !\', distributed=args.distributed, local_rank=args.local_rank)\n' +
                '```\n'},
        {text:'在训练过程中，日志记录仪会在每若干次迭代后记录当前网络训练的迭代数、各项偏差数值等训练信息，检查点组件会定期保存网络模型到配置系统中`cfg.CKPT`所设置的路径下。'},
        {text:'根据`cfg.DISPLAY_ITER`设置的日志记录间隔，在训练过程中每经过20次迭代，日志记录仪会在终端中记录模型的训练状态。'},
        {shell:'```\n' +
                '\t[Training][simple_R-50c-D3K4C256_256x192_adam_1x.yaml][epoch: 1/140][iter: 180/585][lr: 0.002000][eta: 11:08:18]\n' +
                '\t\t  total_loss: 0.001064 (0.002683), iter_time: 0.4629 (0.4907), data_time: 0.0123 (0.0193)\n' +
                '\t[Training][simple_R-50c-D3K4C256_256x192_adam_1x.yaml][epoch: 1/140][iter: 200/585][lr: 0.002000][eta: 11:03:39]\n' +
                '\t\t  total_loss: 0.001151 (0.002533), iter_time: 0.4621 (0.4874), data_time: 0.0130 (0.0186)\n' +
                '\t[Training][simple_R-50c-D3K4C256_256x192_adam_1x.yaml][epoch: 1/140][iter: 220/585][lr: 0.002000][eta: 11:00:41]\n' +
                '\t\t  total_loss: 0.001182 (0.002413), iter_time: 0.4746 (0.4853), data_time: 0.0107 (0.0181)\n' +
                '\t[Training][simple_R-50c-D3K4C256_256x192_adam_1x.yaml][epoch: 1/140][iter: 240/585][lr: 0.002000][eta: 10:57:54]\n' +
                '\t\t  total_loss: 0.001118 (0.002306), iter_time: 0.4616 (0.4834), data_time: 0.0129 (0.0176)\n' +
                '\t[Training][simple_R-50c-D3K4C256_256x192_adam_1x.yaml][epoch: 1/140][iter: 260/585][lr: 0.002000][eta: 10:56:38]\n' +
                '\t\t  total_loss: 0.001101 (0.002219), iter_time: 0.4852 (0.4826), data_time: 0.0130 (0.0172)\n' +
                '```\n'},
        {part_title:'测试'},
        {text:'在完成Simple Baselines模型的训练之后，我们使用Pet/tools/pose/test_net.py在`keypoints_coco_2017_val`上评估模型的精度。同样需需要使用`Dataloader`来加载测试数据集，并对图像做同样尺度的缩放。\n'},
        {text:'通过加载训练最大迭代数之后的模型`$Pet/ckpts/pose/mscoco/simple_R-50c-D3K4C256_256x192_adam_1x/model_latest.pth`，执行下面的命令进行模型的测试，测试日志同样会被`Logger`所记录。\n'},
        {shell:'```\n' +
                '  cd $Pet\n' +
                '\n' +
                '  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/pose/test_net.py --cfg cfgs/pose/mscoco/simple_R-50c-D3K4C256_256x192_adam_1x.yaml\n' +
                '```\n'},

        {text:'测试输出：\n'},

        {shell:'```\n' +
                '\tINFO:pet.utils.misc:[Testing][range:1-794 of 6352][705/794][0.291s = 0.036s + 0.188s + 0.068s][eta: 0:00:25]\n' +
                '\tINFO:pet.utils.misc:[Testing][range:1-794 of 6352][737/794][0.282s = 0.034s + 0.180s + 0.068s][eta: 0:00:16]\n' +
                '\tINFO:pet.utils.misc:[Testing][range:1-794 of 6352][769/794][0.415s = 0.033s + 0.317s + 0.066s][eta: 0:00:10]\n' +
                '\tINFO:pet.utils.misc:Wrote instances to: /home/user/Downloads/Pet-dev/ckpts/pose/mscoco/simple_R-50c-D3K4C256_256x192_adam_1x/test/instances_range_0_794.pkl\n' +
                '\n' +
                '\t......\n' +
                '\n' +
                '\tAccumulating evaluation results...\n' +
                '\tDONE (t=0.08s).\n' +
                '\t Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.721\n' +
                '\t Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.915\n' +
                '\t Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.794\n' +
                '\t Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.691\n' +
                '\t Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.768\n' +
                '\t Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.753\n' +
                '\t Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.928\n' +
                '\t Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.819\n' +
                '\t Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.719\n' +
                '\t Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.805\n' +
                '\tINFO:pet.utils.misc:Evaluating keypoints is done!\n' +
                '```\n'},
        {
            block:{
                title:'姿态估计任务的评价指标',
                children:[
                    {text:'姿态估计任务模拟沿用了目标检测评价指标：平均准确率（Average Precision，AP）和平均召回率（Average Recall，AR）以及两者的其他变体。平均准确率与平均召回率的计算依靠**物体关键点相似度**（Object Keypoints Similarity，OKS）。\n'},
                    {text:'在目标检测任务中，物体检测框与标注框的交并比（IoU）被用来判断图片中的一个目标是否被算法检测到；在语义分割任务中，物体的分割掩模与标注掩模的交并比被用来判断分割模型的输出的分割掩模质量。\n'},
                    {text:'在姿态估计任务中，物体关键点相似度被用来评价一个人物实例被预测出的关键点与其标注关键点的相似度，当OKS大于一定阈值时，即可认为该人物实例的关键点被算法正确预测，常用的OKS阈值如0.5和0.75。OKS的计算公式如下所示：\n'},
                    {text:'```\n' +
                            '  OKS的计算公式如下\n' +
                            '```\n'},
                    {katex:'OKS= \\sum _{i}[exp(-d _{i}^{2}/2s^{2}K _{i}^{2})\\delta(v _{i}>0)] / \\sum _{i}[\\delta(v _{i}>0)]'},
                    {img:'评价指标'},
                    {text:'目标检测、语义分割和姿态估计的评价标准\n' +
                            '\n' +
                            '除了OKS之外，还有其他用于评价关键点检测的指标，如关键点正确比（Percentage of Correct Keypoints，PCK），MSCOCO数据集使用OKS作为评价标准，FLIC、LSP、MPII等数据集使用PCK。\n'}
                ]
            }
        },
        {part_title:'推理结果可视化'},
        {text:'在Pet中Simple Baselines返回每一个人物实例的17个关键点并将它们按照MSCOCO的关键点子集的标注顺序连接起来，将验证集中的一张图片的推理结果进行可视化如下图。\n'},
        {img:'demo_0062355_k'},
        {part_title:'参考文献'},
        {text:'\\[1\\] Bin Xiao, Haiping Wu, Yichen Wei. Simple Baselines for Human Pose Estimation and Tracking. ECCV 2018.\n'},
        {text:'\\[2\\] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. CVPR 2017.\n'},
        {text:'\\[3\\] Mingxing Tan, Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. arXiv:1905.11946.\n'},
        {text:'\\[4\\] Vincent Dumoulin, Francesco Visin. A guide to convolution arithmetic for deep learning. arXiv:1603.07285.\n'},
        {text:'\\[4\\] Vincent Dumoulin, Francesco Visin. A guide to convolution arithmetic for deep learning. arXiv:1603.07285.\n'},
        {text:'\\[5\\] Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh. Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. CVPR 2017.\n'},
        {text:'\\[6\\] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked Hourglass Networks for Human Pose Estimation. CVPR 2016.\n'},
        {text:'\\[7\\] Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang Zhang, Gang Yu, Jian Sun. Cascaded Pyramid Network for Multi-Person Pose Estimation. CVPR 2018.\n'},
        {text:'\\[8\\] Kaiming He, Georgia Gkioxari, Piotr Dolla ́r, Ross Girshick. Mask R-CNN. ICCV 2017.\n'},
    ]
}