export let MSCOCO_SSD = {
    key: 'MSCOCO_SSD',
    dataSource: [
        {title:'在MSCOCO2017数据集上训练SSD模型'},
        {text:'本教程将介绍使用Pet训练以及测试SSD模型进行目标检测的主要步骤，在此我们会指导您如何通过组合Pet的提供的各个功能模块来构\n' +
                '建SSD模型，在此我们仅讲解组件的调用，部分实现细节请查阅系统组件的相应部分。在阅读本教程的之前我们强烈建议您阅读原始论文\n' +
                '[SSD](https://arxiv.org/abs/1512.02325)\\[1\\],[FSSD](https://arxiv.org/abs/1712.00960v1)[2]以了解更多关于\n' +
                'SSD的算法原理。\n'},
        {text:'如果您具有丰富的目标检测算法的研究经验，您也可以直接在Pet中运行[$Pet/tools/ssd/train_net.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/tools/ssd/train_net.py)脚本利己开始训练您的SSD模型.\n' +
                '\n' +
                '用法示例：\n'},
        {ul:'* 在8块GPU上使用`coco_2017_train`训练一个SSD模型，使用VGG16作为特征提取网络：'},
        {shell:'```\n' +
                '      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch tools/ssd/train_net.py \n' +
                '      --cfg cfgs/ssd/mscoco/ssd_VGG16_300x300_1x.yaml\n' +
                '```\n'},
        {text:'在8块GPU上使用`coco_2017_val`数据集上测试训练的SSD模型：'},
        {shell:'```\n' +
                '      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/ssd/test_net.py --cfg ckpts/ssd/mscoco/ssd_VGG16_300x300_1x.yaml\n' +
                '```\n'},
        {text:'在进行任何与模型训练和测试有关的操作之前，需要先选择一个指定的`yaml`文件，明确在训练时候对数据集、模型结构、优化策略以及其他重要参数的需求与设置，本教程以$Pet/cfgs/ssd/mscoco/ssd_VGG16_300x300_1x.yaml为例，讲解训练过程中所需要的关键配置，该套配置将指导此Mask R-CNN模型训练以及测试的全部步骤和细节，全部参数设置请见[ssd_VGG16_300x300_1x.yaml](https://github.com/BUPT-PRIV/Pet-dev/blob/master/cfgs/ssd/mscoco/ssd_VGG16_300x300_1x.yaml)。\n'},
        {part_title:'数据载入'},
        {text:'确保MSCOCO2017数据集已经存放在您的硬盘中，并按照[数据制备](https://github.com/BUPT-PRIV/Pet-DOC/tree/master/%E6%9E%B6%E6%9E%84/component-collects/%E6%95%B0%E6%8D%AE%E5%88%B6%E5%A4%87)中的文件结构整理好MSCOCO数据集的文件结构，接下来我们可以开始加载`coco_2017_train`训练集。\n' +
                '\n' +
                '```Python\n' +
                '      # Create training dataset and loader\n' +
                '      train_set = build_dataset(cfg.TRAIN.DATASETS, is_train=True, local_rank=args.local_rank)\n' +
                '      train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if args.distributed else None\n' +
                '      ims_per_gpu = int(cfg.TRAIN.BATCH_SIZE / args.world_size)\n' +
                '      train_loader = make_train_data_loader(train_set, ims_per_gpu, train_sampler)\n' +
                '\n' +
                '      cfg.TRAIN.ITER_PER_EPOCH = len(train_loader)\n' +
                '```\n' +
                '\n' +
                'SSD中采用数据增强可以大幅度提升网络性能，主要采用的手段有水平翻转，随机裁剪，颜色扭曲，随机采集块域，随机缩放图像以获取小目标训练样本等。下图所示为MSCOCO2017训练集中的一张图片与其进行SSD数据增强之后的效果对比。\n'},
        {img:'ssd_1527'},
        {text:'而在测试阶段，数据增强方式则要简单的多，只对测试图片进行尺度变换以及颜色扭曲。\n'},
        {part_title: 'SSD网络结构'},
        {
            block:{
                title:'SSD网络设计理念',
                children:[
                    {text:'SSD，全称Single Shot MultiBox Detector，是Wei Liu在ECCV 2016上提出的一种目标检测算法，也是截至目前最主要的检测框架之一。SSD网络主体设计的思想是特征分层提取，并依次进行边框回归和分类。因为不同层次的特征图能代表不同层次的语义信息，低层次的特征图能代表低层语义信息(含有更多的细节)，能提高语义分割质量，适合小尺度目标的学习。高层次的特征图能代表高层语义信息，能光滑分割结果，适合对大尺度的目标进行深入学习。所以作者提出的SSD的网络理论上能适合不同尺度的目标检测。下图所示为SSD的网络结构示意图：\n'},
                    {img:'ssd结构'},
                    {text:'目前目标检测分为了2种主流框架：'},
                    {ul:[
                        'Two stages：以Faster R-CNN为代表，即RPN网络先生成proposals目标定位，再对proposals进行classification + bounding box regression完成目标分类。',
                            'Single stage：以YOLO/SSD为代表，一次性完成classification + bounding box regression。'
                        ]},
                    {text:'作为单阶段的目标检测代表性算法，SSD具有如下主要特点：\n'},
                    {ul:[
                            '从YOLO中继承了将目标检测任务转化为回归的思路，一次性完成目标的定位与分类;',
                            '基于Faster R-CNN中的Anchor，提出了相似的Prior Box;',
                            '加入基于特征金字塔的检测方式，即在不同感受野的特征图上预测目标；'
                        ]},
                ]
            }
        },
        {text:'Pet使用`Generalized_SSD`来搭建SSD网络，详情见[模型构建](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%9E%B6%E6%9E%84/component-collects/%E6%A8%A1%E5%9E%8B%E6%9E%84%E5%BB%BA/%E6%A8%A1%E5%9E%8B%E6%9E%84%E5%BB%BA.md)。我们在YAML文件中设置如下参数来使用配置系统构建SSD网络中特征提取网络的构建：\n' +
                '\n' +
                '```\n' +
                '      MODEL:\n' +
                '        TYPE: generalized_ssd\n' +
                '        ANCHOR_ON: True\n' +
                '        NUM_CLASSES: 81\n' +
                '      BACKBONE:\n' +
                '        CONV_BODY: vgg16\n' +
                '```\n' +
                '在本教程中，我们使用vgg16作为特征提取网络，在`cfg.BACKBONE.CONV_BODY`中进行设置。在构建完特征提取网络之后，通过如下参数进行功能网络、任务输出以及损失函数等网络模块的构建：\n' +
                '\n' +
                '```\n' +
                '      ANCHOR:\n' +
                '        BOX_HEAD: ssd_xconv_head\n' +
                '        L2NORM: True\n' +
                '        LOSS: \'ohem\'\n' +
                '        SSD:\n' +
                '          HEAD_DIM: 256\n' +
                '          STRIDE_E3E4: False\n' +
                '```\n'},
        {
            block:{
                title:'SSD先验框设置及匹配策略',
                children:[
                    {text:'在YOLO中，每个单元预测多个边界框，但是其都是相对这个单元本身（正方块），但是真实目标的形状是多变的，Yolo需要在训练过程中自适应目标的形状。而SSD借鉴了Faster R-CNN中anchor的理念，每个单元设置尺度或者长宽比不同的先验框，预测的边界框（bounding boxes）是以这些先验框为基准的，在一定程度上减少训练难度。一般情况下，每个单元会设置多个先验框，其尺度和长宽比存在差异，如图所示，可以看到每个单元使用了4个不同的先验框，图片中针对目标分别采用最适合它们形状的先验框来进行训练。\n'},
                    {img:'SSD2'},
                    {text:'在训练时，ground truth boxes与default boxes（即prior boxes）按照如下方式进行配对：\n'},
                    {
                        ul:[
                            '首先，寻找与每一个ground truth box有最大的jaccard overlap的default box，这样就能保证每一个ground truth box与唯一的一个default box对应起来。',
                            '将剩余还没有配对的default box与任意一个ground truth box尝试配对，只要两者之间的jaccard overlap大于阈值，就认为match（SSD 300设置阈值为0.5）。显然配对到GT的default box就是positive，没有配对到GT的default box就是negative。'
                        ]
                    }
                ]
            }
        },
        {part_title:'训练'},
        {text:'完成了数据载入以及模型构建之后，我们需要在开始训练之前选择训练SSD模型的优化策略，在批次大小为64的情况下，设置初始学习率为0.002，训练120次全数据迭代，组合使用了学习率预热与阶段下降策略，分别在80与110次全数据迭代时将学习率减小十倍。\n' +
                '\n' +
                '```Python\n' +
                '      def train(model, loader, optimizer, scheduler, logger, priors):\n' +
                '          # switch to train mode\n' +
                '          model.train()\n' +
                '\n' +
                '          # main loop\n' +
                '          logger.iter_tic()\n' +
                '          logger.data_tic()\n' +
                '          for i, (inputs, targets, _) in enumerate(loader):\n' +
                '              scheduler.step()  # adjust learning rate\n' +
                '              optimizer.zero_grad()\n' +
                '\n' +
                '              inputs = inputs.to(torch.device(\'cuda\'))\n' +
                '              targets = [target.to(torch.device(\'cuda\')) for target in targets]\n' +
                '              logger.data_toc()\n' +
                '\n' +
                '              outputs = model(inputs.tensors, (targets, priors))\n' +
                '              logger.update_stats(outputs, args.distributed, args.world_size)\n' +
                '              loss = outputs[\'total_loss\']\n' +
                '              if cfg.SOLVER.AMP.ENABLED:\n' +
                '                  with amp.scale_loss(loss, optimizer) as scaled_loss:\n' +
                '                      scaled_loss.backward()\n' +
                '              else:\n' +
                '                  loss.backward()\n' +
                '              optimizer.step()\n' +
                '\n' +
                '              if args.local_rank == 0:\n' +
                '                  logger.log_stats(scheduler.iteration, scheduler.new_lr)\n' +
                '\n' +
                '              logger.iter_toc()\n' +
                '              logger.iter_tic()\n' +
                '              logger.data_tic()\n' +
                '          return None\n' +
                '\n' +
                '      # Train\n' +
                '          logging_rank(\'Training starts.\', distributed=args.distributed, local_rank=args.local_rank)\n' +
                '          start_epoch = scheduler.iteration // cfg.TRAIN.ITER_PER_EPOCH + 1\n' +
                '          for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS + 1):\n' +
                '              train_sampler.set_epoch(epoch) if args.distributed else None\n' +
                '\n' +
                '              # Train model\n' +
                '              logging_rank(\'Epoch {} is starting.\'.format(epoch), distributed=args.distributed, local_rank=args.local_rank)\n' +
                '              train(model, train_loader, optimizer, scheduler, training_logger, priors)\n' +
                '\n' +
                '              # Save model\n' +
                '              if args.local_rank == 0:\n' +
                '                  snap_flag = cfg.SOLVER.SNAPSHOT_EPOCHS > 0 and epoch % cfg.SOLVER.SNAPSHOT_EPOCHS == 0\n' +
                '                  checkpointer.save(model, optimizer, scheduler, copy_latest=snap_flag, infix=\'epoch\')\n' +
                '\n' +
                '          logging_rank(\'Training done.\', distributed=args.distributed, local_rank=args.local_rank)\n' +
                '```\n' +
                '\n' +
                '在训练过程中，日志记录仪会在每若干次迭代后记录当前网络训练的迭代数、各项偏差数值等训练信息，检查点组件会定期保存网络模型到配置系统中`cfg.CKPT`所设置的路径下。\n'},
        {text:'根据`cfg.DISPLAY_ITER`设置的日志记录间隔，在训练过程中每经过20次迭代，日志记录仪会在终端中记录模型的训练状态。\n'},
        {
            block:{
                title:'SSD训练策略',
                children:[
                    {ul:'联合损失函数'},
                    {text:'SSD网络对于每个stage输出的特征图都进行边界框的回归和分类，设计了一个联合损失函数如下：'},
                    {katex:'L(x,c,l,g)=\\frac{1}{N}(L _{conf} (x,c)+\\alpha L _{loc}(x,l,g))'},
                    {text:'其中：$$L _{conf}$$表示分类误差，采用多分类任务中常用的softmax loss；$$L _{loc}$$表示框回归误差，采用SmothL1 Loss；采用这种联合损失函数的方式同时完成框回归与分类。'},
                    {ul:'Hard Negative Mining'},
                    {text:'经过匹配策略会得到大量的负样本，只有少量的正样本。这样会导致正负样本不平衡，而正负样本的不均衡是导致检测正确率偏低的一个重要原因。所以在训练过程中采用了Hard Negative Mining的策略，根据Confidence Loss对所有的box进行排序，按照loss降序取正样本三倍数量的负样本，既保证了负样本的困难程度，同时使得正负样本的比例控制在1:3之内，通过实验验证，这样做能提高4%左右的准确度。\n'},
                    {ul:'Data Augmentation'},
                    {text:'即对每一张image进行如下之一变换获取一个patch进行训练：\n'},
                    {ul:[
                            '直接使用原始的图像（即不进行变换）',
                            '采样一个patch，保证与Ground Truth之间最小的IoU为：0.1，0.3，0.5，0.7 或 0.9',
                            '完全随机的采样一个patch'
                        ]},
                    {text:'同时在数据增强过程中采样的patch占原始图像大小比例在[0.1, 1]之间,采样的patch的长宽比在[0.5 ,2]之间,当Ground truth box中心恰好在采样的patch中时，保留整个Ground truth box,最后每个patch被resize到固定大小，并且以0.5的概率随机的水平翻转最终以这些处理好的patches进行训练。'}
                ]
            }

        },
        {part_title:'测试'},
        {text:'在完成SSD模型的训练之后，我们使用[$Pet/tools/ssd/test_net.py](https://github.com/BUPT-PRIV/Pet/blob/master/tools/ssd/test_net.py)在`coco_2017_val`上评估模型的精度。同样需需要使用`Dataloader`来加载测试数据集。\n'},
        {text:'通过加载训练最大迭代数之后的模型`$Pet/ckpts/ssd/mscoco/ssd_VGG16_300x300_1x/model_latest.pth`，执行下面的命令进行模型的测试。\n'},
        {shell:'```\n' +
                '      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/ssd/test_net.py ----cfg cfgs/ssd/mscoco/ssd_VGG16_300x300_1x.yaml\n' +
                '```\n'},
        {part_title:'推理结果可视化'},
        {text:'在Pet中SSD返回每一个目标的类别ID、置信度分数，边界框坐标。将MSCOCO2017_val中的一张图片的推理结果进行可视化如下图。\n'},
        {img:'ssd_test_062355'},
        {part_title:'参考文献'},
        {text:'\\[1\\] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. ECCV 2016.\n'}
    ]
}