export let MSCOCO_RetinaNet = {
    key: 'MSCOCO_RetinaNet',
    dataSource: [
        {title:'在MSCOCO2017数据集上训练RetinaNet模型'},
        {text:'本教程将介绍使用Pet训练以及测试RetinaNet模型进行目标检测的主要步骤，在此我们会指导您如何通过组合Pet的提供的各个功能模块来构\n' +
                '建RetinaNet模型，在此我们仅讲解组件的调用，部分实现细节请查阅系统组件的相应部分。在阅读本教程的之前我们强烈建议您阅读原始论文\n' +
                '[RetinaNet](https://arxiv.org/pdf/1708.02002.pdf)\\[1\\]以了解更多关于RetinaNet的算法原理。\n'},
        {
            note:[
                {text:'首先参阅[MSCOCO数据准备](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87.md#mscoco%E6%95%B0%E6%8D%AE%E9%9B%86)教程并在硬盘上准备好MSCOCO2017数据集。\n'}
            ]
        },
        {text:'如果您具有丰富的目标检测算法的研究经验，您也可以直接在Pet中运行`$Pet/tools/rcnn/train_net.py`脚本利己开始训练您的RetinaNet模型.\n'},
        {text:'用法示例：\n'},
        {ul:'在8块GPU上使用`coco_2017_train`训练一个RetinaNet模型，使用ResNet50作为特征提取网络：\n'},
        {text:'```\n' +
                '      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch tools/rcnn/train_net.py --cfg cfgs/rcnn/mscoco/retinanet/retinanet_R-50-FPN-P5_1x.yaml\n' +
                '```\n'},
        {ul:'在8块GPU上使用`coco_2017_val`数据集上测试训练的RetinaNet模型：'},
        {text:'```\n' +
                '      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/rcnn/test_net.py --cfg ckpts/rcnn/mscoco/retinanet/retinanet_R-50-FPN-P5_1x.yaml\n' +
                '```\n'},
        {text:'在进行任何与模型训练和测试有关的操作之前，需要先选择一个指定的`yaml`文件，明确在训练时候对数据集、模型结构、优化策略以及其他重要参数的需求与设置，本教程以$Pet/cfgs/rcnn/mscoco/retinanet/retinanet_R-50-FPN-P5_1x.yaml为例，讲解训练过程中所需要的关键配置，该套配置将指导此Mask R-CNN模型训练以及测试的全部步骤和细节，全部参数设置请见[retinanet_R-50-FPN-P5_1x.yaml]()。\n'},
        {part_title:'数据载入'},
        {text:'确保MSCOCO2017数据集已经存放在您的硬盘中，并按照[数据制备]()中的文件结构整理好MSCOCO数据集的文件结构，接下来我们可以开始加载`coco_2017_train`训练集。\n' +
                '\n' +
                '```Python\n' +
                '    # Create training dataset and loader\n' +
                '    datasets = build_dataset(cfg.TRAIN.DATASETS, is_train=True, local_rank=args.local_rank)\n' +
                '    train_loader = make_train_data_loader(datasets, is_distributed=args.distributed, start_iter=scheduler.iteration)\n' +
                '      cfg.TRAIN.ITER_PER_EPOCH = len(train_loader)\n' +
                '```\n'},
        {part_title: 'RetinaNet网络结构'},
        {
            block:{
                title:'RetinaNet网络设计理念',
                children:[
                    {text:'RetinaNet是2018年Facebook AI团队在目标检测领域新的贡献。其本质上是backbone + FPN + 两个FCN子网络的结构。\n' +
                            '一般backbone网络可选用任一有效的特征提取网络如vgg或resnet系列，此处作者分别尝试了resnet-50与resnet-101。而FPN则是对backbone中提取到的多尺度特征进行了强化利用，从而得到了表达力更强、包含多尺度目标区域信息的特征图。最后在FPN所输出的特征图集合上，分别使用了两个有着相同的网络结构却各自独立，并不共享参数的FCN子网络用来完成目标框类别分类与位置回归任务。\n' +
                            '下图所示为RetinaNet的网络结构示意图：\n'},
                    {img:'pic_11'},
                    {text:'首先RetinaNet的特征提取是由ResNet+FPN完成，输入图像经过其特征提取后，可以得到P3～P7特征图金字塔，在得到特征金字塔后，对每层特征金字塔分别使用两个FCN子网络（分类网络+检测框位置回归）。\n'},
                    {ul:[
                            '与RPN网络类似，RetinaNet也使用anchors来产生proposals。特征金字塔的每层对应一个anchor面积，为了产生更加密集的anchor覆盖，增加了三个面积比例尺度与三个不同长宽比。',
                            '原始RPN网络的分类网络只是区分前景与背景两类，在RetinaNet中其类别数为K。',
                            '与分类分支并行，RetinaNet对每一层FPN输出的特征图接上一个框回归分支，该分支本质也是FCN网络，预测的是anchor和它对应的一个GT位置的偏移量，即对每一个anchor，回归一个（x,y,w,h）四维向量。'
                        ]},

                ]
            }
        },
        {text:'在Pet中，RetinaNet网络使用`Generalized_RCNN`模型构建器来搭建网络，详情见[模型构建]()。我们在`yaml`文件中设置如下参数来使用配置系统构建RetinaNet网络中特征提取网络的构建：\n' +
                '\n' +
                '```\n' +
                '  MODEL:\n' +
                '    FPN_ON: True\n' +
                '    RPN_ONLY: True\n' +
                '    RETINANET_ON: True\n' +
                '    NUM_CLASSES: 81\n' +
                '    CONV1_RGB2BGR: False\n' +
                '  BACKBONE:\n' +
                '    CONV_BODY: "resnet"\n' +
                '    RESNET:  # caffe style\n' +
                '    LAYERS: (3, 4, 6, 3)\n' +
                '\n' +
                '```\n' +
                '在本教程中，我们使用ResNet50作为特征提取网络，在``cfg.BACKBONE.CONV_BODY``中进行设置。在构建完特征提取网络之后，通过如下参数进行RetinaNet及FPN相关参数的配置与网络模块的构建：\n' +
                '\n' +
                '```\n' +
                '  RETINANET:\n' +
                '    SCALES_PER_OCTAVE: 3\n' +
                '    STRADDLE_THRESH: -1\n' +
                '    FG_IOU_THRESHOLD: 0.5\n' +
                '    BG_IOU_THRESHOLD: 0.4\n' +
                '  FPN:\n' +
                '    USE_C5: False\n' +
                '    RPN_MAX_LEVEL: 7\n' +
                '    RPN_MIN_LEVEL: 3\n' +
                '    EXTRA_CONV_LEVELS: True\n' +
                '    MULTILEVEL_ROIS: False\n' +
                '```\n'},
        {part_title:'训练'},
        {text:'完成了数据载入以及模型构建之后，我们需要在开始训练之前选择训练RetinaNet模型的优化策略。\n' +
                '\n' +
                '```Python\n' +
                'def train(model, loader, optimizer, scheduler, checkpointer, logger):\n' +
                '    # switch to train mode\n' +
                '    model.train()\n' +
                '\n' +
                '    # main loop\n' +
                '    start_iter = scheduler.iteration\n' +
                '    for iteration, (images, targets, _) in enumerate(loader, start_iter):\n' +
                '        logger.iter_tic()\n' +
                '        logger.data_tic()\n' +
                '\n' +
                '        scheduler.step()    # adjust learning rate\n' +
                '        optimizer.zero_grad()\n' +
                '\n' +
                '        images = images.to(args.device)\n' +
                '        targets = [target.to(args.device) for target in targets]\n' +
                '        logger.data_toc()\n' +
                '\n' +
                '        outputs = model(images, targets)\n' +
                '\n' +
                '        logger.update_stats(outputs, args.distributed, args.world_size)\n' +
                '        loss = outputs[\'total_loss\']\n' +
                '        loss.backward()\n' +
                '        optimizer.step()\n' +
                '\n' +
                '        if args.local_rank == 0:\n' +
                '            logger.log_stats(scheduler.iteration, scheduler.new_lr)\n' +
                '\n' +
                '            # Save model\n' +
                '            if cfg.SOLVER.SNAPSHOT_ITERS > 0 and (iteration + 1) % cfg.SOLVER.SNAPSHOT_ITERS == 0:\n' +
                '                checkpointer.save(model, optimizer, scheduler, copy_latest=True, infix=\'iter\')\n' +
                '        logger.iter_toc()\n' +
                '    return None\n' +
                '```\n' +
                '\n' +
                '在训练过程中，日志记录仪会在每若干次迭代后记录当前网络训练的迭代数、各项偏差数值等训练信息，检查点组件会定期保存网络模型到配置系统中`cfg.CKPT`所设置的路径下。\n' +
                '\n' +
                '根据`cfg.DISPLAY_ITER`设置的日志记录间隔，在训练过程中每经过20次迭代，日志记录仪会在终端中记录模型的训练状态。\n'},
        {
            block:{
                title:'Focal Loss',
                children:[
                    {text:' 在常见的单阶段检测器中，卷积网络在得到特征图后，会产生密集的目标候选区域，而这些大量的候选区域中只有很少一部分是真正的目标，这样就造成了机器学习中经典的训练样本正负不平衡的问题。它往往会造成最终计算出的loss被占绝大多数但包含信息量很少的负样本所支配，少样正样本提供的关键信息却不能在loss中发挥正常指导作用，\n' +
                            ' 同时，两阶段方法得到proposal后，其候选区域要远远小于单阶段检测器产生的候选区域，因此不会产生严重的类别失衡问题。常用的解决此问题的方法就是负样本挖掘，或其它更复杂的用于过滤负样本从而使正负样本数维持一定比率的样本取样方法。\n' +
                            ' [RetinaNet](https://arxiv.org/pdf/1708.02002.pdf)\\[1\\]论文中提出了Focal Loss来对最终的Loss进行校正。\n' +
                            ' \n' +
                            ' Focal Loss是一个动态缩放的交叉熵损失，一言以蔽之，通过一个动态缩放因子，可以动态降低训练过程中易区分样本的权重，从而将loss的重心快速聚焦在那些难区分的样本上，它是在Cross-Entropy(CE) loss的基础上进行改进，主要就是增加了一个调节因子，为不同的样本赋予不同的权重。\n'},
                    {ul:'Cross Entropy(CE) Loss'},
                    {text:' 首先，我们先介绍一下CE loss，二分类的CE loss为： \n'},
                    {katex:'CE(p,y) = \n' +
                            '\\begin{cases}\n' +
                            '-log(p) & if\\ y = 1 \\\\\\\n' +
                            '-log(1-p) & otherwise\n' +
                            '\\end{cases}\n'},
                    {text:'其中，y是ground truth的类别，而p是模型预测的 y = 1 的概率值，为了方便表达，我们定义了pt：\n'},
                    {katex:'p _{t} = \n' +
                            '\\begin{cases}\n' +
                            'p & if\\ y = 1 \\\\\\\n' +
                            '1-p & otherwise\n' +
                            '\\end{cases}\n'},
                    {ul:'Balanced CE loss'},
                    {text:'通常，解决类别不均衡的常用方法就是引入权重α:\n'},
                    {katex:'CE(p _{t}) = -\\alpha _{t}log(p _{t})'},
                    {text:'当类别标签是1时，权重因子是α，当类别标签是-1时，权重因子是1-α。同样为了表示方便，用αt表示权重因子。\n'},
                    {ul:'Focal loss'},
                    {text:'其实，Focal loss可以看成是Balanced CE loss的一种特殊形式，只是将权重 αt 实例化为 （1-pt）^ γ，因此focal loss定义为：\n'},
                    {katex:'FL(p _{t}) = -(1 - p _{t}) ^{\\gamma}log (p _{t})'},
                    {text:'其中，（1-pt）^ γ就是那个调节因子，γ是超参数。当一个样本被分错时，pt就比较小，（1-pt）^ γ就比较大，也就是权重比较大；反之亦反。因此，简单样本的loss会被缩小，而困难样本的loss会被放大。\n' +
                            '此外，当γ取不同值时，可以得到下图：\n'},
                    {img:'pic_16'},
                ]
            }
        },
        {part_title:'测试'},
        {text:'在完成RetinaNet模型的训练之后，我们使用Pet/tools/rcnn/test_net.py在`coco_2017_val`上评估模型的精度。同样需需要使用`Dataloader`来加载测试数据集。\n' +
                '\n' +
                '通过加载训练最大迭代数之后的模型`Pet/ckpts/rcnn/mscoco/retinanet/retinanet_R-50-FPN-P5_1x/model_latest.pth`，执行下面的命令进行模型的测试。\n' +
                '\n' +
                '```\n' +
                '      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/rcnn/test_net.py --cfg cfgs/rcnn/mscoco/retinanet/retinanet_R-50-FPN-P5_1x.yaml\n' +
                '```\n'},
        {part_title:'推理结果可视化'},
        {text:'在Pet中RetinaNet返回每一个目标的类别ID、置信度分数，边界框坐标。将MSCOCO2017_val中的一张图片的推理结果进行可视化如下图。\n'},
        {img:'pic_000000102331'},
        {part_title:'参考文献'},
        {text:'\\[1\\] Lin T Y , Goyal P , Girshick R , et al. Focal Loss for Dense Object Detection[J]. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2017, PP(99):2999-3007.\n'}
    ]
}