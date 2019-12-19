export let ImageNet_train = {
    key: 'ImageNet_train',
    dataSource: [
        {title:'ImageNet分类教程'},
        {text:'本教程将介绍使用Pet在ImageNet数据集上进行分类网络的训练以及测试的基本流程。通过本教程，您将学会如何结合Pet提供的组件进行分类任务模型\n' +
                '的训练。在此期间，我们只进行组件的调用，组件的实现细节您可以参阅[component-collects](https://github.com/BUPT-PRIV/Pet-DOC/tree/master/%E6%9E%B6%E6%9E%84/component-collects)部分。\n'},
        {text:'如果您已经有分类任务的训练经验，您可以直接使用[$Pet/tools/cls/train_net.py](https://github.com/BUPT-PRIV/Pet-dev/blob/e11ef696c92ea5e4cf30609fb67420a262c911ca/tools/cls/train_net.py)脚本开始训练。\n'},
        {text:'用法示例：\n'},
        {shell:'```\n' +
                '    cd $Pet\n' +
                '    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/cls/train_net.py --cfg /$Pet/cfgs/cls/imagenet/resnet/resnet50a.yaml\n' +
                '```\n'},
        {text:'在开始相关训练之前，我们的Pet需要指定一个`yaml`文件，该文件里包含了所有训练时使用到的可以调节的参数，包括学习率、GPU数量、网络结构、全数据迭代次数等等。此次教程以`$Pet/cfgs/cls/imagenet/resnet/resnet50_1x64d.yaml`为例，以训练脚本为主，讲解训练过程中需要的关键配置以及训练步骤，详细参数见[$Pet/cfgs/cls/imagenet/resnet/resnet50a.yaml](https://github.com/BUPT-PRIV/Pet-dev/blob/master/cfgs/cls/imagenet/resnet/resnet50a.yaml)。\n'},
        {part_title:'数据载入'},
        {text:'确保ImageNet数据已经存入您的电脑硬盘中，文件夹结构如下：\n' +
                '\n' +
                '```\n' +
                '    ImageNet\n' +
                '      |--class1\n' +
                '        |--pic1.jpg\n' +
                '        |--pic2.jpg\n' +
                '        ...\n' +
                '      |--class2\n' +
                '      ...\n' +
                '   \n' +
                '```\n' +
                '\n' +
                '训练数据集分两个部分，一个是训练集一个是验证集：\n' +
                '\n' +
                '```Python\n' +
                '    # Dataset and Loader\n' +
                '    train_set = datasets.ImageFolder(\n' +
                '        os.path.join(cfg.DATASET.DATA_ROOT, cfg.TRAIN.DATASETS),\n' +
                '        transforms.Compose([\n' +
                '            transforms.RandomResizedCrop(cfg.TRAIN.AUG.CROP_SIZE),\n' +
                '            transforms.RandomHorizontalFlip(),\n' +
                '            # transforms.ToTensor(), Too slow\n' +
                '            # normalize,\n' +
                '        ])\n' +
                '    )\n' +
                '\n' +
                '```\n' +
                '\n' +
                '利用PyTorch自带的数据处理类`ImageFolder`，`ImageFolder`将父文件夹下的子文件夹读取创建字典，并对应为类别标签，文件夹目录在yaml文件下`DATASET:DATA_ROOT`中进行设置。为了达到数据增强的效果，以减少一定的过拟合现象，我们使用PyTorch自带的transforms函数对数据进行预处理，使用水平方向上以一定的概率进行图片翻转来进行数据增强，将图片进行随机裁剪至`224*224`，送入网络。除此之外，我们还支持另一种数据增强方式——mixup。`cfg.TRAIN.AUG`则是可以在配置文件中针对数据增强可以被设置的参数。`ImageFolder`类的具体实现请阅读PyTorch官方代码。\n'},
        {
            block:{
                title:'mixup',
                children:[
                    {text:'mixup是一种非常规的数据增强方法，一个和数据无关的简单数据增强原则，其以线性插值的方式来构建新的训练样本和标签。\n' +
                            '\n' +
                            '公式为：\n'},
                    {katex:'\\tilde{X} = \\partial \\times X _{i} + (1 - \\partial) \\times X _{j}'},
                    {katex:'\\tilde{Y} = \\partial \\times Y _{i} + (1 - \\partial) \\times Y _{j}'},
                    {text:'效果如下：\n'},
                    {
                        table:{
                            titles:['模型','数据增强方法','epochs','top-1 error','top-5 error'],
                            data:[
                                ['resnet50','ERM',90,23.5,' - '],['resnet50','mixup α=0.2',90,23.3,' - '],
                                ['resnet50','ERM',200,23.6,7.0],['resnet50','mixup α=0.2',200,22.1,6.1]
                            ]
                        }
                    },
                    {text:'简单理解就是从处理好的批数据中随机抽取两张图片以一定比例进行融合，效果就像一张图片在另一图片上以一定的透明度显示，这个透明度就是mixup的参数，\n' +
                            '更加具体的效果和详细的原理以及实验结果请阅读[mixup原文](https://arxiv.org/abs/1710.09412v2)。\n'}
                ]
            },
        },
        {text:'```Python\n' +
                '    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)\n' +
                '    ims_per_gpu = int(cfg.TRAIN.BATCH_SIZE / args.world_size)\n' +
                '    train_loader = torch.utils.data.DataLoader(\n' +
                '        train_set,\n' +
                '        batch_size=ims_per_gpu,\n' +
                '        shuffle=False,5\n' +
                '        num_workers=cfg.DATA_LOADER.NUM_THREADS,\n' +
                '        pin_memory=True,\n' +
                '        sampler=train_sampler,\n' +
                '        collate_fn=fast_collate\n' +
                '    )\n' +
                '    cfg.TRAIN.ITER_PER_EPOCH = len(train_loader)\n' +
                '```\n' +
                '\n' +
                '我们通过PyTorch自带的`DataLoader`类进行数据的封装，它的作用是把上一步准备好的数据进行批量划分、分发，有必要时可以将数据进行打乱。`DataLoader`类的具体实现请阅读PyTorch官方代码。\n' +
                '\n' +
                '```Python\n' +
                '    if cfg.TEST.DATASETS:\n' +
                '        # val_set = datasets.ImageFolder(os.path.join(cfg.DATASET.DATA_ROOT, cfg.TEST.DATASETS),\n' +
                '        #                                transforms.Compose(test_aug)\n' +
                '        #                                )\n' +
                '        val_set = datasets.ImageFolder(\n' +
                '            os.path.join(cfg.DATASET.DATA_ROOT, cfg.TEST.DATASETS),\n' +
                '            transforms.Compose([\n' +
                '                transforms.Resize(cfg.TEST.AUG.BASE_SIZE),\n' +
                '                transforms.CenterCrop(cfg.TEST.AUG.CROP_SIZE),\n' +
                '            ])\n' +
                '        )\n' +
                '        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)\n' +
                '        val_loader = torch.utils.data.DataLoader(\n' +
                '            val_set,\n' +
                '            batch_size=cfg.TEST.BATCH_SIZE,\n' +
                '            shuffle=False,\n' +
                '            num_workers=cfg.DATA_LOADER.NUM_THREADS,\n' +
                '            pin_memory=True,\n' +
                '            sampler=val_sampler,\n' +
                '            collate_fn=fast_collate\n' +
                '        )\n' +
                '```\n' +
                '\n' +
                '验证数据集的处理和封装与训练数据集相同，这里不再作详细解释。\n'},
        {part_title: '网络结构'},
        {text:'在训练脚本开始，引入**Pet**下models模块，生成基础网络，网络结构名称在yaml文件下`MODEL:CONV_BODY`中进行设置。\n' +
                '\n' +
                '```Python\n' +
                '    import pet.models.imagenet as models\n' +
                '\n' +
                '    ...\n' +
                '\n' +
                '    model = models.__dict__[cfg.MODEL.CONV_BODY]()\n' +
                '```\n' +
                '\n'
                },
        {
            block:{
                title:'深度残差网络',
                children:[
                    {text:'网络结构如下所示，resnet最核心的在于“捷径连接（shortcut connection）”，就是将某一层的输入直接添加到一层或几层之后，可以避免梯度在反传时出\n' +
                            '现梯度消失，每一个包含shortcut connection的块称之为“残差块（residual block）”。\n'},
                    {img:'resnet'},
                    {text:'而根据论文所述，在这些块中有一种比较特殊的情况，就是bottleneck，根据以下打印出的网络结构或是上图右边的结构可以看到，在Sequential中的第一个\n' +
                            '参数里有一个Bottleneck字样，里面包含着`1*1`,`3*3`,`1*1`的卷积核，这样卷积核尺寸为“小大小”的卷积块结构就是bottleneck结构。在resnet-v1\n' +
                            '中提出并且开始使用bottleneck结构，`1*1`卷积核的使用加入了更多的非线性，因为将会引入两个ReLU激活函数，同时能够在保持特征图尺度不变的情况下改\n' +
                            '变通道数量，另外一个优点是能够在相同计算量的情况下降低参数量，我们来计算一下，普通的残差块参数量为：(3x3x256x256)x2=1179648，bottleneck\n' +
                            '的参数量为：1x1x256x64+3x3x64x64+1x1x64x256=69632，优势显而易见。在resnet-v2中，实验结果结果证明效果最好的顺序是BN、ReLU、conv顺序连\n' +
                            '接，同时也是我们所使用的顺序。在使用bottleneck降低了参数量之后，还可以从计算量的角度去优化网络，resnext中引入的分组卷积就能够比较好的实现这\n' +
                            '样的优化（如下图），我们来计算一下分组卷积和普通卷积的计算量：以28x28x28的输入、3x3x28x64的卷积核、分为2组为例，普通卷积的计算量:3x3x28x\n' +
                            '(28-3+1)x(28-3+1)x64=10902528,分组卷积的计算量为：3x3x14x(28-3+1)x(28-3+1)x32x2=5451264，可以看出分组卷积的计算量要小于普通卷积。\n' +
                            '更多详细的结构介绍、原理和实验结果请阅读[resnet-v1](https://arxiv.org/abs/1512.03385v1)\\[1\\]、[resnet-v2](https://arxiv.org/abs/1603.05027v3)\\[2\\]、[resnext](https://arxiv.org/abs/1611.05431v2)\\[3\\]\n'},
                    {img:'ResNeXt'}
                ]
            }
        },
        {text:'```\n' +
                '    ResNet(\n' +
                '      (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\n' +
                '      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n' +
                '      (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n' +
                '      (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n' +
                '      (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n' +
                '      (bn3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n' +
                '      (relu): ReLU(inplace)\n' +
                '      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)\n' +
                '      (layer1): Sequential(\n' +
                '        (0): Bottleneck(\n' +
                '          (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)\n' +
                '          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n' +
                '          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n' +
                '          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n' +
                '          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n' +
                '          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n' +
                '          (relu): ReLU(inplace)\n' +
                '          (downsample): Sequential(\n' +
                '            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n' +
                '            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n' +
                '          )\n' +
                '        )\n' +
                '         (1): Bottleneck(\n' +
                '          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)\n' +
                '          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n' +
                '          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n' +
                '          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n' +
                '          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n' +
                '          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n' +
                '          (relu): ReLU(inplace)\n' +
                '        )\n' +
                '        (2): Bottleneck(\n' +
                '          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)\n' +
                '          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n' +
                '          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n' +
                '          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n' +
                '          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n' +
                '          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n' +
                '          (relu): ReLU(inplace)\n' +
                '        )\n' +
                '      )\n' +
                '......\n' +
                '```\n'},
        {
            block:{
                title:'网络参数与计算量',
                children:[
                    {text:'除了网络结构，我们需要对网络的参数量以及计算量有所了解，参数量的多少主要涉及到内存的大小，一般我们的参数是单精度浮点数，即float32，其占据4B\n' +
                            '的字节，而计算量我们以FLOPs表示，代表着网络计算需要的时间，从而可以估计出训练模型需要的时间长短。\n'},
                    {text:'参数量计算公式：\n'},
                    {katex:'params _{conv}= K _{h} \\times K _{w} \\times C _{in} \\times C _{out}+C _{out}'},
                    // {katex:'(K _{h},K _{w})表示卷积核的高和宽,(C _{in},C _{out})表示输入通道以及输出通道数量)'},
                    {katex:'params _{fc} = U _{in} \\times U _{out} + U _{out}'},
                    {text:'FLOPs的计算公式：\n'},
                    {katex:'FLOPs = params \\times H  \\times W'},
                    {text:'计算示例：输入以mnist手写数据集为例\n'},
                    {
                        table:{
                            titles:['网络层','输入','卷积核','输出','参数量','占用内存','FLOPs'],
                            data:[
                                ['输入图尺寸','32x32x1','-','32x32x1','-','-','-'],
                                ['conv1','32x32x1','5x5x6','28x28x6','5x5x1x6+6=156','624B','156x28x28'],
                                ['maxpool1','28x28x6','2x2','14x14x6','-','-','-'],
                                ['conv2','14x14x6','5x5x16','10x10x16','5x5x6x16+6=2416','9664B','9664x10x10'],
                                ['maxpool2','10x10x16','2x2','5x5x16','-','-','-'],
                                ['fc1','5x5x16','-',120,'5x5x16x120+1=48001','192004B','48001x1'],
                                ['fc2',120,'-',84,'120x84+1=10081','40324B','40324x1']
                            ]
                        }
                    }
                ]
            }
        },
        {part_title:'训练'},
        {text:'完成数据载入之后，我们将进行训练，训练中所涉及到的参数，例如，批数据量大小、全数据迭代次数、学习率、优化器等都可以在`resnet50_1x64d.yaml`文件中进行设置。这里我们将使用标准的训练策略进行网络的训练，即学习率为0.1，优化器为SGD，批数据量大小为256（每块GPU放入64张图），动量为0.9，这里我们使用效果比较好的`SGD+Momentum`的优化方式。训练脚本中所涉及的其他组件请参阅[Documents](https://github.com/BUPT-PRIV/Pet-DOC/tree/master/%E6%9E%B6%E6%9E%84/component-collects)部分，这里不作详细说明。\n'},
        {text:'```Python\n' +
                '    def train(model, criterion, loader, optimizer, scheduler, logger):\n' +
                '        # switch to train mode\n' +
                '        model.train()\n' +
                '\n' +
                '        prefetcher = DataPreFetcher(loader, means=cfg.PIXEL_MEANS, stds=cfg.PIXEL_STDS)\n' +
                '        inputs, targets = prefetcher.next()\n' +
                '\n' +
                '        # main loop\n' +
                '        logger.iter_tic()\n' +
                '        logger.data_tic()\n' +
                '        while inputs is not None:\n' +
                '            scheduler.step()  # adjust learning rate\n' +
                '            optimizer.zero_grad()\n' +
                '            logger.data_toc()\n' +
                '\n' +
                '            outputs = model(inputs)\n' +
                '            loss = criterion(outputs, targets)\n' +
                '            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))\n' +
                '            logger.update_stats({\'losses\': {\'loss\': loss}, \'metrics\': {\'acc1\': acc1, \'acc5\': acc5}},\n' +
                '                                args.distributed, args.world_size)\n' +
                '            if cfg.SOLVER.AMP.ENABLED:\n' +
                '                with amp.scale_loss(loss, optimizer) as scaled_loss:\n' +
                '                    scaled_loss.backward()\n' +
                '            else:\n' +
                '                loss.backward()\n' +
                '            optimizer.step()\n' +
                '\n' +
                '            if args.local_rank == 0:\n' +
                '                logger.log_stats(scheduler.iteration, scheduler.new_lr, skip_losses=True)\n' +
                '\n' +
                '            logger.iter_toc()\n' +
                '            logger.iter_tic()\n' +
                '            logger.data_tic()\n' +
                '\n' +
                '            inputs, targets = prefetcher.next()\n' +
                '        return None\n' +
                '\n' +
                '    for epoch in range(cur_epoch, cfg.SOLVER.MAX_EPOCHS + 1):\n' +
                '        train_sampler.set_epoch(epoch)\n' +
                '\n' +
                '        logging_rank(\'Epoch {} is starting !\'.format(epoch), distributed=args.distributed, local_rank=args.local_rank)\n' +
                '        train(model, criterion, train_loader, optimizer, scheduler, logger)\n' +
                '```\n' +
                '\n' +
                '在训练一个全数据迭代结束时会自动保存模型，以便由于意外而终止的进程能够及时恢复，是否从中断的地方继续训练可以通过`cfg.TRAIN.AOTU_RESUME`进行设置，默认为True。\n' +
                '\n' +
                '```Python\n' +
                '    # Train and test\n' +
                '    logging_rank(\'Training starts !\', distributed=args.distributed, local_rank=args.local_rank)\n' +
                '    cur_epoch = scheduler.iteration // cfg.TRAIN.ITER_PER_EPOCH + 1\n' +
                '    for epoch in range(cur_epoch, cfg.SOLVER.MAX_EPOCHS + 1):\n' +
                '        train_sampler.set_epoch(epoch)\n' +
                '\n' +
                '        logging_rank(\'Epoch {} is starting !\'.format(epoch), distributed=args.distributed, local_rank=args.local_rank)\n' +
                '        train(model, criterion, train_loader, optimizer, scheduler, logger)\n' +
                '\n' +
                '        # Save model\n' +
                '        if args.local_rank == 0:\n' +
                '            snap_flag = cfg.SOLVER.SNAPSHOT_EPOCHS > 0 and epoch % cfg.SOLVER.SNAPSHOT_EPOCHS == 0\n' +
                '            checkpointer.save(model, optimizer, scheduler, copy_latest=snap_flag, infix=\'epoch\')\n' +
                '\n' +
                '    logging_rank(\'Training done !\', distributed=args.distributed, local_rank=args.local_rank)\n' +
                '```\n' +
                '在训练过程中每经过20次迭代之后会在控制台输出日志如下，输出日志的周期通过`cfg.DISPLAY_ITER`进行设置。\n' +
                '\n'
                },
        {shell:'```\n' +
                '    [Training][resnet50_1x64d.yaml][epoch: 1/90][iter: 20/5005][lr: 0.010072][eta: 2 days, 11:32:12]\n' +
                '          total_loss: 6.907886 (6.905636), iter_time: 0.3186 (0.4758), data_time: 0.0101 (0.0094)\n' +
                '          acc1: 0.0000 (0.3125), acc5: 0.0000 (0.7031)\n' +
                '    [Training][resnet50_1x64d.yaml][epoch: 1/90][iter: 40/5005][lr: 0.010144][eta: 2 days, 1:47:49]\n' +
                '          total_loss: 6.906398 (6.901059), iter_time: 0.3179 (0.3980), data_time: 0.0100 (0.0096)\n' +
                '          acc1: 0.0000 (0.0781), acc5: 0.0000 (0.7031)\n' +
                '    [Training][resnet50_1x64d.yaml][epoch: 1/90][iter: 60/5005][lr: 0.010216][eta: 1 day, 22:27:54]\n' +
                '          total_loss: 6.865390 (6.871011), iter_time: 0.3216 (0.3714), data_time: 0.0099 (0.0096)\n' +
                '          acc1: 0.0000 (0.5469), acc5: 0.0000 (1.3281)\n' +
                '    [Training][resnet50_1x64d.yaml][epoch: 1/90][iter: 80/5005][lr: 0.010288][eta: 1 day, 21:33:19]\n' +
                '          total_loss: 6.851057 (6.854321), iter_time: 0.3203 (0.3641), data_time: 0.0100 (0.0121)\n' +
                '          acc1: 0.0000 (0.6250), acc5: 1.5625 (1.7188)\n' +
                '          ......\n' +
                '```\n'},
        {text:'测试'},
        {text:'使用[CheckPointer](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%9E%B6%E6%9E%84/component-collects/%E6%A8%A1%E5%9E%8B%E5%8A%A0%E8%BD%BD%E5%92%8C%E4%BF%9D%E5%AD%98.md)组件，我们将最后一次训练的模型以及效果最好的模型进行保存，在测试脚本`$Pet/tools/cls/test_net.py`加载，进行测试，在`resnet50_1x64d.yaml`中设置`Test:Weights:\\\'$Pet/ckpts/.../best_model.pth\\\'`,其余流程中所涉及的参数与训练阶段相差无几，这里不再赘述。测试脚本中所涉及的其他组件请参阅[component-collects](https://github.com/BUPT-PRIV/Pet-DOC/tree/master/%E6%9E%B6%E6%9E%84/component-collects)部分，这里不作详细说明。'},
        {text:'用法示例：\n'},
        {shell:'```\n' +
                '    cd $Pet\n' +
                '\n' +
                '    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/cls/test_net.py \n' +
                '    --cfg /$Pet/cfgs/cls/imagenet/resnet/resnet50_1x64d.yaml\n' +
                '```\n'},
        {text:'测试结果：\n'},

        {shell:'```\n' +
                '    INFO:pet.utils.misc:[Testing][range:1-250 of 250][249/250][120.903s = 120.747s + 0.154s + 0.002s][eta: 0:02:00][acc1:\n' +
                '    77.48% | acc5: 93.75%]\n' +
                '    INFO:pet.utils.misc:[Testing][range:1-250 of 250][250/250][121.293s = 121.138s + 0.154s + 0.002s][eta: 0:00:00][acc1:\n' +
                '    77.44% | acc5: 93.75%]\n' +
                '    INFO:pet.utils.misc:val_top1: 77.4360% | val_top5: 93.7520%\n' +
                '```\n'},
        {part_title:'参考文献'},
        {text:'\\[1\\] Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun. Deep Residual Learning for Image Recognition. CVPR 2016.\n'},
        {text:'\\[2\\] Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun. Identity Mappings in Deep Residual Networks. ECCV 2016.\n'},
        {text:'\\[3\\] Saining Xie and Ross Girshick and Piotr Dolla ́r and Zhuowen Tu and Kaiming He. Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017.\n'}
    ]
}