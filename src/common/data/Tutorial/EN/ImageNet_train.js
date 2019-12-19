export let ImageNet_train = {
    key: 'ImageNet_train',
    dataSource: [
        {title:'Train Classfier on ImageNet'},
        {text:'This tutorial will introduce the basic process of training and testing classification networks on ImageNet datasets using Pet. You will learn how to train classification models with components provided by Pet. In the meantime, we only make component calls, and you can refer to the [component-collects](https://github.com/BUPT-PRIV/Pet-DOC/tree/master/%E6%9E%B6%9E%84/component-collects) section for details of component implementation.\n'},
        {text:'If you are already very proficient in image classification, you can start training directly with the [$Pet/tools/cls/train_net.py](https://github.com/BUPT-PRIV/Pet-dev/blob/e11ef696c92ea5e4cf30609fb67420a262c911ca/tools/cls/train_net.py) script.\n'},
        {text:'Training command：\n'},
        {shell:'```\n' +
                '    cd $Pet\n' +
                '\n' +
                '    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/cls/train_net.py \n' +
                '    --cfg /$Pet/cfgs/cls/imagenet/resnet/resnet50a.yaml\n' +
                '```\n'},
        {text:'Before starting the training process, Pet needs to be specified a `yaml` file, which contains all the adjustable hyper parameters used in during training, including learning rate, number of GPUs, network structure, total data iteration number and so on. Here we take `$Pet/cfgs/cls/imagenet/resnet/resnet50_1x64d.yaml` as an example, explaining the key configurations and training steps needed in the training process. Detailed parameters can be found in [$Pet/cfgs/cls/imagenet/resnet/resnet50a.yaml](https://github.com/BUPT-PRIV/Pet-dev/blob/master/cfgs/cls/imagenet/resnet/resnet50a.yaml).\n'},
        {part_title:'Load Data'},
        {text:'Make sure that ImageNet dataset is stored on your disk. The folder structure is as follows：\n' +
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
                'The ImageNet dataset is divided into two parts: the training subset and the validation subset.：\n' +
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
                'Pet uses the data processing class `ImageFolder` as the dataloader, `ImageFolder` read and create a dictionary for the subfolders under the parent folder, the subfolders\' name are corresponded to the category label, The folder directory is set in `DATASET: DATA_ROOT` in the yaml file. In order to achieve the effect of data augmentation and reduce the phenomenon of over-fitting, we use PyTorch\'s `transforms` function to preprocess the img data, and use a certain probability to flip the picture in the horizontal direction to enhance the data, and cut the picture randomly to `224 * 224` pixels, then argumented images are send it to the network as training batches, ` cfg.TRAIN.AUG` is a parameter that can be set for data enhancement in configuration files. In addition, we support another way of data augmentation, mixup. Read the PyTorch official code for the specific implementation of the `ImageFolder `class\n'},
        {
            block:{
                title:'mixup',
                children:[
                    {text:'Mix up is an unconventional data argumentation method, it is a simple data-independent argument principle, which constructs new training samples and labels in a linear interpolation way.\n' +
                            '\n' +
                            'The formulas of mixup：\n'},
                    {katex:'\\tilde{X} = \\partial \\times X _{i} + (1 - \\partial) \\times X _{j}'},
                    {katex:'\\tilde{Y} = \\partial \\times Y _{i} + (1 - \\partial) \\times Y _{j}'},
                    {text:'Effects are as follows：\n'},
                    {
                        table:{
                            titles:['Model','data augmentation ','epochs','top-1 error','top-5 error'],
                            data:[
                                ['resnet50','ERM',90,23.5,' - '],['resnet50','mixup α=0.2',90,23.3,' - '],
                                ['resnet50','ERM',200,23.6,7.0],['resnet50','mixup α=0.2',200,22.1,6.1]
                            ]
                        }
                    },
                    {text:'A simple understanding of mixup is to randomly extracting two images from the processed batch data and fusing at a certain ratio, the visual effect is like a picture is displayed with a certain transparency on another image, the transparency is the adjustable parameter of mixup method. More specific effects, detailed principles and experimental results please see [mixup](https://arxiv.org/abs/1710.09412v2).\n'}
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
                'We use the `torch.utils.data.DataLoader` class that comes with PyTorch to encapsulate the data. `torch.utils.data.DataLoader` divides and distributes the data prepared in the previous step, and the data can be shuffled if necessary. Please read the PyTorch official code for the specific implementation of the `torch.utils.data.DataLoader` class.\n' +
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
                'The processing and encapsulation of the validation data set is the same as that of the training data set, which is not explained in detail here.\n'},
        {part_title: 'Network Structure'},
        {text:'At the beginning of the training script, the `models` module under Pet is introduced to generate the backbone. The network structure name is set in the `MODEL:CONV_BODY` under the yaml file.\n' +
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
                title:'Deep Residual Neural Network',
                children:[
                    {text:'In the network structure bellow, the core of ResNet lies in the "shortcut connection", which is to add the input of a layer directly to one or several layers to avoid the gradient explosion during back propogation. Each block containing a shortcut connection is called a residual block.\n'},
                    {img:'resnet'},
                    {text:'According to the paper, there is a special case in these residual blocks, which is bottleneck. According to the network structure printed below or the structure on the right side of the above figuret, here is a Bottleneck in the first parameter of Sequential, which contains the convolution kernel of `1*1`, `3*3`, `1*1`, so that the convolution kernel with a convolution kernel size of "small-large-small" is the bottleneck structure. bottleneck structure is proposed in ResNet-v1\n' +
                            ', the use of the `1*1` convolution kernel adds more nonlinearity to the CNN, because two ReLU activation functions are introduced meanwhile, while being able to change the the number of channels with unchanged feature map size. \n'},
                    {text:'Another advantage of bottleneck is the ability to reduce the amount of parameters with same calculation cost. Let\'s take a calculation, the ordinary residual block parameter quantity is: (3x3x256x256)x2=1179648, the parameter amount of bottleneck is: 1x1x256x64+3x3x64x64+1x1x64x256=69632, the advantage is obvious. In resnet-v2, the experimental results show that the best order is BN-ReLU-conv order, which is also the order that Pet uses. \n'},
                    {text:'After using bottleneck to reduce the amount of parameters, the network can also be optimized from calculation cost. The group convolution introduced by ResNeXt can achieve better optimization. As shown in the figure below, we calculate the calculation of the group convolution and the ordinary convolution: with 28x28x28 input, 3x3x28x64 convolution kernel, divided into 2 groups as an example, the calculation of ordinary convolution is: 3x3x28x(28-3+1)x(28-3+1)x64=10902528, the calculation amount of the group convolution is: 3x3x14x(28-3+1)x(28-3+1)x32x2=5451264, it can be seen The amount of calculation for packet convolution is less than that of ordinary convolution.\n'},
                    {text:'More detailed structure introduction, principles and experimental results please refer to [resnet-v1](https://arxiv.org/abs/1512.03385v1)\\[1\\]、[resnet-v2](https://arxiv.org/abs/1603.05027v3)\\[2\\]、[resnext](https://arxiv.org/abs/1611.05431v2)\\[3\\]\n'
                            },
                    {img:'ResNeXt'},
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
                title:'Model parameters and calculation amount',
                children:[
                    {text:'Excepy for network structures, we need to know the amount of parameters and calculation of CNN. The amount of the parameter mainly relates to the size of the memory. Generally, our parameter are single-precision floating-point number, that is, float32, which occupies 4 bit of RAM, and the amount of computation we represent in FLOPs, represent the time required for network calculations, so that the length of time required to train the model can be estimated.\n'},
                    {text:'Formula for calculating the amount of parameters:\n'},
                    {katex:'params _{conv}= K _{h} \\times K _{w} \\times C _{in} \\times C _{out}+C _{out}'},
                    // {katex:'(K _{h},K _{w})表示卷积核的高和宽,(C _{in},C _{out})表示输入通道以及输出通道数量)'},
                    {katex:'params _{fc} = U _{in} \\times U _{out} + U _{out}'},
                    {text:'Formula for calculating FLOPs:\n'},
                    {katex:'FLOPs = params \\times H  \\times W'},
                    {text:'Example of LeNet：\n'},
                    {
                        table:{
                            titles:['CNN layer','Input','Kernel','Output','Parameters','Memory','FLOPs'],
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
        {part_title:'Training'},
        {text:'After the data loading, we start to train. The parameters can be set in the `resnet50_1x64d.yaml` file, such as batch data size, full data iteration, learning rate, optimizer, etc. Here we will use the standard training strategy for network training, that is, the learning rate is 0.1, the optimizer is SGD, the batch data size is 256 (64 pictures per GPU), the momentum is 0.9, with `SGD+Momentum` optimization method. See the other [system-components](https://github.com/BUPT-PRIV/Pet-DOC/tree/master/%E6%9E%B6%E6%9E%84/component-collects) involved in the training script.\n'},
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
                '`Checkpointer` will automatically save checkpoints after each epoch, so that the training process can be reversed in time, in case the training terminates for any resaon. you can set `cfg.TRAIN.AOTU_RESUME` in config system for resume training, the degault is `True`.\n' +
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
                'For every 20 iterations, the log will be output in the console as follows, The period of the output log is set by `cfg.DISPLAY_ITER`.\n' +
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
        {text:'Tesing'},
        {text:'[CheckPointer](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%9E%B6%E6%9E%84/component-collects/%E6%A8%A1%E5%9E%8B%E5%8A%A0%E8%BD%BD%E5%92%8C%E4%BF%9D%E5%AD%98.md) saves the last trained model and the best-performing model, load it in the test script `$Pet/tools/cls/test_net.py`, and set `Test.Weights` in yaml file as  `$Pet/ckpts/cls/imagenet/resnet/resnet50a/best_model.pth\'`. Please refer to [Document](https://github.com/BUPT-PRIV/Pet-DOC/tree/master/%E6%9E%B6%E6%9E%84/component-collects) for other components involved in the test script.'},
        {text:'Testing command：：\n'},
        {shell:'```\n' +
                '    cd $Pet\n' +
                '\n' +
                '    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/cls/test_net.py \n' +
                '    --cfg /$Pet/cfgs/cls/imagenet/resnet/resnet50_1x64d.yaml\n' +
                '```\n'},
        {text:'Testing result：\n'},

        {shell:'```\n' +
                '    INFO:pet.utils.misc:[Testing][range:1-250 of 250][249/250][120.903s = 120.747s + 0.154s + 0.002s][eta: 0:02:00][acc1:\n' +
                '    77.48% | acc5: 93.75%]\n' +
                '    INFO:pet.utils.misc:[Testing][range:1-250 of 250][250/250][121.293s = 121.138s + 0.154s + 0.002s][eta: 0:00:00][acc1:\n' +
                '    77.44% | acc5: 93.75%]\n' +
                '    INFO:pet.utils.misc:val_top1: 77.4360% | val_top5: 93.7520%\n' +
                '```\n'},
        {part_title:'Reference'},
        {text:'\\[1\\] Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun. Deep Residual Learning for Image Recognition. CVPR 2016.\n'},
        {text:'\\[2\\] Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun. Identity Mappings in Deep Residual Networks. ECCV 2016.\n'},
        {text:'\\[3\\] Saining Xie and Ross Girshick and Piotr Dolla ́r and Zhuowen Tu and Kaiming He. Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017.\n'}
    ]
}