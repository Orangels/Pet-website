export let MSCOCO_SSD = {
    key: 'MSCOCO_SSD',
    dataSource: [
        {title:'Train SSD Model on MSCOCO2017 Dataset'},
        {text:'This tutorial will introduce the main steps of using Pet to train and test SSD model for detection. Here we will show you how to construct SSD model by combining various functional modules provided by Pet. We will only explain the component invocation, part of the implementation details please refer to the corresponding parts of the system components. Before reading this tutorial, we strongly recommend that you read the original paper [SSD](https://arxiv.org/abs/1512.02325)\\[1\\], [FSSD](https://arxiv.org/abs/1712.00960v1)[2] to learn more about the detials of SSD algorithm.\n'},
        {
          note:[
              {text:'First please refer to [prepare data on MSCOCO](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87.md#mscoco%E6%95%B0%E6%8D%AE%E9%9B%86) and download MSCOCO2017 dataset on on your disk。\n'}
          ]
        },
        {text:'If you have rich experience in object detection, you can also run [$Pet/tools/ssd/train_net.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/tools/ssd/train_net.py) script directly in Pet to start training your SSD model.\n'},
        {text:'Examples'},
        {ul:'* Train a SSD model on `coco_2017_train` with 8 GPUs, use VGG16 to feature extraction:'},
        {shell:'```\n' +
                '      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch tools/ssd/train_net.py \n' +
                '      --cfg cfgs/ssd/mscoco/ssd_VGG16_300x300_1x.yaml\n' +
                '```\n'},
        {text:'Test a SSD model on `coco_2017_val` with 8 GPUs:'},
        {shell:'```\n' +
                '      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/ssd/test_net.py --cfg ckpts/ssd/mscoco/ssd_VGG16_300x300_1x.yaml\n' +
                '```\n'},
        {text:'Before any operations to train and test a model, it is necessary to select a specified `yaml` file to specify the requirements and settings for data set, model structure, optimization strategy and other important parameters during training. This tutorial will explain the key configurations needed in the training process with $Pet/cfgs/ssd/mscoco/ssd_VGG16_300x300_1x.yaml. This configuration will guide all steps and details of Mask R-CNN model training and testing. See [ssd_VGG16_300x300_1x.yaml](https://github.com/BUPT-PRIV/Pet-dev/blob/master/cfgs/ssd/mscoco/ssd_VGG16_300x300_1x.yaml) for all parameters.\n'},
        {part_title:'Data loader'},
        {text:'Make sure that the MSCOCO2017 dataset is stored on your hard disk and the file structure of the MSCOCO2017 dataset is organized according to the file structure in Data Preparation (https://github.com/BUPT-PRIV/Pet-DOC/tree/master/%E6%9E%B6%E6%9E%84/component-collects/%E6%95%B0%E6%8D%AE%E5%88%B6%E5%A4%87). Then we can start loading the `coco_2017_train` training set\n' +
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
                'Data argumentation in SSD can greatly improve network performance. The main methods used are horizontal flip, random clipping, color distortion, random acquisition of block domains, random scaling of images to obtain small target training samples, etc. The following figure shows a picture of MSCOCOCO2017 training set and its effect after SSD data enhancement.\n'},
        {img:'ssd_1527'},
        {text:'In the test stage, the data argumentation method is much simpler, only scale transformation and color jitter for the test image.\n'},
        {part_title: 'SSD structure'},
        {
            block:{
                title:'SSD network design',
                children:[
                    {text:'SSD, known as Single Shot MultiBox Detector, is a object detection algorithm proposed by Wei Liu on ECCV 2016 and one of the most important detection frameworks up to now. The main idea of SSD network design is to extract features hierarchically, and then regression and classification in turn. Because feature maps at different levels can represent different levels of semantic information, low-level feature maps can represent low-level semantic information (containing more details), improve the quality of semantic segmentation, and are suitable for small-scale target. High-level feature maps can represent high-level semantic information, smoothly segment the results, and are suitable for large-scale targets. Therefore, the SSD network proposed by the author can theoretically be suitable for target detection at different scales. The following diagram shows the network structure of SSD:'},
                    {img:'ssd结构'},
                    {text:'Present, target detection is divided into two main frameworks:\n'},
                    {ul:[
                        'Two stages：Represented by Faster RCNN, use RPN network to location the proposals, and then classifies and bounding box regression the proposals to complete the target classification.',
                            'Single stage：Represented by YOLO/SSD, completion of classification and bounding box regression in one stage.',
                        ]},
                    {text:'As a representative algorithm of single-stage target detection, SSD has the following main characteristics:'},
                    {ul:[
                        'The main idea is transforming target detection task into regression, and achieve target location and classification at one time.',
                            'Based on Anchor used in Faster RCNN，proposed Prior Box.',
                            'feature pyramid is added to predict the target on the feature map of different receptive fields.'
                        ]}
                ]
            }
        },
        {text:'In Pet, SSD use the `Generalized_SSD` model builder to build networks, as detailed in `Model Building`. We set the following parameters in the `yaml`file to construct feature extraction network in SSD network using configuration system:\n' +
                '\n' +
                '```\n' +
                '      MODEL:\n' +
                '        TYPE: generalized_ssd\n' +
                '        ANCHOR_ON: True\n' +
                '        NUM_CLASSES: 81\n' +
                '      BACKBONE:\n' +
                '        CONV_BODY: vgg16\n' +
                '```\n' +
                'In this tutorial, we use vgg16 as feature extraction network, whiech can be set in ``cfg.BACKBONE.CONV_BODY``, after constructing the feature extraction network, the network modules such as function network, task output and loss function are set by the following parameters:\n' +
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
                title:'Setting and Matching Strategy of Prior Box in SSD',
                children:[
                    {text:'In Yolo, each unit predicts multiple boundary boxes, but they are all relative to the unit itself (square), but the shape of the real target is changeable. Yolo needs to adapt the shape of the target in the training process. SSD refers to the method of anchor in Faster R-CNN. Each cell sets a Prior Box with different scale or aspect ratio. The bounding boxes of prediction are based on these Prior Box, which can reduce the training difficulty to a certain extent. In general, each unit will set up a number of Prior Boxes with different scales and aspect ratios. As shown in the figure, we can see that each unit uses four different Prior Boxes. In the picture, the Prior Box which are most suitable for the shape of the object are used to train.\n'},
                    {img:'SSD2'},
                    {text:'In train stage，groundtruth boxes and default boxes（Prior Box） are matched as following methed：\n'},
                    {ul:[
                        'First, look for default boxes with the largest overlap for each ground truth box, so that each groundtruth box corresponds to the only one default box.',
                            'If the overlap between the remaining default box and any groundtruth box is larger than a threshold, match is considered (SSD 300 sets the threshold to 0.5). Obviously the default box paired to GT is positive, and the default box without GT is negative'
                        ]},
                ]
            }
        },
        {part_title:'Train'},
        {text:'After completing data loading and model building, we need to choose the optimization strategy before training. In the case of batch size 64, set the initial learning rate 0.002, train 120 epoch, learning rate is reduced ten times at 80 and 110 epech.\n' +
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
                'During the training stage, the log will records the training information such as the number of iterations and the deviation values of the current network training after each iteration. The checkpoint component will saves the network model to the path set by `cfg.CKPT\' in the configuration system.\n'},
        {
            block:{
                title:'SSD train strategy',
                children:[
                    {ul:'Joint loss function'},
                    {text:'The SSD network performs boundary box regression and classification for each stage output feature graph, and designs a joint loss function as follows:'},
                    {katex:'L(x,c,l,g)=\\frac{1}{N}(L _{conf} (x,c)+\\alpha L _{loc}(x,l,g))'},
                    {text:'Lconf represent classification error，use softmax loss；Lloc represent regression error, use SmothL1 Loss.'},
                    {ul:'Hard Negative Mining'},
                    {text:'After matching strategy, a large number of negative samples and only a small number of positive samples will be obtained. This will lead to the imbalance of positive and negative samples, and the imbalance of positive and negative samples is an important reason for the low detection accuracy. So the strategy of Hard Negative Mining is adopted in the training process. All boxes are sorted according to Confidence Loss, and negative samples are taken three times as many as positive ones according to loss descending order, which not only ensures the difficulty of negative samples, also keeps the proportion of positive and negative samples within 1:3. Improve the accuracy by about 4%.\n'},
                    {ul:'Data Augmentation'},
                    {text:'Each mage is trained as a patch obtained by the following transformation.\n'},
                    {ul:[
                        'original image (no transformation)',
                            'Sampling a patch directly from the original image (i.e. without transformation) ensures that the minimum IoU between Ground Truth and the original image is 0.1, 0.3, 0.5, 0.7 or 0.9.',
                            'Completely random sampling of a patch'
                        ]},
                    {text:'At the same time, in the process of data enhancement, the proportion of the patch sampled to the size of the original image is between [0.1, 1], the length-width ratio of the patch sampled is between [0.5, 2]. When the center of the Ground Trubox in the patch sampled, the whole Ground Trubox is retained. Finally, each patch is resized to a fixed size and at a random level of 0.5 probability. The flips are eventually trained with these processed patches.'}
                ]
            }

        },
        {part_title:'Test'},
        {text:'After the training of SSD model, we use [$Pet/tools/ssd/test_net.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/tools/ssd/test_net.py) to evaluate the accuracy of the model on `coco_2017_val\'. You also need to use `Dataloader\' to load test data sets.\n'},
        {text:'By loading the model `Pet/ckpts/ssd/mscoco/ssd_VGG16_300x300_1x/model_latest.pth` after the maximum number of iterations of training, the following commands are executed to test the model.\n'},
        {shell:'```\n' +
                '      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/ssd/test_net.py \n' +
                '      ----cfg cfgs/ssd/mscoco/ssd_VGG16_300x300_1x.yaml\n' +
                '```\n'},
        {part_title:'Visualize results'},
        {text:'In Pet, SSD returns the category ID, confidence score and boundary box coordinates of each target. Visualize the inference result of a picture in MSCOCO2017_val as follows.\n'},
        {img:'ssd_test_062355'},
        {part_title:'Reference'},
        {text:'\\[1\\] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. ECCV 2016.\n'}
    ]
}