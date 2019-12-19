export let MSCOCO_RetinaNet = {
    key: 'MSCOCO_RetinaNet',
    dataSource: [
        {title:'Train RetinaNet model on MSCOCO2017 dataset'},
        {text:'This tutorial will introduce the main steps of using Pet to train and test RetinaNet model for detection. Here we will show you how to construct RetinaNet model by combining various functional modules provided by Pet. We will only explain the component invocation, part of the implementation details please refer to the corresponding parts of the system components. Before reading this tutorial, we strongly recommend that you read the original paper [RetinaNet](https://arxiv.org/pdf/1708.02002.pdf)\\[1\\] to learn more about the detials of RetinaNet algorithm.\n'},
        {
            note:[
                {text:'First please refer to [prepare data on MSCOCO](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87.md#mscoco%E6%95%B0%E6%8D%AE%E9%9B%86) and download MSCOCO2017 dataset on on your disk。\n'}
            ]
        },
        {text:'If you have rich experience in object detection, you can also run `$Pet/tools/rcnn/train_net.py` script directly in Pet to start training your RetinaNet model.\n'},
        {text:'Examples：\n'},
        {ul:'Train a RetinaNet model on `coco_2017_train` with 8 GPUs, use ResNet50 to feature extraction:\n'},
        {text:'```\n' +
                '      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch tools/rcnn/train_net.py \n' +
                '      --cfg cfgs/rcnn/mscoco/retinanet/retinanet_R-50-FPN-P5_1x.yaml\n' +
                '```\n'},
        {ul:'Test a RetinaNet model on `coco_2017_val` with 8 GPUs:'},
        {text:'```\n' +
                '      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/rcnn/test_net.py --cfg ckpts/rcnn/mscoco/retinanet/retinanet_R-50-FPN-P5_1x.yaml\n' +
                '```\n'},
        {text:'Before any operations to train and test a model, it is necessary to select a specified `yaml` file to specify the requirements and settings for dataset, model structure, optimization strategy and other important parameters during training. This tutorial will explain the key configurations needed in the training process with $Pet/cfgs/rcnn/mscoco/retinanet/retinanet_R-50-FPN-P5_1x.yaml. This configuration will guide all steps and details of Mask R-CNN model training and testing. See [[retinanet_R-50-FPN-P5_1x.yaml](https://github.com/BUPT-PRIV/Pet-dev/blob/master/cfgs/rcnn/mscoco/retinanet/retinanet_R-50-FPN-P5_1x.yaml) for all parameters.\n'},
        {part_title:'Data loader'},
        {text:'Make sure that the MSCOCO2017 dataset is stored on your hard disk and the file structure of the MSCOCO2017 dataset is organized according to the file structure in Data Preparation. Then we can start loading the `coco_2017_train` training set\n' +
                '\n' +
                '```Python\n' +
                '    # Create training dataset and loader\n' +
                '    datasets = build_dataset(cfg.TRAIN.DATASETS, is_train=True, local_rank=args.local_rank)\n' +
                '    train_loader = make_train_data_loader(datasets, is_distributed=args.distributed, start_iter=scheduler.iteration)\n' +
                '      cfg.TRAIN.ITER_PER_EPOCH = len(train_loader)\n' +
                '```\n'},
        {part_title: 'RetinaNet structure'},
        {
            block:{
                title:'RetinaNet network design',
                children:[
                    {text:'RetinaNet is a new contribution of FAIR team in object detection field in 2018. Essentially, it is the structure of backbone + FPN + 2xFCN subnetworks.\n' +
                            '\n' +
                            'General backbone networks can choose any effective feature extraction network such as VGG or RESNET series. Here the authors try resnet-50 and resnet-101 respectively. FPN is to enhance the use of multi-scale features extracted from backbone, and thus get a more expressive feature map containing multi-scale target region information. Finally, on the set of feature maps output by FPN, two FCN subnetworks with the same structure but independent of each other are used to accomplish the task of classification and location regression without sharing parameters.\n'},
                    {img:'pic_11'},
                    {text:'Firstly, RetinaNet feature extraction is accomplished by ResNet+FPN. After feature extraction of input image, the Pyramid of P3~P7 feature graph can be obtained. After feature pyramid is obtained, two FCN subnetworks (classification network+detection box position regression) are used for each layer of feature pyramid.\n'},
                    {ul:[
                            'Similar to RPN networks, RetinaNet uses anchors to generate proposals. Each layer of the feature pyramid corresponds to an anchor area. In order to generate more intensive anchor coverage, three area scales and three different aspect ratios are added.',
                            'The classification network of the original RPN network only distinguishes the foreground from the background. In RetinaNet, the number of categories is K.',
                            'Parallel to the classification branch, RetinaNet connects a box regression branch to the feature map of each FPN output layer, which is essentially a FCN network. It predicts the offset of anchor and its corresponding ground truth position, i.e. for each anchor, it returns a (x, y, w, h) four-dimensional vector.'
                        ]},

                ]
            }
        },
        {text:'In Pet, RetinaNet use the `Generalized_RCNN` model builder to build networks, as detailed in `Model Building`. We set the following parameters in the `yaml`file to construct feature extraction network in RetinaNet network using configuration system:\n' +
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
                '```\n' +
                '\n' +
                'In this tutorial, we use ResNet50 as feature extraction network, whiech can be set in ``cfg.BACKBONE.CONV_BODY``, after constructing the feature extraction network, the network modules such as RetinaNet and FPN are set by the following parameters:\n' +
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
        {part_title:'Train'},
        {text:'After completing data loading and model building, we need to choose the optimization strategy before training.\n' +
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
                'During the training stage, the log will records the training information such as the number of iterations and the deviation values of the current network training after each iteration. The checkpoint component will saves the network model to the path set by `cfg.CKPT` in the configuration system.\n'},
        {
            block:{
                title:'Focal Loss',
                children:[
                    {text:'In common single-stage detectors, convolution networks generate dense target candidate regions after feature maps are obtained, and only a few of these large candidate regions are real targets, which results in the imbalance between positive and negative training samples. It often leads the loss to be dominated by negative samples, which account for the vast majority but contain little information. The key information provided by few positive samples can not play a normal guiding role in loss.\n' +
                            'At the same time, the candidate regions of the two-stage method are much smaller than those of the single-stage detector, so there is no serious class imbalance problem. Commonly used methods to solve this problem are negative sample mining or other more complex sampling methods for filtering negative samples so as to maintain a certain ratio of positive and negative samples.\n' +
                            'So, [Focal Loss](https://arxiv.org/pdf/1708.02002.pdf)\\[1\\] is proposed to correct the final loss.\n' +
                            '\n' +
                            'Focal Loss is a cross-entropy loss of dynamic scaling. In short, through a dynamic scaling factor, it can dynamically reduce the weight of easily distinguished samples in the training process, so that the focus of loss can be quickly focused on those difficult to distinguish samples. It is an improvement based on Cross-Entropy (CE) loss. It adds an adjusting factor to assign different weights to different samples.\n'},
                    {ul:'Cross Entropy(CE) Loss'},
                    {text:'Firstly, we will introduce CE loss:\n'},
                    {katex:'CE(p,y) = \n' +
                            '\\begin{cases}\n' +
                            '-log(p) & if\\ y = 1 \\\\\\\n' +
                            '-log(1-p) & otherwise\n' +
                            '\\end{cases}\n'},
                    {text:'In the CE loss,y is the category of ground truth, and P is the probability value of y = 1 predicted by the model. For convenience of expression, we define pt as follow:\n'},
                    {katex:'p _{t} = \n' +
                            '\\begin{cases}\n' +
                            'p & if\\ y = 1 \\\\\\\n' +
                            '1-p & otherwise\n' +
                            '\\end{cases}\n'},
                    {ul:'Balanced CE loss'},
                    {text:'Usually, the common method to solve the class imbalance is to introduce the weight α:\n'},
                    {katex:'CE(p _{t}) = -\\alpha _{t}log(p _{t})'},
                    {text:'When the class label is 1, the weight factor is α, and when the class label is - 1, the weight factor is 1 - α. Similarly, for convenience, the weight factor is expressed by αt.\n'},
                    {ul:'Focal loss'},
                    {text:'In fact, Focal loss can be seen as a special form of Balanced CE loss, which only instantiates the weight alpha t into  （1-pt）^ γ, so focal loss is defined as:\n'},
                    {katex:'FL(p _{t}) = -(1 - p _{t}) ^{\\gamma}log (p _{t})'},
                    {text:'Among them,  （1-pt）^ γ is the regulatory factor, and γ is a superparameter. When a sample is misclassified, pt is relatively small,（1-pt）^ γ is relatively large, that is, the weight is relatively large, and vice versa. Therefore, the loss of simple samples will be reduced, while the loss of difficult samples will be enlarged.\n' +
                            'In addition, when γ takes different values, the following figure can be obtained:\n'},
                    {img:'pic_16'},
                ]
            }
        },
        {part_title:'Test'},
        {text:'After the training of model, we use [$Pet/tools/rcnn/test_net.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/tools/rcnn/test_net.py) to evaluate the accuracy of the model on `coco_2017_val`. You also need to use `Dataloader` to load test data sets.\n' +
                '\n' +
                'By loading the model `Pet/ckpts/rcnn/mscoco/retinanet/retinanet_R-50-FPN-P5_1x/model_latest.pth` after the maximum number of iterations of training, the following commands are executed to test the model.\n' +
                '\n' +
                '```\n' +
                '      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/rcnn/test_net.py 、\n' +
                '      ----cfg cfgs/rcnn/mscoco/retinanet/retinanet_R-50-FPN-P5_1x.yaml\n' +
                '```\n'},
        {part_title:'Visualize results'},
        {text:'In Pet, RetinaNet returns the category ID, confidence score and boundary box coordinates of each target. Visualize the inference result of a picture in MSCOCO2017_val as follows.\n'},
        {img:'pic_000000102331'},
        {part_title:'Reference'},
        {text:'\\[1\\] Lin T Y , Goyal P , Girshick R , et al. Focal Loss for Dense Object Detection[J]. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2017, PP(99):2999-3007.\n'}
    ]
}