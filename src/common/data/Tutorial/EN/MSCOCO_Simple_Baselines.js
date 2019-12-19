export let MSCOCO_Simple_Baselines = {
    key: 'MSCOCO_Simple_Baselines',
    dataSource: [
        {title:'Train Simple Baselines on MSCOCO2017'},
        {text:'This tutorial will introduce the main steps of using Pet to train and test Simple Baseline model for single human body pose estimation. Here we will show you how to construct SSD model by combining various functional modules provided by Pet. We will only explain the component invocation, part of the implementation details please refer to the corresponding parts of the system components. Before reading this tutorial, we strongly recommend that you read the original paper [Simple Baselines](https://arxiv.org/abs/1804.06208v2)\\[1\\] to learn more about the detials of Simple Baseline algorithm.\n'},
        {
            note:[
                {text:'For consistency of data source files, pose estimation task does not require data source file format. Using the same data format as RCNN task, click [here](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87.md#mscoco%E6%95%B0%E6%8D%AE%E9%9B%86) for details.\n'}
            ]
        },
        {text:'If you have rich experience in human body pose estimation, you can also run [$Pet/tools/pose/train_net.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/tools/pose/train_net.py) script directly in Pet to start training your Simple Baseline model.\n'},
        {text:'Examples：\n'},
        {ul:'Train a Simple Baseline model on `keypoints_coco_2017_train` with 4 GPUs:\n'},
        {shell:'```\n' +
                '  cd $Pet\n' +
                '\n' +
                '  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch tools/pose/train_net.py \n' +
                '  --cfg cfgs/pose/mscoco/simple_R-50c-D3K4C256_256x192_adam_1x.yaml\n' +
                '```\n'},
        {ul:'Test a Simple Baseline model on `keypoints_coco_2017_val` with 4 GPUs:'},
        {shell:'```\n' +
                '  cd $Pet\n' +
                '\n' +
                '  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/pose/test_net.py \n' +
                '  --cfg cfgs/pose/mscoco/simple_R-50c-D3K4C256_256x192_adam_1x.yaml\n' +
                '```\n'},
        {text:'This tutorial uses the settings in [Pet/cfgs/pose/mscoco/simple_R-50c-D3K4C256_256x192_adam_1x.yaml](https://github.com/BUPT-PRIV/Pet-dev/blob/master/cfgs/pose/mscoco/simple_R-50c-D3K4C256_256x192_adam_1x.yaml) to specify the configuration of data set, model structure and optimization strategy during training. This set of configuration will guide the whole steps and details of Simple Baselines model training and testing.\n'},
        {part_title:'Data loader'},
        {text:'Make sure that the MSCOCO2017 dataset and the keypoint annotation files have been stored on your hard disk according to the standard form. Then we can start loading the `keypoints_coco_2017_train` training set.\n' +
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
                'In human pose estimation task, each human instance is intercepted from a picture containing multiple human by [COCOInstanceDataset](), `train_loader ` outputs a batch of images consisting of multiple persona instances, each persona instance image is scaled to 192 * 256 pixels during training. Set heatmap size is 48 * 64 pixels in `cfg.POSE.HEATMAP_SIZE`, size is a quarter of the input image to generate training labels for keypoints.',
                'Simple Baselines also performs color jitter and rotation on each instance during data loading to enhance the generalization of the model. The original image and `train_loader` output image in dataset are as follows:'
            ]},
        {img:'pose_ori'},
        {img:'pose_ins_gaussian'},
        {text:'picture in Dataloader before and after transform',type:'center'},
        {
            block:{
                title:'Training labels for human pose estimation',
                children:[
                    {text:'emantic segmentation tasks need to predict the categories of all the pixels in the image. When training the semantic segmentation model, the training label loaded together with the image is a segmentation mask of the same size as the image. Attitude estimation task is similar to semantics segmentation task. The model needs to predict which pixels are the key points in the image, so a mask image bearing the key points is also needed as the training label in training.\n'},
                    {text:'If only a limited number of keypoints in an image are assigned to the mask label, the loss function will remain high during the training process. Therefore, in the training label of pose estimation task, a Gauss distribution is generated around each keypoint of human body, and the Gauss distribution of each keypoint is generated around each keypoint. As a label thermogram, the values of the Gauss thermograms for invisible keypoints are all zero.\n'}
                ]
            }
        },
        {text:'output：\n'},
        {shell:'```\n' +
                '  images：(1, (3, 256, 192))\n' +
                '  target：(1, (17, 64, 48))  # training heatmaps of keypoints\n' +
                '  target_weight：(1, 17, 1))\n' +
                '```\n'},
        {part_title:'Simple Baseline structure'},
        {text:'In Pet，Simple Baselines use `Generalized_CNN` to build networks, details in [model builder](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%9E%B6%E6%9E%84/component-collects/%E6%A8%A1%E5%9E%8B%E6%9E%84%E5%BB%BA/Model%20Building.md). We set the following parameters in the yaml file to construct feature extraction network in Simple Baseline network using configuration system:\n' },
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

        {text:'ResNet50 is selected as feature extraction network according to the settings of `cfg.BACKBONE.RESNET.LAYERS`, besides ResNet, Pet supports [MobileNet-v1](https://arxiv.org/abs/1704.04861v1)\\[2\\]、[EfficientNet](https://arxiv.org/abs/1905.11946v2)\\[3\\]、[HRNet](https://arxiv.org/abs/1902.09212)\\[4\\] as backbone.\n' +
                '\n' +
                'after constructing the feature extraction network, the network modules such as function network, task output and loss function are set by the following parameters:\n'},

        {yaml:'```\n' +
                'KEYPOINT:\n' +
                '  NUM_JOINTS: 17\n' +
                '  HEATMAP_SIZE: (48, 64)\n' +
                '```\n'},
        {ul:[
            'The key `POSE_HEAD` in dect `cfg.POSE` specifies the functional network, `simple_xdeconv_head` means 3 * ConvTranspose2d + BatchNormlization + ReLU.',
                'NUM_JOINTS` is used to determine the number of channels for task output, which depends on the number of human pose keypoints in coco dataset. The human pose keypoints in MSCOCO2017 dataset labels 17.',
            ]},
        {
            block:{
                title:'Bottom-up and Top-down multi-person pose estimation',
                children:[
                    {text:'At present, there are two main methods to realize multi-person pose estimation tasks. One is bottom-up. The bottom-up method predicts the connection between the keypoints and the keypoints of all tasks at the same time. The connection can be in the form of embedding vectors between the keypoints or the connection intensity field. (It\'s also a form of heatmap). Representational work such as Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1611.08050)\\[5\\], the corresponding open source project is known as [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).\n'},
                    {img:'bottom_up'},
                    {text:'The algorithm of OpenPost is to detect the different keypoints of all human bodies in the picture, and predict the correlation heatmap between any two keypoints on the same trunk. The number of the correlation heatmap is equal to the logarithm of the adjacent keypoints of human body. Then, by calculating the integral of the associated thermograms corresponding to the two keypoints, the different keypoints belonging to the same human body are connected.'},
                    {img:'apf'},
                    {text:'Simple Baseline are often used in the second phase of the top-down method. In the first stage, a detector is used to detect the bounding frames of multiple human bodies in a picture, then the human body parts are extracted from the original image, and then the keypoint detection network is used to estimate the pose of a single human body.Under the premise of single human detection frame, the keypoint detection problem mainly focuses on network design and training strategies. Before Simple Baselines, [HourGlass] (https://arxiv.org/abs/1603.06937)\\[6\\] and [CPN] (https://arxiv.org/abs/1711.07319)\\[7\\] respectively made some contributions to the network structure and keypoint occlusion problem of keypoint detection problem.\n'},
                    {img:'two_stage_top_down'},
                    {text:'Bottom-up pose estimation', type:'center'},
                    {text:'The top-down method also has an end-to-end solution: [Mask R-CNN](https://arxiv.org/abs/1703.06870) \\[8\\]. While detecting the branch prediction bounding box, a keypoint detection branch is parallel to predict the keypoints of human body instances directly within the bounding box. Different from the two-stage method, the input of the keypoint detection network is the feature map entering RPN network. Lower resolution and less functional feature maps often result in the lower detection accuracy of the keypoints than the two-stage top-down method such as simple Baselines.\n'}
                ]
            }
        },

        {part_title:'Train'},
        {text:'After completing data loading and model building, we need to choose the optimization strategy before training. In the case of batch size 256, set the initial learning rate 0.002, train 140 epoch, learning rate is reduced ten times at 90 and 110 epech.\n' +
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
        {text:'During the training stage, the log will records the training information such as the number of iterations and the deviation values of the current network training after each iteration. The checkpoint component will saves the network model to the path set by `cfg.CKPT` in the configuration system.'},
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
        {part_title:'Test'},
        {text:'After the training of Simple Baselines model, we use `et/tools/pose/test_net.py` to evaluate the accuracy of the model on `keypoints_coco_2017_val`. You also need to use `Dataloader` to load and scale the test data sets.\n'},
        {text:'Load the model `Pet/ckpts/pose/mscoco/simple_R-50c-D3K4C256_256x192_adam_1x/model_latest.pth`. Execute the following command to test the model, and the test log will also be logged by `Logger\'\n'},
        {shell:'```\n' +
                '  cd $Pet\n' +
                '\n' +
                '  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/pose/test_net.py 、\n' +
                '  --cfg cfgs/pose/mscoco/simple_R-50c-D3K4C256_256x192_adam_1x.yaml\n' +
                '```\n' +
                '\n' +
                'output：\n' +
                '\n' +
                '```\n' +
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
                title:'Evaluation in pose estimation',
                children:[
                    {text:'The evaluation of pose estimation are Average Precision (AP) and Average Recall (AR) and other variants of them. The calculation of average accuracy and average recall depends on Object Keypoints Similarity (OKS).\n'},
                    {text:'In the task of object detection, IoU is used to judge whether an object in a picture is detected by the algorithm; in the task of semantic segmentation, the IoU of object segmentation mask and label mask is used to judge the quality of the segmentation mask of the output of the segmentation model.\n'},
                    {text:'In pose estimation task, object keypoint similarity is used to evaluate the similarity between the predicted keypoints and the annotated keypoints. When OKS is larger than a certain threshold, it can be considered that the keypoints of the person instance are correctly predicted by the algorithm. The commonly used OKS thresholds are 0.5 and 0.75. The calculation formula of OKS is as follows:\n'},
                    {text:'```\n' +
                            '  OKS \n' +
                            '```\n'},
                    {katex:'OKS= \\sum _{i}[exp(-d _{i}^{2}/2s^{2}K _{i}^{2})\\delta(v _{i}>0)] / \\sum _{i}[\\delta(v _{i}>0)]'},
                    {img:'评价指标'},
                    {text:'In addition to OKS, there are other indicators used to evaluate keypoint detection, such as Percentage of Correct Keypoints (PCK), MSCOCO data sets using OKS as evaluation criteria, FLIC, LSP, MPII data sets using PCK.'},
                ]
            }
        },
        {part_title:'Visualize results'},
        {text:'In Pet, Simple Baselines returns 17 keypoints of each human instance and links them in the order of annotation of  MSCOCO2017. The reasoning results of a picture in the verification set are visualized as follows. \n'},
        {img:'demo_0062355_k'},
        {part_title:'Reference'},
        {text:'\\[1\\] Bin Xiao, Haiping Wu, Yichen Wei. Simple Baselines for Human Pose Estimation and Tracking. ECCV 2018.\n'},
        {text:'\\[2\\] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. CVPR 2017.\n'},
        {text:'\\[3\\] Mingxing Tan, Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. arXiv:1905.11946.\n'},
        {text:'\\[4\\] Vincent Dumoulin, Francesco Visin. A guide to convolution arithmetic for deep learning. arXiv:1603.07285.\n'},
        {text:'\\[5\\] Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh. Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. CVPR 2017.\n'},
        {text:'\\[6\\] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked Hourglass Networks for Human Pose Estimation. CVPR 2016.\n'},
        {text:'\\[7\\] Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang Zhang, Gang Yu, Jian Sun. Cascaded Pyramid Network for Multi-Person Pose Estimation. CVPR 2018.\n'},
        {text:'\\[8\\] Kaiming He, Georgia Gkioxari, Piotr Dolla ́r, Ross Girshick. Mask R-CNN. ICCV 2017.\n'},
    ]
}