export let model_construction_data = {
    key:'model_construction',
    dataSource: [
        {title:'Model Building'},
        {text:'Network structure is one of the core contents of deep learning research. For different computer vision tasks, Pet uses the following python classes to build the corresponding convolutional neural network:\n'},
        {
            ul:[
                'Generalized_CNN: image classification(cls), pose estimation(pose), semantic segmentation(semseg).',
                'Generalized_SSD: single stage object detection(SSD).',
                'Generalized_RCNN: region-based object detection and case analysis(RCNN).'
            ]
        },
        {text:'Network building tools such as `Generalized_CNN`, `Generalized_SSD`, and `Generalized_RCNN` are defined and used in the form of classes. Here we take the `Generalized_CNN` of pose estimation task as an example, to introduce the construction process of network in Pet.\n',className:'segmentation'},
        {part_title:'Generalized_CNN'},
        {text:'`Generalized_CNN`、`Generalized_SSD `、`Generalized_RCNN ` and other network construction tools are defined and used in the form of Python class. These model building tools have high consistency in structure and style. They can help you quickly build network structures of different visual tasks and develop computer vision algorithms. In the above series of tools, Pet clearly divides the convolutional neural network into several sub-network modules according to their functions, including feature extraction network, task-specific network, task output and loss function. Each module is connected with each other through input and output sequentially.\n'},
        {h3_title:'Initialization'},
        {text:'The model is constructed correspond to the specific Python class `Generalized_CNN`, `Generalized_CNN` constructes `Conv_Body`, `Pose_Head`, `Pose_Out`, `Pose_Loss`, and the main member function of `Generalized_CNN` is `forward`. Before understanding the function of `Generalized_CNN`, we first initialize the `Generalized_CNN`:\n' +
                '\n' +
                '```Python\n' +
                '    class Generalized_CNN(nn.Module):\n' +
                '        def __init__(self, with_loss=False):\n' +
                '            super().__init__()\n' +
                '            # assert cfg.MODEL.POSE_ON\n' +
                '            self.with_loss = with_loss\n' +
                '\n' +
                '            # Backbone for feature extraction\n' +
                '            conv_body = registry.BACKBONES[cfg.BACKBONE.CONV_BODY]\n' +
                '            self.Conv_Body = conv_body()\n' +
                '            self.dim_in = self.Conv_Body.dim_out\n' +
                '            self.spatial_scale = self.Conv_Body.spatial_scale\n' +
                '\n' +
                '            # Feature Pyramid Networks\n' +
                '            if cfg.MODEL.FPN_ON:\n' +
                '                self.Conv_Body_FPN = FPN.fpn(self.dim_in, self.spatial_scale)\n' +
                '                self.dim_in = [self.Conv_Body_FPN.dim_out]\n' +
                '                self.spatial_scale = self.Conv_Body_FPN.spatial_scale\n' +
                '            else:\n' +
                '                self.dim_in = self.dim_in[-1:]\n' +
                '                self.spatial_scale = self.spatial_scale[-1:]\n' +
                '\n' +
                '            # Pose Estimation Branch\n' +
                '            if cfg.MODEL.POSE_ON:\n' +
                '                pose_head = registry.POSE[cfg.POSE.POSE_HEAD]\n' +
                '                self.Pose_Head = pose_head(self.dim_in)\n' +
                '\n' +
                '                pose_output = registry.POSE[cfg.POSE.POSE_OUTPUT]\n' +
                '                self.Pose_Out = pose_output(self.Pose_Head.dim_out)\n' +
                '\n' +
                '                self.Pose_Loss = registry.POSE[\'pose_loss\']\n' +
                '\n' +
                '            # Parsing Estimation Branch\n' +
                '            if cfg.MODEL.PARSING_ON:\n' +
                '                parsing_head = registry.PARSING[cfg.PARSING.PARSING_HEAD]\n' +
                '                self.Parsing_Head = parsing_head(self.dim_in)\n' +
                '\n' +
                '                parsing_output = registry.PARSING[cfg.PARSING.PARSING_OUTPUT]\n' +
                '                self.Parsing_Out = parsing_output(self.Parsing_Head.dim_out)\n' +
                '\n' +
                '                self.Parsing_Loss = registry.PARSING[\'parsing_loss\']\n' +
                '\n' +
                '            # Mask Estimation Branch\n' +
                '            if cfg.MODEL.MASK_ON:\n' +
                '                mask_head = registry.MASK[cfg.MASK.MASK_HEAD]\n' +
                '                self.Mask_Head = mask_head(self.dim_in)\n' +
                '\n' +
                '                mask_output = registry.MASK[cfg.MASK.MASK_OUTPUT]\n' +
                '                self.Mask_Out = mask_output(self.Mask_Head.dim_out)\n' +
                '\n' +
                '                self.Mask_Loss = registry.MASK[\'mask_loss\']\n' +
                '\n' +
                '            # Semantic Segmentation Branch\n' +
                '            if cfg.MODEL.SEMSEG_ON:\n' +
                '                raise NotImplementedError\n' +
                '```\n'},
        {text:'For single-person pose estimation, instance segmentation, human body position analysis and densepose estimation tasks, Pet picks instances from the original image for correlation analysis. Therefore, in the initialization process of `Generalized_CNN`, a functional network . The construction of tast specific network, task output and loss function, for other tasks is added, as well as the construction of FPN. The network model of all tasks can be constructed by setting corresponding fields in the configuration file. The fields need to be exactly the same as the function names of the network modules provided in Pet. Different network modules can be invoked through the function of Pet modifier, turn to [$Pet/pet/pose/modeling/registry.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/pet/pose/modeling/registry.py)、[$Pet/pet/utils/registry.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/pet/utils/registry.py) for details.\n'},
        {text:'Set `cfg.BACKBONE.CONV_BODY` fields in configuration system, you can call such as [ResNet](https://arxiv.org/abs/1512.03385)\\[1\\]、[ResNeXt](https://arxiv.org/abs/1611.05431v2)\\[2\\]、[MobileNet-v1](https://arxiv.org/abs/1704.04861v1)\\[3\\]、[EfficientNet](https://arxiv.org/abs/1905.11946v2)\\[4\\] network structures as the feature extraction network.\n'},
        {text:'The `cfg.POSE.POSE_HEAD` field in the configuration system is `simple_xdeconv_head`, and can be used to construct task specific networks proposed in [Simple Baselines](https://arxiv.org/abs/1804.06208v2)\\[5\\].\n'},
        {
            ul:[
                'The `cfg.BACKBONE.CONV_BODY` and `cfg.POSE.POSE_HEAD` only determine the structure of the called network module, further settings can select detailed structure of the feature extraction network and the tesk specific network. For example, you can select the layers of the ResNet network by setting `cfg.BACKBONE.RESNET.LAYERS` to build ResNet18, ResNet50, ResNet101; or set the `cfg.BACKBONE.MV1.WIDEN_FACTOR` to determine the channel scaling factor when using mobilenet-v1. For more information, see the [configuration system](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%9E%B6%E6%9E%84/component-collects/%E9%85%8D%E7%BD%AE%E7%B3%BB%E7%BB%9F.md).',
            ]
        },
        {h3_title:'Feature Extraction Network'},
        {text:'Take ResNet50 as an example, after the `resnet` function in [$Pet/pet/pose/modeling/backbone/ResNet.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/pet/pose/modeling/backbone/ResNet.py) file is registered with the Python modifier, the `cfg.BACKBONE.CONV_BODY` field can be set to `resnet` in the configuration file, and then the `ResNet` class can be called to construct the feature extraction network.\n' +
                '\n' +
                '```Python\n' +
                '    import torch.nn as nn\n' +
                '    import pet.models.imagenet.resnet as res\n' +
                '    from pet.utils.net import make_norm\n' +
                '    from pet.pose.modeling import registry\n' +
                '\n' +
                '    # ---------------------------------------------------------------------------- #\n' +
                '    # ResNet Conv Body\n' +
                '    # ---------------------------------------------------------------------------- #\n' +
                '    @registry.BACKBONES.register("resnet")\n' +
                '    def resnet():\n' +
                '        model = ResNet(norm=get_norm())\n' +
                '        return model\n' +
                '```\n' },
        {text:'Feature extraction network is the main structure of classification network, also known as BACKBONE network, which is used to extract features in visual tasks. Pet strictly follows the relevant papers and makes the standard implementation for the basic network.\n'},
        {text:'Under the path of `$Pet/Pet/models`, excellent convolutional neural networks have been implemented, such as [VGG](https://arxiv.org/abs/1409.1556)\\[6\\], ResNet, construction-v3 \\[7\\], MobileNet, [ShuffleNet-v2](https://arxiv.org/pdf/1807.11164)[8]. Example code refer to the [$Pet/Pet/models/imagenet/resnet.py](https://github.com/BUPT-PRIV/Pet-dev/tree/master/pet/models/imagenet).\n'},
        {text:'In pose estimation task, `ResNet` inherits the ResNet parent class in [$Pet/Pet/models/imagenet/resnet.py](https://github.com/BUPT-PRIV/Pet-dev/tree/master/pet/models/imagenet) to initialize, the main purpose is to adjust the depth, width and detailed network structure of the first stage of ResNet according to different visual tasks. Other basic network structures are built in the same way.\n' +
                '\n' +
                '```Python\n' +
                '    class ResNet(nn.Module):\n' +
                '        def __init__(self, bottleneck=True, aligned=False, use_se=False, use_3x3x3stem=False, stride_3x3=False,\n' +
                '                     avg_down=False, base_width=64, layers=(3, 4, 6, 3), norm=\'bn\',\n' +
                '                     stage_with_conv=(\'normal\', \'normal\', \'normal\', \'normal\'), num_classes=1000):\n' +
                '            """ Constructor\n' +
                '            Args:\n' +
                '                layers: config of layers, e.g., (3, 4, 23, 3)\n' +
                '                num_classes: number of classes\n' +
                '            """\n' +
                '            super(ResNet, self).__init__()\n' +
                '            if aligned:\n' +
                '                block = AlignedBottleneck\n' +
                '            else:\n' +
                '                if bottleneck:\n' +
                '                    block = Bottleneck\n' +
                '                else:\n' +
                '                    block = BasicBlock\n' +
                '            self.expansion = block.expansion\n' +
                '            self.use_se = use_se\n' +
                '            self.stride_3x3 = stride_3x3\n' +
                '            self.avg_down = avg_down\n' +
                '            self.base_width = base_width\n' +
                '            self.norm = norm\n' +
                '\n' +
                '            self.inplanes = base_width  # default 64\n' +
                '            self.use_3x3x3stem = use_3x3x3stem\n' +
                '            if not self.use_3x3x3stem:\n' +
                '                self.conv1 = nn.Conv2d(3, base_width, 7, 2, 3, bias=False)\n' +
                '                self.bn1 = make_norm(base_width, norm=self.norm)\n' +
                '            else:\n' +
                '                self.conv1 = nn.Conv2d(3, base_width // 2, 3, 2, 1, bias=False)\n' +
                '                self.bn1 = make_norm(base_width // 2, norm=self.norm)\n' +
                '                self.conv2 = nn.Conv2d(base_width // 2, base_width // 2, 3, 1, 1, bias=False)\n' +
                '                self.bn2 = make_norm(base_width // 2, norm=self.norm)\n' +
                '                self.conv3 = nn.Conv2d(base_width // 2, base_width, 3, 1, 1, bias=False)\n' +
                '                self.bn3 = make_norm(base_width, norm=self.norm)\n' +
                '            self.relu = nn.ReLU(inplace=True)\n' +
                '            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)\n' +
                '\n' +
                '            self.layer1 = self._make_layer(block, base_width, layers[0], 1, conv=stage_with_conv[0])\n' +
                '            self.layer2 = self._make_layer(block, base_width * 2, layers[1], 2, conv=stage_with_conv[1])\n' +
                '            self.layer3 = self._make_layer(block, base_width * 4, layers[2], 2, conv=stage_with_conv[2])\n' +
                '            self.layer4 = self._make_layer(block, base_width * 8, layers[3], 2, conv=stage_with_conv[3])\n' +
                '\n' +
                '            self.avgpool = nn.AdaptiveAvgPool2d(1)\n' +
                '            self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)\n' +
                '\n' +
                '            self._init_weights()\n' +
                '```\n'},
        {
            note:[
                {text:'`Conv_Body` contains `avgpool` layer and full connected layer for classification, which will be deleted in other tasks, otherwise redundant parameters will be introduced in other visual task models.\n'}
            ]
        },
        {h3_title:'Task Specific Network Task Output Loss Function'},
        {text:'Task specific networks, task outputs, and loss functions are constructed in the same way as feature extraction networks and are guaranteed to be called by `Generalized_CNN` with Python modifier. Three network modules are implemented in [$Pet/pet/pose/modeling/pose_head/simple_head.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/pet/pose/modeling/pose_head/simple_head.py).\n'},
        {ul:'Task specific network:'},
        {text:'```Python\n' +
                '    from pet.utils.net import make_conv\n' +
                '    from pet.pose.modeling import registry\n' +
                '    from pet.pose.core.config import cfg\n' +
                '\n' +
                '    # ---------------------------------------------------------------------------- #\n' +
                '    # Simple heads\n' +
                '    # ---------------------------------------------------------------------------- #\n' +
                '    from pet.pose.modeling import registry\n' +
                '\n' +
                '    @registry.POSE.register("simple_xdeconv_head")\n' +
                '    class simple_xdeconv_head(nn.Module):\n' +
                '        def __init__(self, dim_in):\n' +
                '            super().__init__()\n' +
                '            self.dim_in = dim_in[-1]\n' +
                '\n' +
                '            hidden_dim = cfg.POSE.SIMPLE.DECONV_HEAD_DIM  # default: 256\n' +
                '            self.deconv_kernel = cfg.POSE.SIMPLE.DECONV_HEAD_KERNEL  # default: 4\n' +
                '            padding, output_padding = self._get_deconv_param()\n' +
                '            deconv_with_bias = cfg.POSE.SIMPLE.DECONV_WITH_BIAS\n' +
                '\n' +
                '            # deconv module\n' +
                '            deconv_list = []\n' +
                '            for _ in range(cfg.POSE.SIMPLE.NUM_DECONVS):\n' +
                '                deconv_list.extend([\n' +
                '                    nn.ConvTranspose2d(self.dim_in, hidden_dim, kernel_size=self.deconv_kernel, stride=2,\n' +
                '                                       padding=padding, output_padding=output_padding, bias=deconv_with_bias),\n' +
                '                    nn.BatchNorm2d(hidden_dim),\n' +
                '                    nn.ReLU(inplace=True)\n' +
                '                ])\n' +
                '                self.dim_in = hidden_dim\n' +
                '            self.deconv_module = nn.Sequential(*deconv_list)\n' +
                '            self.dim_out = self.dim_in\n' +
                '\n' +
                '            self._init_weights()\n' +
                '\n' +
                '        def _init_weights(self):\n' +
                '            # weight initialization\n' +
                '            for m in self.modules():\n' +
                '                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):\n' +
                '                    nn.init.kaiming_normal_(m.weight, mode=\'fan_out\')\n' +
                '                    if m.bias is not None:\n' +
                '                        nn.init.zeros_(m.bias)\n' +
                '                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):\n' +
                '                    nn.init.constant_(m.weight, 1)\n' +
                '                    nn.init.constant_(m.bias, 0)\n' +
                '\n' +
                '        def _get_deconv_param(self):\n' +
                '            if self.deconv_kernel == 4:\n' +
                '                return 1, 0\n' +
                '            elif self.deconv_kernel == 3:\n' +
                '                return 1, 1\n' +
                '            elif self.deconv_kernel == 2:\n' +
                '                return 0, 0\n' +
                '            else:\n' +
                '                raise ValueError(\'only support POSE.SIMPLE.DECONV_HEAD_KERNEL in [2, 3, 4]\')\n' +
                '\n' +
                '        def forward(self, x):\n' +
                '            c5_out = x[-1]\n' +
                '\n' +
                '            out = self.deconv_module(c5_out)\n' +
                '\n' +
                '            return out\n' +
                '```\n'},
        {ul:'Task output:'},
        {text:'```Python\n' +
                '    from pet.pose.modeling import registry\n' +
                '    from pet.pose.core.config import cfg\n' +
                '    \n' +
                '    # ---------------------------------------------------------------------------- #\n' +
                '    # Pose head outputs\n' +
                '    # ---------------------------------------------------------------------------- #\n' +
                '    @registry.POSE.register("conv1x1_outputs")\n' +
                '    class conv1x1_outputs(nn.Module):\n' +
                '        def __init__(self, dim_in):\n' +
                '            super().__init__()\n' +
                '            self.classify = nn.Conv2d(dim_in, cfg.POSE.NUM_JOINTS, kernel_size=1, stride=1, padding=0)\n' +
                '\n' +
                '        def forward(self, x):\n' +
                '            x = self.classify(x)\n' +
                '            return x\n' +
                '```\n'},
        {ul:'Loss function:'},
        {text:'```Python\n' +
                '    from pet.pose.modeling import registry\n' +
                '    from pet.pose.core.config import cfg\n' +
                '    from pet.pose.ops import JointsMSELoss\n' +
                '    \n' +
                '    # ---------------------------------------------------------------------------- #\n' +
                '    # Pose head loss\n' +
                '    # ---------------------------------------------------------------------------- #\n' +
                '    @registry.POSE.register("pose_loss")\n' +
                '    def pose_loss(outputs, targets, target_weight=None):\n' +
                '        device = torch.device(cfg.DEVICE)\n' +
                '        targets = targets.to(device)\n' +
                '        target_weight = target_weight.to(device)\n' +
                '        criterion = JointsMSELoss().to(device)\n' +
                '        loss = criterion(outputs, targets, target_weight)\n' +
                '\n' +
                '        return loss\n' +
                '```\n'},
        {h3_title:'forward'},
        {text:'Four network modules are connected by input and output, task specific network receives feature map output by feature extraction network, output feature maps of task specific network are recieved by task output, loss function receives prediction result of task output. In the training process, four network modules will be used, `self.with_loss` variable is set to `True`, which identifies the need for loss function. Only the former three parts are used in the test phase, and `self.with_loss` is set to `False` during testing. In the training and testing modes, loss and prediction results are returned respectively. The following is the forward calculation process of `Generalized_CNN\'.\n' +
                '\n' +
                '```Python\n' +
                '    def forward(self, x, targets=None, target_weight=None):\n' +
                '        return_dict = {}  # A dict to collect return variables\n' +
                '        blob_conv = self.Conv_Body(x)  # x respresents input data\n' +
                '\n' +
                '        if cfg.MODEL.POSE_ON:\n' +
                '            pose_feat = self.Pose_Head(blob_conv)\n' +
                '            output = self.Pose_Out(pose_feat)\n' +
                '\n' +
                '            if self.with_loss:\n' +
                '                return_dict[\'losses\'] = {}\n' +
                '                return_dict[\'metrics\'] = {}\n' +
                '                loss = self.Pose_Loss(output, targets, target_weight)\n' +
                '                return_dict[\'losses\'][\'joint_loss\'] = loss\n' +
                '            else:\n' +
                '                return_dict[\'preds\'] = {}\n' +
                '                return_dict[\'preds\'] = output\n' +
                '\n' +
                '        if cfg.MODEL.SEMSEG_ON:\n' +
                '            # TODO: complete the returns for POSE_ON situation\n' +
                '            pass\n' +
                '\n' +
                '        return return_dict\n' +
                '```\n'},
        {text:'Generation process of `Generalized_SSD` is similar to `Generalized_CNN`, slightly difference exists in building `Generalized_RCNN`, the **region proposal network**(RPN) and the **feature pyramid network**(FPN) are added, as well as the **mask branch**， **keypoint branch**, **parsing bracnh**, **densepose branch** parallel with **detection branch**, details in corresponding tutorial: [rcnn](https://github.com/BUPT-PRIV/Pet-DOC/tree/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E5%9C%A8MSCOCO2017%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E8%AE%AD%E7%BB%83Mask%20R-CNN%E6%A8%A1%E5%9E%8B)、[ssd](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/Train%20SSD%20model%20on%20MSCOCO2017%20dataset.md).\n'},
        {part_title: 'Use Case'},
        {text:'Let\'s see how to use `Generalized_CNN` to build model:\n' +
                '\n' +
                'First, introduce the `Generalized_CNN` class:\n' +
                '\n' +
                '```Python\n' +
                'from pet.pose.modeling.model_builder import Generalized_CNN\n' +
                '```\n' +
                'Instantiate the model in the training script:\n' +
                '```Python\n' +
                '   # Create model\n' +
                '   model = Generalized_CNN(with_loss=True)\n' +
                '```\n' +
                'The construction of the network during the training process can be completed through these processes. The loss function is not needed during testing, therefore setting the parameter `with_loss` to False, then the network will output the task outputs instead of the losses.\n',className:'segmentation'},
        {part_title:'Convenience and Specification'},
        {text:'According to the standards set by Pet for model construction, network function modules can be quickly and flexibly selected through the configuration system to meet diversified deep learning research requirements.\n'},
        {text:'When enriching and expanding the deep learning model structure for Pet, you are suggested to modify or add new model building tools in full compliance with Pet\'s code implementation standards:\n'},
        {
            ul:[
                'When adding new model building tools for computer vision tasks, the overall network is firstly divided into four network modules according to functions: feature extraction network, task specific network, task output and loss function. Each network module is implemented separately, and the names of different network function modules are registered in the configuration system.',
                'The backbone network is implemented in the order of the down sampling operation and follows the code style of Pet.'
            ]
        },
        {text:'You are welcome to submit valuable methods and code to our github, and we appreciate any useful contributions to the development of Pet.'},
        {part_title:'Reference'},
        {text:'\\[1\\] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CVPR 2016.\n'},
        {text:'\\[2\\] Saining Xie, Ross Girshick, Piotr Dolla ́r, Zhuowen Tu, Kaiming He. Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017.\n'},
        {text:'\\[3\\] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. CVPR 2017.\n'},
        {text:'\\[4\\] Mingxing Tan, Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. arXiv:1905.11946\n'},
        {text:'\\[5\\] Bin Xiao, Haiping Wu, Yichen Wei. Simple Baselines for Human Pose Estimation and Tracking. ECCV 2018.\n'},
        {text:'\\[6\\] Karen Simonyan, Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR 2015.\n'},
        {text:'\\[7\\] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.\n'},
    ],
    dataNav:[
        {
            'Generalized_CNN':[
                'Initialization',
                'forward'
            ]
        },
        'Use Case','Convenience and Specification','Reference'
    ]
}