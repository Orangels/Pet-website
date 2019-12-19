export let model_construction_data = {
    key:'model_construction',
    dataSource: [
        {title:'模型构建'},
        {text:'网络结构是深度学习算法研究的核心内容之一，对于不同的计算机视觉任务，Pet使用如下几个python类构建相应的卷积神经网络：\n'},
        {
            ul:[
                '`Generalized_CNN`：图像分类（CNN）、姿态估计（pose）、语义分割（semseg）。',
                '`Generalized_SSD`：单阶段目标检测（ssd）。',
                '`Generalized_RCNN`：基于区域的目标检测与实例分析（rcnn）。'
            ]
        },
        {text:'`Generalized_CNN`、`Generalized_SSD `、`Generalized_RCNN `等网络构建工具是以类的形式来定义和使用的，这里以`pose`、`semseg`任务的网络构建工具`Generalized_CNN`为例，来介绍Pet中网络的构建过程。\n',className:'segmentation'},
        {part_title:'Generalized_CNN'},
        {text:'`Generalized_CNN`、`Generalized_SSD `、`Generalized_RCNN `等网络构建工具是以类的形式来定义和使用的，这些模型构建工具具有高度的结构一致性和风格一致性，能够最大限度得帮助您快速使用Pet搭建不同视觉任务的网络结构，进行计算机视觉算法的研究。在以上的一系列工具中，Pet清晰地将卷积神经网络根据功能分为特征提取网络、功能网络、任务输出、损失函数几个网络模块，将各个模块通过输入和输出相互连接，顺序构建网络模型。\n'},
        {h3_title:'初始化'},
        {text:'模型构建在Pet姿态估计任务中对应`Generalized_CNN`这一具体的Python类，姿态估计任务主要实现的是单实例的分析。`Generalized_CNN`构建了`Conv_Body`、`Pose_Head`、`Pose_Out`、`Pose_Loss`，`Generalized_CNN`的主要成员函数为`forward`。在了解`Generalized_CNN`的功能函数之前，我们首先对`Generalized_CNN`类进行初始化：\n' +
                '\n' +
                '```Python\n' +
                'class Generalized_CNN(nn.Module):\n' +
                '    def __init__(self, with_loss=False):\n' +
                '        super().__init__()\n' +
                '        assert cfg.MODEL.POSE_ON\n' +
                '\n' +
                '        # Backbone for feature extraction\n' +
                '        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()\n' +
                '\n' +
                '        self.with_loss = with_loss\n' +
                '        # Pose Estimation Branch\n' +
                '        if cfg.MODEL.POSE_ON:\n' +
                '            self.Pose_Head = get_func(cfg.POSE.POSE_HEAD)(self.Conv_Body.dim_out)\n' +
                '            self.Pose_Out = get_outputs(cfg.POSE.POSE_HEAD)(self.Pose_Head.dim_out)\n' +
                '            self.Pose_Loss = get_loss(cfg.POSE.POSE_HEAD)\n' +
                '\n' +
                '        # Semantic Segmentation Branch\n' +
                '        if cfg.MODEL.SEMSEG_ON:\n' +
                '            raise NotImplementedError\n' +
                '```\n'},
        {text:'对于单人体姿态估计、实例分割、人体部位分析、密集姿态估计等实例分析任务，Pet支持将实例从原始图像中切分成单个实例对象进行相关分析，因此在`Generalized_CNN`的初始化过程中，加入了对其他任务的功能网络、任务输出和损失函数的构建，同时还支持FPN的构建。Pet中所有任务的网络模型均可以通过在配置文件中设置相应的字段来构建，设置的字段需要与Pet中所提供的网络模块的函数名完全一致，通过Pet修饰器功能即可调用不同的网络模块，详细代码参考[$Pet/pet/pose/modeling/registry.py](https://github.com/BUPT-PRIV/Pet/blob/master/pet/pose/modeling/registry.py)、[$Pet/pet/utils/registry.py](https://github.com/BUPT-PRIV/Pet/blob/master/pet/utils/registry.py)。'},
        {
            ul:[
                '在配置系统中的`cfg.BACKBONE.CONV_BODY`字段进行设置，即可调用诸如[ResNet](https://arxiv.org/abs/1512.03385)\\[1\\]、[ResNeXt](https://arxiv.org/abs/1611.05431v2)\\[2\\]、[MobileNet-v1](https://arxiv.org/abs/1704.04861v1)\\[3\\]、[EfficientNet](https://arxiv.org/abs/1905.11946v2)\\[4\\]等网络结构作为特征提取网络。',
                '在配置系统中的`cfg.POSE.POSE_HEAD`字段为`simple_xdeconv_head`即可构建如[Simple Baselines](https://arxiv.org/abs/1804.06208v2)\\[5\\]中的姿态估计功能网络。',
                '`cfg.BACKBONE.CONV_BODY`与`cfg.POSE.POSE_HEAD`只是决定了所调用的网络模块的结构，还可以进一步对特征提取网络和功能网络的细致结构进行设置。例如可以通过设置`cfg.BACKBONE.RESNET.LAYERS`选择ResNet网络的层数，由此可以构建ResNet18、ResNet50、ResNet101；或者在使用MobileNet-v1时，设置`cfg.BACKBONE.MV1.WIDEN_FACTOR`来决定MobileNet-v1的通道缩放因子，更多的配置说明见[配置系统](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%9E%B6%E6%9E%84/component-collects/%E9%85%8D%E7%BD%AE%E7%B3%BB%E7%BB%9F.md)。'
            ]
        },
        {h3_title:'特征提取网络'},
        {text:'以ResNet50为例，[$Pet/pet/pose/modeling/backbone/ResNet.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/pet/pose/modeling/backbone/ResNet.py)中的`resnet`函数经过Python修饰器注册之后，可在在配置文件中将`cfg.BACKBONE.CONV_BODY`字段设为`resnet`，即可调用`ResNet`类进行特征提取网络的搭建。\n' +
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
        {text:'特征提取网络是分类网络的主体结构，也被称为基础网络（BACKBONE），在各个视觉任务中都会被用来进行特征提取，Pet严格遵循相关论文对于基础网络进行了标准实现。在`$Pet/pet/models`路径下，进行了[VGG](https://arxiv.org/abs/1409.1556)\\[6\\]、ResNet、[Inception-v3](https://arxiv.org/abs/1512.00567)\\[7\\]、MobileNet、[ShuffleNet-v2](https://arxiv.org/pdf/1807.11164)[8]等优秀卷积神经网络的实现，示例参考[$Pet/pet/models/imagenet/resnet.py](https://github.com/BUPT-PRIV/Pet-dev/tree/master/pet/models/imagenet)。\n'},
        {text:'姿态估计任务中，上面代码中所示的`ResNet`类在进行初始化时继承了[$Pet/pet/models/imagenet/resnet.py](https://github.com/BUPT-PRIV/Pet/tree/master/pet/models/imagenet)中的`ResNet`父类，然后针对视觉任务的不同，调整网络的深度、宽度、ResNet第一阶段等细致化网络结构。其他基础网络结构采用相同的方式进行构建。\n' +
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
                {text:'基础网络结构包含用于分类的`avgpool`层和全连接层，在其他任务中会删除，否则会在其他视觉任务的模型中引入冗余参数。\n'}
            ]
        },
        {h3_title:'功能网络、任务输出、损失函数'},
        {text:'功能网络、任务输出、损失函数的搭建与特征提取网络相同，通过Python修饰器即可保证能够被`Generalized_CNN`所调用。在[$Pet/pet/pose/modeling/pose_head/simple_head.py](https://github.com/BUPT-PRIV/Pet/blob/master/pet/pose/modeling/pose_head/simple_head.py)中实现了三个网络模块。\n'},
        {ul:'功能网络'},
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
        {ul:'任务输出'},
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
        {ul:'损失函数'},
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
        {text:'四个网络模块通过上一模块的输出连接到下一模块的输入构建了整个网络，功能网络接收特征提取网络输出的特征图，任务输出接收功能网络输出的特征图，损失函数接收任务输出的预测结果。网络在训练过程中，四个网络模块都会被使用， `self.with_loss`变量设为`True`来标识需要损失函数(loss)，测试中仅用到了前面三个部分， `self.with_loss`在测试时将其设为`False`。在训练与测试两个模式下，分别返回偏差与预测内容，下面是`Generalized_CNN`的前向计算过程。\n' +
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
        {text:'`Generalized_SSD`的网络生成过程与`Generalized_CNN`相似，`Generalized_RCNN`的模型构建过程与前两者略有不同，主要是加入了**区域建议网络**（RPN）与**特征金字塔网络**（FPN），以及与**检测分支**并行的**实例分析分支**、**关键点分支**、**人体部分分析分支**和**密集姿态分支**等，详细介绍见相应教程：[rcnn](https://github.com/BUPT-PRIV/Pet-DOC/tree/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E5%9C%A8MSCOCO2017%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E8%AE%AD%E7%BB%83Mask%20R-CNN%E6%A8%A1%E5%9E%8B)、[ssd](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/Train%20SSD%20model%20on%20MSCOCO2017%20dataset.md)。\n'},
        {part_title: '使用案例'},
        {text:'接下来介绍Pet在训练过程中如何使用模型构建这一Python类：\n' +
                '\n' +
                '首先，引入Generalized_CNN类：\n' +
                '\n' +
                '```Python\n' +
                'from pet.pose.modeling.model_builder import Generalized_CNN\n' +
                '```\n' +
                '在训练脚本中将模型结构实例化：\n' +
                '```Python\n' +
                '   # Create model\n' +
                '   model = Generalized_CNN(with_loss=True)\n' +
                '```\n' +
                '通过这一过程即可完成在训练过程中网络的构建，模型测试过程不需要损失函数这一网络模块，因此在实例化姿态估计网络`Generalzed_CNN`时需要将`with_loss`设置为默认值False，届时网络将会输出任务输出，而不是损失函数。\n',className:'segmentation'},
        {part_title:'便利与规范'},
        {text:'遵循Pet对于模型构建所设定的标准，可以快速、灵活地通过配置系统选择网络功能模块，满足多样化的深度学习研究需求。\n'},
        {text:'当在Pet下对深度学习模型结构进行丰富与拓展时，您必须完全遵循Pet的代码实现标准对模型构建工具进行修改或添加新的模型构建工具：\n'},
        {
            ul:[
                '在添加新的计算机视觉任务的模型构建工具时，首先将整体网络按功能分为特征提取网络、功能网络、任务输出、损失函数四个网络模块，每个网络模块分别实现，并为不同的网络功能模块在配置系统中注册名称。',
                '主干网络按照下采样阶段顺序实现并遵循Pet的代码风格。'
            ]
        },
        {text:'欢迎您将有价值的方法和代码提交到我们的github，我们十分感谢任何对于发展Pet有益的贡献。'},
        {part_title:'参考文献'},
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
                '初始化',
                'forward'
            ]
        },
        '使用案例','便利与规范','参考文献'
    ]
}