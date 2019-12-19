export let opt_data={
    key:'opt',
    dataSource:[
        {title:'优化与迭代', className:'title_1'},
        {text:'优化与迭代是训练深度学习模型的核心内容，Pet对深度学习模型的优化迭代操作的实现方法设置了一套标准的实现，将深度学习模型的训练与优化操作归纳为优化器和学习率调度器两项组合操作，在Pet的代码实现中，优化器和学习率调度器具体对应`Optimizer`和`LearningRateScheduler`两个基本Python操作类，两个Python类会在整个训练的过程中一直被用于指导模型的优化。您可以使用配置系统中SOLVER模块的设置来构建优化器和学习率调度器，对模型训练过程中的优化算法、学习率变化以及参数差异化优化策略进行系统设置。',className:'segmentation'},
        {text:'[优化器](#优化器)\n' +
                '\n' +
                '>&#8195;&#8195;[初始化](#初始化)\n' +
                '\n' +
                '>&#8195;&#8195;[build](#build)\n' +
                '\n' +
                '>>&#8195;&#8195;&#8195;&#8195;[get_params_list](#get_params_list)\n' +
                '\n' +
                '>>&#8195;&#8195;&#8195;&#8195;[get_params](#get_params)\n' +
                '\n' +
                '>&#8195;&#8195;[优化算法](#优化算法)\n' +
                '\n' +
                '[学习率调度器](#学习率调度器)\n' +
                '\n' +
                '>&#8195;&#8195;[初始化(init)](#初始化(init))\n' +
                '\n' +
                '>&#8195;&#8195;[step](#step)\n' +
                '\n' +
                '>>&#8195;&#8195;&#8195;&#8195;[get_lr](#get_lr)\n' +
                '\n' +
                '>>&#8195;&#8195;&#8195;&#8195;[update_learning_rate](#update_learning_rate)\n' +
                '\n' +
                '[高效计算]\n' +
                '\n' +
                '>&#8195;&#8195;[模型并行]\n' +
                '\n' +
                '>&#8195;&#8195;[混合精度训练]\n' +
                '\n' +
                '[使用案例](#使用案例)\n' +
                '\n' +
                '[便利与规范](#便利与规范)\n' +
                '\n' +
                '[参考文献](#参考文献)\n' +
                '\n'},
        {part_title:'优化器',className:'title_2'},
        {text:'当您在完成网络模型的构建之后，优化器可以帮助您对网络结构中不同种类的参数的学习率、权重衰减因子以及学习率倍率倍率进行差异化设置，同时还提供一些主流的优化算法，您可以根据您不同的训练需求来配置优化器，优化器的完整代码请参考[$Pet/pet/utils/optimizer.py]()。\n' +
                '\n' +
                '优化器的实现规则与流程如下：\n' ,className:'text_2'},
        {ul:'优化器对构建好的模型进行参数解析，根据需要对模型参数进行归类，不同类型的参数会被分配以不同的权重衰减和学习率倍率；\n' ,className:'ul_1'},
        {ul:'将归类和差异化配置之后的模型参数送入torch提供的优化算法，完成优化器的配置。\n',className:'ul_2'},
        {text:'当您需要对网络模型中的其他参数进行差异化优化设置，或者您需要使用新的优化算法时，您需要在优化器内遵循以上的代码实现标准，将您的改进加入Pet。', className:'segmentation'},
        {h3_title:'初始化'},
        {text:'优化器在Pet中对应`Optimizer`这一具体的Python类，接收`model`、`solver`作为输入，`Optimizer`的主要成员函数包括`get_params_list`、`get_params`以及`build`。在了解`Optimizer`的功能函数之前，我们首先对`Optimizer`类进行初始化：\n'},
        {text:'```Python\n' +
                'class Optimizer(object):\n' +
                '    def __init__(self, model, solver, local_rank=0):\n' +
                '        self.model = model\n' +
                '        self.solver = solver\n' +
                '        self.local_rank = local_rank\n' +
                '\n' +
                '        self.bias_params_list = []\n' +
                '        self.gn_params_list = []\n' +
                '        self.nonbias_params_list = []\n' +
                '\n' +
                '        self.params = []\n' +
                '        self.gn_param_nameset = self.get_gn_param_nameset()\n' +
                '```\n'},
        {text:'`model`是网络结构以及网络参数的集合，Pet将`model`中包含的参数分为`bias_params_list`、`gn_params_list`、`nonbias_params_list`三大类：\n'},
        {ul:'`bias_params_list`：卷积、全连接操作中的偏置\n'},
        {ul:'`gn_params_list`：[Group Normalization]()操作对应的xxxxx参数\n'},
        {ul:'`nonbias_params_list`：卷积、全连接操作中的权重\n' },
        {text:'在目前主流的深度学习算法中，这三大类参数在训练的过程中需要被分配不同的权重衰减因子（weight_decay）和学习率倍率（lr_scale），在`get_params`函数中会有详细讲解。\n'},
        {h3_title:'build',className:'title_3'},
        {text:'解析模型参数解析、差异化设置参数优化策略、选择优化算法。\n' +
                '\n' +
                '```Python\n' +
                '    def build(self):\n' +
                '        assert self.solver.OPTIMIZER in [\'SGD\', \'RMSPROP\', \'ADAM\']\n' +
                '        self.get_params_list()\n' +
                '        self.get_params()\n' +
                '\n' +
                '        if self.solver.OPTIMIZER == \'SGD\':\n' +
                '            optimizer = torch.optim.SGD(\n' +
                '                self.params,\n' +
                '                momentum=self.solver.MOMENTUM\n' +
                '            )\n' +
                '        elif self.solver.OPTIMIZER == \'RMSPROP\':\n' +
                '            optimizer = torch.optim.RMSprop(\n' +
                '                self.params,\n' +
                '                momentum=self.solver.MOMENTUM\n' +
                '            )\n' +
                '        elif self.solver.OPTIMIZER == \'ADAM\':\n' +
                '            optimizer = torch.optim.Adam(\n' +
                '                self.model.parameters(),\n' +
                '                lr=self.solver.BASE_LR\n' +
                '            )\n' +
                '        else:\n' +
                '            optimizer = None\n' +
                '        return optimizer\n'},
        {
            h4_block:[
                {h4_title:'get_params_list'},
                {text:'解析了`model`中各个操作层所包含的参数，将权重、偏置、gn三大类参数分别进行打包。\n' +
                        '\n' +
                        '```Python\n' +
                        '    def get_params_list(self):\n' +
                        '        for key, value in self.model.named_parameters():\n' +
                        '            if value.requires_grad:\n' +
                        '                if \'bias\' in key:\n' +
                        '                    self.bias_params_list.append(value)\n' +
                        '                elif key in self.gn_param_nameset:\n' +
                        '                    self.gn_params_list.append(value)\n' +
                        '                else:\n' +
                        '                    self.nonbias_params_list.append(value)\n' +
                        '            else:\n' +
                        '                logging_rank(\'{} does not need grad.\'.format(key), local_rank=self.local_rank)\n' +
                        '```\n'}
        ],
        },
        {
            h4_block:[
                {h4_title:'get_params',},
                {text:'对三大类参数的权重衰减因子和学习率倍率进行了差异化设置。\n' +
                        '\n' +
                        '```Python\n' +
                        '    def get_params(self):\n' +
                        '        self.params += [\n' +
                        '            {\'params\': self.nonbias_params_list,\n' +
                        '             \'lr\': 0,\n' +
                        '             \'weight_decay\': self.solver.WEIGHT_DECAY,\n' +
                        '             \'lr_scale\': 1},\n' +
                        '            {\'params\': self.bias_params_list,\n' +
                        '             \'lr\': 0 * (self.solver.BIAS_DOUBLE_LR + 1),\n' +
                        '             \'weight_decay\': self.solver.WEIGHT_DECAY if self.solver.BIAS_WEIGHT_DECAY else 0,\n' +
                        '             \'lr_scale\': self.solver.BIAS_DOUBLE_LR + 1},\n' +
                        '            {\'params\': self.gn_params_list,\n' +
                        '             \'lr\': 0,\n' +
                        '             \'weight_decay\': self.solver.WEIGHT_DECAY_GN,\n' +
                        '             \'lr_scale\': 1}\n' +
                        '        ]\n' +
                        '```\n'}
            ]
        },
        {ul:'卷积、全连接层偏置：通常情况下，偏置参数的学习率倍率是权重学习倍率的2倍，且不需要权重衰减，您可以根据您的需要对偏置进行权重衰减。\n'},
        {ul:'gn参数：gn操作的参数默认不进行权重衰减，您可以根据您的需要开启权重衰减。\n'},
        {text:'在此Pet设置模型中所有参数的学习率为0，除偏置外所有参数的学习率倍率为1，这只是一个简单的学习率初始化，在模型训练过程的每一次迭代中，学习率调度器会根据学习率调度策略对所有参数的学习率进行调整。\n' +
                '\n' +
                '当您需要对网络中的某些特定参数进行差异化优化设置时，例如使用[Cascade-RCNN]()算法训练`rcnn`模型时，三个阶段的`RCNN`网络的权重和偏置分贝被赋予不同的学习率倍率，您可以在优化器中通过参数索引的形式尽心该操作，详情请见[教程]()。\n'},
        {h3_title:'优化算法'},
        {text:'Pet提供了随机梯度下降（SGD）、均方根支持（RMSPROP）和自适应矩估计（ADAM）三种优化算法，可以满足绝大部分卷积神经网络的优化。`get_params`将模型中所有参数打包，在此Pet调用`torch`提供的优化算法接口，将所有参数和动量因子作为参数输入优化器，代码实现见`build`函数。\n'},
        {part_title: '学习率调度器'},
        {text:'在优化器构建完毕后，还需要构建学习率调度器，学习率调度器会根据您在配置系统中`SOLVER`部分设定的学习率变化策略在训练过程的每一次迭代中计算新的基础学习率，并对模型中的不同参数调整其差异化学习率，优化器的完整代码请参考[$Pet/pet/utils/lr_scheduler.py]()。\n' +
                '\n' +
                '学习率调度器的实现规则与流程如下：\n'},
        {ul:'根据学习率变化策略与当前迭代数，计算当前这一次迭代所对应的学习率；'},
        {ul:'根据优化器内的差异化设置，将学习率乘以不同的倍率赋值给不同的参数。'},
        {text:'当您需要使用其他的学习率变化策略时，您需要在学习率调度器内遵循以上的代码实现标准，将您的改进加入Pet。\n', className:'segmentation'},
        {h3_title:'初始化(init)'},
        {text:'学习率调度器在Pet中对应`LearningRateScheduler`这一具体的Python类，接收`optimizer`和`solver`作为输入，`LearningRateScheduler`的主要成员函数包括`get_lr`、`update_learning_rate`以及`step`。在了解`LearningRateScheduler`的功能函数之前，我们首先对`Optimizer`类进行初始化：\n' +
                '\n' +
                '```Python\n' +
                'class LearningRateScheduler(object):\n' +
                '    def __init__(self, optimizer, solver, start_iter=1, iter_per_epoch=-1, local_rank=0):\n' +
                '        if not isinstance(optimizer, Optimizer):\n' +
                '            raise TypeError(\'{} is not an Optimizer\'.format(type(optimizer).__name__))\n' +
                '        self.optimizer = optimizer\n' +
                '\n' +
                '        self.solver = solver\n' +
                '        assert self.solver.LR_POLICY in [\'STEP\', \'COSINE\', \'POLY\']\n' +
                '        assert self.solver.WARM_UP_METHOD in [\'CONSTANT\', \'LINEAR\']\n' +
                '        self.base_lr = self.solver.BASE_LR\n' +
                '        self.new_lr = self.base_lr\n' +
                '\n' +
                '        self.iteration = start_iter\n' +
                '        self.iter_per_epoch = iter_per_epoch\n' +
                '        self.local_rank = local_rank\n' +
                '\n' +
                '        if \'MAX_ITER\' in self.solver:\n' +
                '            self.max_iter = self.solver.MAX_ITER\n' +
                '            self.warm_up_iters = self.solver.WARM_UP_ITERS\n' +
                '            self.steps = self.solver.STEPS  # only useful for step policy\n' +
                '        else:\n' +
                '            assert self.iter_per_epoch > 0  # need to specify the iter_per_epoch\n' +
                '            self.conver_epoch2iter()\n' +
                '```\n'},
        {ul:'初始化学习率调度器的第一步是要先检查`optimizer`是否具有一个真正的优化器结构体，这是Pet的一种异常预警机制，可以在您需要扩展新的代码时，提醒您在训练脚本中可能遗忘或错误地定义优化器，这种预警机制被广泛运用在目前的深度学习平台和算法工程中。'},
        {ul:'在获取`optimizer`和`solver`并将它们设置为学习率调度器的成员变量之后，还需要通过Pet的异常预警机制检查您在配置系统的`solver`模块中配置的字段是否是Pet所支的持学习率衰减和预热策略。'},
        {ul:'学习率调度器的主要全局成员变量如下表所示，这些全局成员变量将在整个深度学习模型训练过程中被用来调整每一次迭代的学习率：'},
        {table:{
                titles:['成员变量','含义'],
                data:[["base_lr","基础学习率"],["new_Lr","当前学习率"],["iteration","当前迭代数"],["max_iter","最大迭代数"],["warm_up_iters","用于学习率预热的迭代数"],["steps","阶段下降策略中的学习率下降迭代数"]]
            }
            , className:'table_1'},
        {note:{
            text:'尽管Pet在记录日志时根据不同的视觉任务支持iter与epoch两种迭代计数单位，但是在学习率调度器内只使用迭代数作为调度学习的率的计数规则。\n' +
                '\n' +
                'Rcnn任务使用迭代数作为日志计数单位，但对于姿态分析、单阶段目标检测、图像分类、图像分割等训练全数据迭代次数较多的计算机视觉任务，学习率调度器在初始化时将全数据迭代数通过`conver_epoch2iter`函数转化成最大迭代数，通过检查迭代次数来调整学习率。\n' +
                '\n' +
                '详细代码请见[conver_epoch2iter]()\n'
            }},
        {h3_title:'step'},
        {text:'在对学习率调度器进行初始化之后，需要构建学习率调度器在迭代过程中的成员函数`step`，通过`step`函数可以在每一次迭代中，根据学习率优化策略以及当前迭代数计算学习率，并将当前学习率分配给网络模型中不同的参数。\n' +
                '\n' +
                '```Python\n' +
                '    def step(self, cur_iter=None):\n' +
                '        if cur_iter is None:\n' +
                '            cur_iter = self.iteration + 1\n' +
                '        self.iteration = cur_iter\n' +
                '\n' +
                '        # update learning rate\n' +
                '        self.new_lr = self.get_lr()\n' +
                '        self.update_learning_rate()\n' +
                '```\n'},
        {
            h4_block:[
                {h4_title:'get_lr',},
                {text:'根据配置系统中设置的学习率变化策略，在每一次迭代中计算当前的学习学习率，Pet将模型训练过程中学习率变化分为两个阶段：预热阶段和下降阶段。\n',},
                {h5_title:'学习率预热策略(Learning rate warming up)'},
                {text:'Pet将学习率预热策略收纳于学习率调度器中，提供了连续（CONSTANT）和线性（LINEAR）两种学习率预热策略。\n'},
                {block:[
                        {text:'在当前深度学习模型的训练过程中，批量优化技术已经成为一种通用的训练方法，但是小批量的数据不足以代表整个用于训练的数据集的统计分布，当学习率设置不合理时，模型的优化方向可能并不是全局最优，这可能导致模型在迭代优化过程中出现局部最优或者是不收敛的情况。学习率预热策略在训练的开始阶段将学习率保持在一个比较小的水平，并在最大预热迭代次数之内使学习率缓慢增长，保证模型在优化的最开始不会偏向错误的方向。\n'}
                    ],
                },
                {h5_title: '学习率下降策略'},
                {text:'Pet为深度卷积神经网络模型的训练和优化提供了阶段下降、余弦下降、复数下降三种学习率下降策略，他们在配置系统的`SOLVER`模块中对应的字段分别是`STEP`、`COSINE`和`POLY`，在学习率预热之后，学习率变成`SOLVER`中设置的基础学习率，并随着迭代次数的增长按照策略下降。\n' +
                        '\n' +
                        '```Python\n' +
                        '    def get_lr(self):\n' +
                        '        new_lr = self.base_lr\n' +
                        '        if self.iteration <= self.warm_up_iters:  # warm up\n' +
                        '            if self.solver.WARM_UP_METHOD == \'CONSTANT\':\n' +
                        '                warmup_factor = self.solver.WARM_UP_FACTOR\n' +
                        '            elif self.solver.WARM_UP_METHOD == \'LINEAR\':\n' +
                        '                alpha = self.iteration / self.warm_up_iters\n' +
                        '                warmup_factor = self.solver.WARM_UP_FACTOR * (1 - alpha) + alpha\n' +
                        '            else:\n' +
                        '                raise KeyError(\'Unknown SOLVER.WARM_UP_METHOD: {}\'.format(self.solver.WARM_UP_METHOD))\n' +
                        '            new_lr = self.base_lr * warmup_factor\n' +
                        '        elif self.iteration > self.warm_up_iters:\n' +
                        '            if self.solver.LR_POLICY == \'STEP\':\n' +
                        '                new_lr = self.base_lr * self.solver.GAMMA  bisect_right(self.steps, self.iteration)\n' +
                        '            elif self.solver.LR_POLICY == \'COSINE\':\n' +
                        '                actual_iter = self.max_iter - self.warm_up_iters  # except warm up\n' +
                        '                new_lr = 0.5 * self.base_lr * (\n' +
                        '                    np.cos((self.iteration - self.warm_up_iters - 1) * np.pi / actual_iter) + 1.0)\n' +
                        '            elif self.solver.LR_POLICY == \'POLY\':\n' +
                        '                actual_iter = self.max_iter - self.warm_up_iters  # except warm up\n' +
                        '                new_lr = self.base_lr * (\n' +
                        '                    (1. - float(self.iteration - self.warm_up_iters - 1) / actual_iter)  self.solver.LR_POW)\n' +
                        '            else:\n' +
                        '                raise KeyError(\'Unknown SOLVER.LR_POLICY: {}\'.format(self.solver.LR_POLICY))\n' +
                        '        return new_lr\n' +
                        '```\n'}
            ]

        },
        {
            h4_block:[
                {h4_title:'update_learning_rate'},
                {text:'更新模型中所有参数的学习率，偏置、权重、gn参数因为不同的学习率倍率获得不同的学习率。\n' +
                        '\n' +
                        '```Python\n' +
                        '    def update_learning_rate(self):\n' +
                        '        """Update learning rate\n' +
                        '        """\n' +
                        '        cur_lr = self.optimizer.param_groups[0][\'lr\']\n' +
                        '        if cur_lr != self.new_lr:\n' +
                        '            ratio = _get_lr_change_ratio(cur_lr, self.new_lr)\n' +
                        '            if ratio > self.solver.LOG_LR_CHANGE_THRESHOLD and self.new_lr >= 1e-7:\n' +
                        '                logging_rank(\'Changing learning rate {:.6f} -> {:.6f}\'.format(cur_lr, self.new_lr),\n' +
                        '                             local_rank=self.local_rank)\n' +
                        '            # Update learning rate, note that different parameter may have different learning rate\n' +
                        '            for ind, param_group in enumerate(self.optimizer.param_groups):\n' +
                        '                if \'lr_scale\' in param_group:\n' +
                        '                    lr_scale = param_group[\'lr_scale\']\n' +
                        '                else:\n' +
                        '                    lr_scale = 1\n' +
                        '                param_group[\'lr\'] = self.new_lr * lr_scale\n' +
                        '```\n'}
            ]
        },
        {part_title:'高效计算'},
        {text:'深度学习模型以庞大的数据规模作为基础，学习能够普遍表达目标特征的网络参数，这一过程十分漫长，一个深度学习模型训练的时间动辄十数个小时，某些大型模型甚至需要数天的时间才能训练完毕。\n',},
        {part_title:'使用案例'},
        {text:'Pet在训练过程中通过调用`Optimizer`类与`LearningRateScheduler`类来构建优化器，代码如下：\n' +
                '\n' +
                '```Python\n' +
                'from pet.utils.optimizer import Optimizer\n' +
                'from pet.utils.lr_scheduler import LearningRateScheduler\n' +
                '\n' +
                'optimizer = Optimizer(model, cfg.SOLVER, local_rank=args.local_rank).build()\n' +
                'scheduler = LearningRateScheduler(optimizer, cfg.SOLVER, start_iter=0, iter_per_epoch=cfg.TRAIN.ITER_PER_EPOCH,\n' +
                '                                  local_rank=args.local_rank)\n' +
                '```\n',},
        {part_title:'便利与规范'},
        {text:'基于Pet对优化迭代操作所设置的一系列规范化实现，可以方便地定制出不同的优化器与学习率调度器来满足您训练深度学习模型时的需求。使用优化器与学习率调度器，您将在训练深度学习模型时获得以下便利：\n'},
        {ul:'您可以在Pet所支持的所有视觉任务中使用优化器和学习率调度器来训练您的网络模型，我们根据不同的视觉任务与算法提供了一些建议性优化策略与配置，因视觉任务的差异而导致的代码差异将不再是您的困扰。'},
        {ul:'优化器和学习率调度器提供了丰富的优化算法以及学习率调度策略，在此我们支持真正有价值的优化迭代方法，让您可以高效率地进行深度学习研究。'},
        {text:'当您需要对深度学习模型的优化与迭代方法进行丰富拓展时，您需要完全遵循Pet的代码实现标准来改进优化器与学习率调度器，欢迎您将有价值的方法和代码提交到我们的github，我们十分感谢任何对于发展Pet有益的贡献。\n'},
        {part_title:'参考文献'},
        {text:'[[1]]() YuXin Wu and Kaiming He. Group normalization. CVPR 2018.\n' +
                '\n' +
                '[[2]]() Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich\n' +
                'Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaev, Ganesh\n' +
                'Venkatesh, and others. 2017. MIXED PRECISION TRAINING. ICLR 2018.\n'}
    ],
    };