export let opt_data={
    key:'opt',
    dataSource:[
        {title:'Optimization', className:'title_1'},
        {text:'Iterative optimization is the core content of the training deep learning model. Pet sets a set of standard implementations for the optimization of the iterative operation of the deep learning model. The training and optimization operations of the deep learning model are summarized into the optimizer and the learning rate scheduler. The item combines operations and uses mixed precision training and distributed data parallel methods provided by PyTorch to improve the efficiency of model training. In the code implementation of Pet, the optimizer and the learning rate scheduler correspond to two basic Python operation classes: `Optimizer` and `LearningRateScheduler`. Two Python classes are used to guide the optimization of the model throughout the training process. The optimizer and the learning rate scheduler can be built using the settings of the `SOLVER` module in the configuration system, and the optimization algorithm, learning rate change, and parameter differentiation optimization strategy in the model training process are systematically set.',className:'segmentation'},
        {part_title:'Optimizer'},
        {text:'After you complete the construction of the network model, the optimizer can help you to differentiate the learning rate, weight attenuation factor and learning rate magnification ratio of different kinds of parameters in the network structure, and also provide some mainstream optimization algorithms. The optimizer can be configured according to your different training needs. For complete code of the optimizer, please refer to [$Pet/pet/utils/optimizer.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/pet/utils/optimizer.py).\n'},
        {text:'The implementation rules and processes of the optimizer are as follows:\n', className:'text_1'},
        {
            ul:[
                'The optimizer performs parameter analysis on the constructed model, classifies the model parameters as needed, and different types of parameters are assigned different weight attenuation and learning rate overrides;',
                'The model parameters after the categorization and differential configuration are sent to the optimization algorithm provided by torch to complete the configuration of the optimizer.'
            ]
        },
        {text:'When you need to make different optimization settings for other parameters in the network model, or if you need to use the new optimization algorithm, you need to follow the above code implementation standards in the optimizer and add your improvements to Pet.\n'},
        {h3_title:'initialization'},
        {text:'The optimizer corresponds to `Optimizer`, a concrete Python class that accepts `model` and `solver` as input. The main member functions of `Optimizer` include `get_params_list`, `get_params`, and `build`. Before we understand the function of `Optimizer`, we first initialize the `Optimizer` class:\n' +
                '\n' +
                '```Python\n' +
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
                '```\n' +
                '\n' +
                '`model` is a collection of network structure and network parameters. Pet divides the parameters contained in `model` into three categories: `bias_params_list`, `gn_params_list`, `nonbias_params_list`:'},
        {
            ul:[
                '`bias_params_list`: offset in convolution, full join operation','scale and shift parameters in [Group Normalization](https://arxiv.org/abs/1803.08494)\\[1\\] operation',
                '`nonbias_params_list`：weights in convolution, full join operations.'
            ]
        },
        {text:'In the current mainstream deep learning algorithms, these three types of parameters need to be assigned different weight attenuation factors (weight_decay) and learning rate multipliers (lr_scale) during the training process. See `get_params` for details.\n'},
        {h3_title:'build'},
        {text:'Analyze model parameter analysis, differentiate parameter setting optimization strategy, and select optimization algorithm.\n' +
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
                {text:'The parameters contained in each operation layer in `model` are parsed, and the weight, offset, and gn parameters are packaged separately.' +
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
                        '```\n'},
                {h4_title: 'get_params'},
                {text:'The weight attenuation factor and the learning rate multiplier of the three types of parameters are differentiated.\n' +
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
                        '```\n'},
                {
                    ul:[
                        'Convolution, full connection layer offset: In general, the learning rate multiplier of the offset parameter is twice the weight learning magnification, and no weight attenuation is required. You can weight the offset according to your needs.',
                        'gn parameter: The parameter of gn operation does not perform weight attenuation by default. You can turn on weight attenuation according to your needs.'
                    ]
                },
                {text:'In this Pet setting model, the learning rate of all parameters is 0, and the learning rate multiplier of all parameters except offset is 1, which is just a simple learning rate initialization. In each iteration of the model training process, the learning rate scheduler The learning rate of all parameters is adjusted according to the learning rate scheduling strategy.\n'},
                {text:'When you need to make differentiated optimization settings for certain parameters in the network, such as using the [Cascade-RCNN](https://arxiv.org/abs/1712.00726) algorithm to train the `rcnn` model, the weights and offsets of the three phases of the `RCNN` network are To give different learning rate overrides, you can do this in the optimizer by means of parameter indexing. \n'},
            ],
        },
        {h3_title:'Optimization'},
        {text:'Pet provides three optimization algorithms: **Random Gradient Descent** (SGD), ** Root Mean Square Support** (RMSPROP) and **Adaptive Moment Estimation** (ADAM), which can satisfy most convolutional nerves. Network optimization. `get_params` packs all the parameters in the model. In this Pet call the optimization algorithm interface provided by `torch`, all parameters and momentum factors are input as parameters to the optimizer, and the code implementation sees the `build` function.\n'},
        {part_title: 'Learning Rate Scheduler'},
        {text:'After the optimizer is built, you need to build a learning rate scheduler. The learning rate scheduler calculates new basic learning in each iteration of the training process based on the learning rate change strategy you set in the `SOLVER` section of the configuration system. Rate, and adjust the differential learning rate for different parameters in the model. For the complete code of the optimizer, please refer to [$Pet/pet/utils/lr_scheduler.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/pet/utils/lr_scheduler.py).\n'},
        {text:'The implementation rules and procedures of the learning rate scheduler are as follows:：\n'},
        {
            ul:['Calculate the learning rate corresponding to the current iteration based on the learning rate change strategy and the current number of iterations;','Assign the learning rate to different magnifications to different parameters according to the differentiating settings in the optimizer.']
        },
        {text:'When you need to use other learning rate change strategies, you need to follow the above code implementation standards in the learning rate scheduler and add your improvements to Pet.\n'},
        {h3_title:'init'},
        {text:'The learning rate scheduler corresponds to the specific Python class `LearningRateScheduler` in Pet, receiving `optimizer` and `solver` as input. The main member functions of `LearningRateScheduler` include `get_lr`, `update_learning_rate` and `step`. Before we understand the function of `LearningRateScheduler`, we first initialize the `Optimizer` class:\n' +
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
                '            self.conver_epoch2iter(https://github.com/BUPT-PRIV/Pet-dev/blob/02a15242c6f0bae47d28bad205953c26462c38c1/pet/utils/lr_scheduler.py)\n' +
                '```\n'},
        {
            ul:[
                'The first step in initializing the learning rate scheduler is to check if the `optimizer` has a real optimizer structure. This is an abnormal warning mechanism for Pet, which reminds you when you need to extend the new code. The optimizer may be forgotten or incorrectly defined in the training script. This early warning mechanism is widely used in current deep learning platforms and algorithm engineering.',
                'After getting `optimizer` and `solver` and setting them as member variables of the learning rate scheduler, you also need to check whether the field configured in the `solver` module of the configuration system is Pet by the abnormal warning mechanism of Pet. The learning rate attenuation and warm-up strategy.',
                'The main global member variables of the learning rate scheduler are shown in the following table. These global member variables will be used to adjust the learning rate of each iteration throughout the deep learning model training process:'
            ]
        },
        {
            table:{
                titles:['Member Variable','Meaning'],
                data:[["base_lr","Basic Learning Rate"],["new_Lr","Current Learning Rate"],["iteration","Current iteration numbe"],["max_iter"," Maximum Iterations"],["warm_up_iters","Iterations for learning rate warming"],["steps","Learning Rate Decline Iterations in Phase Down Strategy"]]
            }
            , className:'table_1'},
        {
            note:[
                {text:'Although Pet supports both iter and epoch iteration count units according to different visual tasks when recording, only the iteration number is used in the learning rate scheduler as the counting rule of the scheduling learning rate.\n'},
                {text:'The Rcnn task uses the iteration number as the log count unit, but for the computer vision tasks that train the full data iterations, such as attitude analysis, single-stage target detection, image classification, image segmentation, etc., the learning rate scheduler will initialize the full data at initialization time. The `conver_epoch2iter` function is converted to the maximum number of iterations, and the learning rate is adjusted by checking the number of iterations.\n'},
                {text:'For detailed code, please see [conver_epoch2iter]()\n'}
            ]
        },
        {h3_title:'step'},
        {text:'After the learning rate scheduler is initialized, the member function `step` of the learning rate scheduler in the iterative process needs to be constructed. Through the `step` function, the learning rate optimization strategy and the current iteration number can be calculated in each iteration. Rate and assign the current learning rate to different parameters in the network model.\n' +
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
            h4_block: [
                {h4_title:'get_lr'},
                {text:'According to the learning rate change strategy set in the configuration system, the current learning learning rate is calculated in each iteration, and Pet divides the learning rate change in the model training process into two phases: the warm-up phase and the descending phase.\n'},
                {h5_title:'Learning Rate Warm-up Strategy'},
                {text:'Pet puts the learning rate warm-up strategy in the learning rate scheduler, and provides two learning rate warm-up strategies: **Continuous** (CONSTANT) and **Linear** (LINEAR).\n'},
                {text:'In the training process of the current deep learning model, the batch optimization technique has become a general training method, but the small batch of data is not enough to represent the statistical distribution of the entire data set used for training. When the learning rate is not set properly, the model The optimization direction may not be globally optimal, which may result in a local optimization or non-convergence of the model during iterative optimization. The learning rate warm-up strategy keeps the learning rate at a relatively small level at the beginning of the training, and slowly increases the learning rate within the maximum number of warm-up iterations, ensuring that the model does not tend to be in the wrong direction at the beginning of optimization.\n'},
                {h5_title: 'Learning Rate Reduction Strategy'},
                {text:'Pet provides a ** phase drop **, ** cosine drop **, ** complex number drop ** for the training and optimization of the deep convolutional neural network model. They are configuring the system\'s `SOLVER` module. The corresponding fields are `STEP`, `COSINE`, and `POLY`. After the learning rate is warmed up, the learning rate becomes the basic learning rate set in `SOLVER`, and decreases according to the strategy as the number of iterations increases.\n' +
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
                        '                new_lr = self.base_lr * self.solver.GAMMA ** bisect_right(self.steps, self.iteration)\n' +
                        '            elif self.solver.LR_POLICY == \'COSINE\':\n' +
                        '                actual_iter = self.max_iter - self.warm_up_iters  # except warm up\n' +
                        '                new_lr = 0.5 * self.base_lr * (\n' +
                        '                    np.cos((self.iteration - self.warm_up_iters - 1) * np.pi / actual_iter) + 1.0)\n' +
                        '            elif self.solver.LR_POLICY == \'POLY\':\n' +
                        '                actual_iter = self.max_iter - self.warm_up_iters  # except warm up\n' +
                        '                new_lr = self.base_lr * (\n' +
                        '                    (1. - float(self.iteration - self.warm_up_iters - 1) / actual_iter) ** self.solver.LR_POW)\n' +
                        '            else:\n' +
                        '                raise KeyError(\'Unknown SOLVER.LR_POLICY: {}\'.format(self.solver.LR_POLICY))\n' +
                        '        return new_lr\n' +
                        '```\n'},
                {h4_title:'update_learning_rate'},
                {text:'Update the learning rate, bias, weight, and gn parameters of all parameters in the model to obtain different learning rates because of different learning rate overrides.\n' +
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
                        '```\n'},
            ]
        },
        {part_title:'Efficient Calculation'},
        {text:'The deep learning model is based on a large data scale and learns the network parameters that can universally express the target features. This process is very long. A deep learning model is trained for ten hours, and some large models even take several days. Only after training is completed. There have been many studies dedicated to improving the computational efficiency of deep learning models, and the development teams of major deep learning frameworks are constantly introducing accelerated computing methods. Pet continues to follow PyTorch\'s latest interfaces and methods, applying distributed data parallelism and mixed-precision training techniques to deep learning.\n'},
        {h3_title:'Model Parallel'},
        {text:'Pet uses a distributed data parallel model based on the `torch.distributed` toolkit to train all computer vision models, and distributed data parallelism at the model level. For a distributed toolkit for torch, see [torch.distributed]( Https://pytorch.org/docs/stable/distributed.html#distributed-basics). `DistributedDataParallel` implements parallel computing of models by slicing and distributing data on different devices on the batch dimension. The model is assigned to each host or GPU. Each model backup is responsible for processing part of the input data during the forward calculation process. During backpropagation, the gradients on multiple devices will be averaged.\n'},
        {text:'`Distributed` has two methods: single-threaded multi-GPU and multi-threaded single GPU. Single-threaded multi-GPU mode generates a thread on each host/node that invokes all GPU computing resources; multi-threaded single GPU mode generates multiple threads on each host/node, each thread running on a single GPU . Pet uses the latter as a way to improve the computational efficiency of the model. This is the fastest method for data parallel training using PyTorch, which is suitable for single-node (multi-GPU) and multi-node data parallel training. For single-node multi-GPU data parallel training, it proved to be much faster than the other data parallel training tool `torch.nn.DataParallel` provided by PyTorch.\n'},
        {
            note:[
                {text:'Pet uses NCCL as a cross-GPU communication framework. NCCL can achieve the fastest computing speed in multi-threaded single GPU mode. nccl can also support single-node multi-GPU mode. Use `torch.nn.parallel.DistributedDateParallel` to note the following:'},
                {
                    ul:[
                        'Before the network is distributed, all the parameters must be registered. After the distributed deployment, the newly added parameters in the model will be invalid, and no parameters can be removed.',
                        'When the distributed parallel tool is used, the multi-threaded data loader is also required for data loading. See [Data Loading](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%9E%B6%E6%9E%84/component-collects/%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD/%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD.md) for details.',
                        'When using mixed-precision training techniques, Pet uses the APEX driver from NVIDIA to speed up calculations.'
                    ]
                },
            ]
        },
        {h3_title:'Mixed Precision Training'},
        {text:'The deep learning in the field of artificial intelligence is becoming more and more demanding for computing resources. The powerful neural network needs the powerful computing power of processors such as GPU and TPU. However, in actual use, the power consumption, computing power, etc. of the device. The actual conditions limit the volume of the neural network everywhere, and the training of large neural networks is also very time consuming. Model quantification techniques such as lightweight network and model pruning have always been the research hotspots, but the weakening of the model\'s volume will inevitably bring losses to the accuracy of deep neural networks to some extent, in order to accelerate the neural network without loss of precision. The training and testing speed, while reducing the model capacity, mixed-precision training technology was jointly launched by Baidu and Nvidia in 2018.\n'},
        {text:'The hybrid precision training technique is to use the FP32 and FP16 two kinds of calculation precision to jointly train the model without losing the accuracy of the model. Finally, the model parameters are stored in semi-precision floating point numbers on the NVIDIA [official website] (https://devblogs) You can find out more about mixing precision training at .nvidia.com/mixed-precision-training-deep-neural-networks/).\n'},
        {text:'Prior to the advent of hybrid precision training techniques, deep neural networks have been performing forward and reverse calculations with 32-bit floating-point numbers (float32), known as single-precision floating-point calculations (FP32), and double-precision floating-point calculations (FP64). Also often used. The CUDA programming language supports the FP16 data type for semi-precision calculations. The architecture of the NVIDIA GPU has long provided the numerical representation of FP16. After the emergence of the hybrid precision training technology, the computational efficiency of the neural network has been greatly improved, using the mixing precision training. Technology will yield great convenience in two ways:\n'},
        {
            ul:['Significant reduction in memory space requirements: Compared to single-precision and double-precision, the semi-precision floating-point format (FP16) uses 16 bits to express values, and the memory space saved can be used to train a much larger neural network model, or Use a larger batch size.',
                'Reduce training and test time: The computational execution speed inside the GPU is very sensitive to memory and bandwidth. Half-precision calculations halve the number of bytes accessed, reducing the time spent in some memory-constrained layers. Compared to single-precision, NVIDIA GPUs provide 8 times the half-precision throughput, which speeds up the computational speed of the math limit layer. For details, please refer to NVIDIA [Volta](https://devblogs.nvidia.com/inside-volta) Introduction to the characteristics of the architecture.'
            ]
        },
        {text:'The core of the hybrid precision training technique is to ensure that the values, parameters and gradients of the two types of tensions in the model training process are not excessively lost due to excessive information loss from the FP32 to the FP16 process. Joint details such as precision weight copy, loss quantization, and forced single-precision training for individual layers, please refer to ICLR2018 paper [MIXED PRECISION TRAINING](https://arxiv.org/abs/1710.03740v3)\\[2\\] for details. A complete hybrid precision training iterative process is divided into the following steps:\n'},
        {
            ul:['1, Generate a semi-precision copy of the weight;',
                '2, using half-precision values and weights for forward calculation;',
                '3, multiply the loss obtained by the scale factor S;',
                '4, using half-precision values, weights and their gradients for backpropagation propagation;',
                '5, multiply the weight gradient by the scale factor S;',
                '6, Use weight truncation and weight attenuation to selectively process weight gradients;',
                '7, Update the weight in the master copy to single precision.'
            ]
        },
        {text:'The effect of training computer vision model using mixed precision calculation technology is shown in the following table (extracted from the experimental results of the paper). The accuracy of three computer vision tasks in ImageNet classification, Faster-RCNN and SSD is not lower than single precision training, even slightly There is improvement.\n'},
        // {table_header:'不同卷积神经网络在ILSVRC12验证集上的Top-1精度。'},
        {
            table:{
                titles:['Model','FP32','Mixed Precision'],
                data:[['AlexNet','56.77%','56.93%'],['VGG-D','65.40%','65.43%'],['GoogleNet','68.33%','68.43%'],['Inception-v1','70.03%','70.02%'],['Resnet50','73.61%','73.75%']]
            }
        },
        // {table_header:'Faster-RCNN与SSD在混合精度训练时在PASCAL VOC 2007测试集上的mAP，比例因子S=8，Faster-RCNN在VOC 2007上训练，SSD在VOC 2007-2012上训练'},
        {
            table:{
                titles:['Model','FP32 baseline','Mixed Precision without loss-scaling','Mixed Precision with loss-scaling'],
                data:[['Faster-RCNN','69.1%','68.6%','69.7%'],['Multibox SSD','76.9%','--','77.1%']]
            }
        },
        // {text:'我们也在Pet上进行了一系列混合精度训练的实验，在此给出了不同任务在单精度与混合精度训练模式下的精度、内存与速度对比。\n' +
        //         '\n' +
        //         '姿态估计任务中，以`simple_R-50-1x64d-D3K4C256_256x192_adam_1x`为实验配置，混合精度训练与单精度训练的模型性能对比。\n'},
        {table_header:'batch_size=256'},
        {
            table:{
                titles:['GPU','mAP','Memory Usage','Iteration Time Lag','Data Time Lag'],
                data:[['1080Ti',71.6,'4,739MB','0.3002','0.0055'],['1080Ti-amp',71.7,'2,977MB','0.3002','0.0055'],['titan-xp','-','4,739MB','0.3002','0.0055'],['titan-xp-amp','-','4,739MB','0.3002','0.0055'],]
            }
        },
        {table_header:'batch_size=1024'},
        {
            table:{
                titles:['GPU','mAP','Memory Usage','Iteration Time Lag','Data Time Lag'],
                data:[['1080Ti',71.6,'4,739MB','0.3002','0.0055'],['1080Ti-amp',71.7,'2,977MB','0.3002','0.0055'],['titan-xp','-','4,739MB','0.3002','0.0055'],['titan-xp-amp','-','4,739MB','0.3002','0.0055'],]
            }
        },
        {text:'The SSD algorithm uses `ssd_VGG16_300x300_1x` as the experimental configuration, and the performance comparison between the hybrid precision training and the single precision training model.\n'},
        {table_header:'batch_size=64\n'},
        {
            table:{
                titles:['GPU','mAP','Memory Usage','Iteration Time Lag','Data Time Lag'],
                data:[['1080Ti',25.0,'3,305MB','0.3002','0.0055'],['1080Ti-amp','--','2,641MB','0.3420','0.0120'],['titan-xp','-','4,739MB','0.3002','0.0055'],['titan-xp-amp','-','4,739MB','0.3002','0.0055'],]
            }
        },
        {table_header:'batch_size=256\n'},
        {
            table:{
                titles:['GPU','mAP','Memory Usage','Iteration Time Lag','Data Time Lag'],
                data:[['1080Ti',25.0,'3,305MB','0.3002','0.0055'],['1080Ti-amp','--','2,977MB','0.3002','0.0120'],['titan-xp','-','4,739MB','0.3002','0.0055'],['titan-xp-amp','-','4,739MB','0.3002','0.0055'],]
            }
        },
        {part_title:'Use Cases'},
        {h3_title:'Optimizer and Learning Rate Scheduler'},
        {text:'Pet builds the optimizer by calling the `Optimizer` class and the `LearningRateScheduler` class during training. The code is as follows:\n' +
                '\n' +
                '```Python\n' +
                'from pet.utils.optimizer import Optimizer\n' +
                'from pet.utils.lr_scheduler import LearningRateScheduler\n' +
                '\n' +
                'optimizer = Optimizer(model, cfg.SOLVER, local_rank=args.local_rank).build()\n' +
                'scheduler = LearningRateScheduler(optimizer, cfg.SOLVER, start_iter=0, iter_per_epoch=cfg.TRAIN.ITER_PER_EPOCH,\n' +
                '                                  local_rank=args.local_rank)\n' +
                '```\n'},
        {h3_title:'Distributed Data Parallel'},
        {text:'Pet completes the distributed data parallelism of the model on multiple GPUs with the following code. Set `cfg.DEVICE` to `cuda` to ensure training using GPU, local_rank, world_size.\n' +
                '\n' +
                '```Python\n' +
                '    args.device = torch.device(cfg.DEVICE)\n' +
                '    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1\n' +
                '    args.distributed = num_gpus > 1\n' +
                '    if args.distributed:\n' +
                '        torch.cuda.set_device(args.local_rank)\n' +
                '        torch.distributed.init_process_group(\n' +
                '            backend="nccl", init_method="env://"\n' +
                '        )\n' +
                '        args.world_size = torch.distributed.get_world_size()\n' +
                '    else:\n' +
                '        assert args.device == \'cpu\'\n' +
                '        args.world_size = 1\n' +
                '    \n' +
                '    # Model Distributed\n' +
                '    if args.distributed:\n' +
                '        model = torch.nn.parallel.DistributedDataParallel(\n' +
                '            model, device_ids=[args.local_rank], output_device=args.local_rank\n' +
                '        )\n' +
                '```\n'},
        {h3_title:'Mixed Precision Calculation'},
        {text:'The following steps are required for mixed-precision training in classification, single-body pose estimation, and single-stage target detection tasks:\n'},
        {text:'After building the network model, we first initialized the various key settings in the hybrid precision training process using the [APEX] (https://github.com/nvidia/apex) driver provided by NVIDIA, which was developed by NVIDIA for PyTorch. It is mainly used to optimize the automatic mixed precision calculation and distributed training driver. APEX\'s [documentation] (https://nvidia.github.io/apex/) has detailed feature introduction. For the super parameters that control the mixing precision training, please refer to [cfg.SOLVER.AMP](https://github.com/BUPT-PRIV/Pet-dev/blob/e11ef696c92ea5e4cf30609fb67420a262c911ca/pet/cls/core/config.py) in the configuration system. The main contents include:\n'},
        {
            ul:[
                'Calculation accuracy used during training;；',
                'Whether to force the BatchNorm layer to use single precision calculations;',
                'Loss quantification, support for dynamic adjustment of quantitative strategies and static quantification strategies'
            ]
        },
        {text:'```Python\n' +
                '    from apex import amp\n' +
                '\n' +
                '    if cfg.SOLVER.AMP.ENABLED:\n' +
                '        # Create Amp for mixed precision training\n' +
                '        model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.SOLVER.AMP.OPT_LEVEL,\n' +
                '                                          keep_batchnorm_fp32=cfg.SOLVER.AMP.KEEP_BN_FP32,\n' +
                '                                          loss_scale=cfg.SOLVER.AMP.LOSS_SCALE)\n' +
                '```\n' +
                '\n' +
                'Pet uses the \'Distributed DataParallel` provided by APEX for distributed training in the case of mixed precision training.\n' +
                '\n' +
                '```Python\n' +
                '    from apex.parallel import DistributedDataParallel\n' +
                '\n' +
                '    # Model Distributed\n' +
                '    if args.distributed:\n' +
                '        if cfg.SOLVER.AMP.ENABLED:\n' +
                '            model = DistributedDataParallel(model)  # use apex.parallel\n' +
                '        else:\n' +
                '            model = torch.nn.parallel.DistributedDataParallel(\n' +
                '                model, device_ids=[args.local_rank], output_device=args.local_rank\n' +
                '            )\n' +
                '```\n'},
        {part_title:'Convenience and Regulation'},
        {text:'Based on Pet\'s set of normalized implementations for optimizing iterative operations, you can easily customize different optimizers and learning rate schedulers to meet your needs when training deep learning models. Using the optimizer and the learning rate scheduler, you get the following conveniences when training deep learning models:\n'},
        {
            ul:[
                'You can use the optimizer and the learning rate scheduler to train your network model in all the visual tasks supported by Pet. We provide some suggested optimization strategies and configurations based on different visual tasks and algorithms, due to differences in visual tasks. The resulting code differences will no longer be a problem for you.',
                'The Optimizer and Learning Rate Scheduler provides a rich optimization algorithm and a learning rate scheduling strategy. Here we support a truly valuable optimization iteration method that allows you to conduct deep learning research efficiently.'
            ]
        },
        {text:'When you need to extensively extend the optimization and iterative methods of deep learning models, you need to fully follow Pet\'s code implementation standards to improve the optimizer and learning rate scheduler. You are welcome to submit valuable methods and code to our github. We are very grateful for any contribution to the development of Pet.\n'},
        {part_title:'References'},
        {text:'\\[1\\] YuXin Wu and Kaiming He. Group normalization. CVPR 2018.\n'},
        {text:'\\[2\\] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaev, Ganesh Venkatesh, and others. 2017. MIXED PRECISION TRAINING. ICLR 2018.'}
    ],
    dataNav:[
        'Optimizer',
        // {

            // 'Optimizer':[
            //     'initialization',
            //     // {
            //     //     'build':[
            //     //         'get params list','get params'
            //     //     ]
            //     // },
            //     'build',
            //     'Optimization',
            // ]
        // },
        // {
            // 'Learning Rate Scheduler':[
            //     'init',
                // {
                //     'step':[
                //         'get lr','update learning rate',
                //     ]
                // }
                // 'step'
            // ]
        // },
        'Learning Rate Scheduler',
        {
            'Efficient Calculation':['Model Parallel', 'Mixed Precision Training']
        },
        'Use Cases',
        'Convenience and Regulation',
        'References'
    ]
};