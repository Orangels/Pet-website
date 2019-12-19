# 迭代优化

迭代优化是训练深度学习模型的核心内容，Pet对深度学习模型的优化迭代操作的实现方法设置了一套标准的实现，将深度学习模型的训练与优化操作归纳为优化器和学习率调度器两项组合操作，并且使用了混合精度训练与PyTorch提供的分布式数据并行方法提高模型训练的效率。在Pet的代码实现中，优化器和学习率调度器具体对应`Optimizer`和`LearningRateScheduler`两个基本Python操作类，两个Python类会在整个训练的过程中一直被用于指导模型的优化。可以使用配置系统中`SOLVER`模块的设置来构建优化器和学习率调度器，对模型训练过程中的优化算法、学习率变化以及参数差异化优化策略进行系统设置。

[优化器](#优化器)

>[init](#init)

>[build](#build)

>>[get_params_list](#get_params_list)

>>[get_params](#get_params)

>[优化算法](#优化算法)

[学习率调度器](#学习率调度器)

>[init](#init)

>[step](#step)

>>[get_lr](#get_lr)

>>[update_learning_rate](#update_learning_rate)

[高效计算](#高效计算)

>[模型并行](#模型并行)

>[混合精度训练](#混合精度训练)

[使用案例](#使用案例)

[便利与规范](#便利与规范)

[参考文献](#参考文献)


## 优化器

当您在完成网络模型的构建之后，优化器可以帮助您对网络结构中不同种类的参数的学习率、权重衰减因子以及学习率倍率倍率进行差异化设置，同时还提供一些主流的优化算法，您可以根据您不同的训练需求来配置优化器，优化器的完整代码请参考[$/pet/utils/optimizer.py]()。

优化器的实现规则与流程如下：

* 优化器对构建好的模型进行参数解析，根据需要对模型参数进行归类，不同类型的参数会被分配以不同的权重衰减和学习率倍率；

* 将归类和差异化配置之后的模型参数送入torch提供的优化算法，完成优化器的配置。

当您需要对网络模型中的其他参数进行差异化优化设置，或者您需要使用新的优化算法时，您需要在优化器内遵循以上的代码实现标准，将您的改进加入Pet。

### 初始化

优化器在Pet中对应`Optimizer`这一具体的Python类，接收`model`、`solver`作为输入，`Optimizer`的主要成员函数包括`get_params_list`、`get_params`以及`build`。在了解`Optimizer`的功能函数之前，我们首先对`Optimizer`类进行初始化：

```Python
class Optimizer(object):
    def __init__(self, model, solver, local_rank=0):
        self.model = model
        self.solver = solver
        self.local_rank = local_rank

        self.bias_params_list = []
        self.gn_params_list = []
        self.nonbias_params_list = []

        self.params = []
        self.gn_param_nameset = self.get_gn_param_nameset()
```

`model`是网络结构以及网络参数的集合，Pet将`model`中包含的参数分为`bias_params_list`、`gn_params_list`、`nonbias_params_list`三大类：

* `bias_params_list`：卷积、全连接操作中的偏置

* `gn_params_list`：[Group Normalization]()操作对应的xxxxx参数

* `nonbias_params_list`：卷积、全连接操作中的权重

在目前主流的深度学习算法中，这三大类参数在训练的过程中需要被分配不同的权重衰减因子（weight_decay）和学习率倍率（lr_scale），详情见`get_params`。


### build

解析模型参数解析、差异化设置参数优化策略、选择优化算法。

```Python
    def build(self):
        assert self.solver.OPTIMIZER in ['SGD', 'RMSPROP', 'ADAM']
        self.get_params_list()
        self.get_params()

        if self.solver.OPTIMIZER == 'SGD':
            optimizer = torch.optim.SGD(
                self.params,
                momentum=self.solver.MOMENTUM
            )
        elif self.solver.OPTIMIZER == 'RMSPROP':
            optimizer = torch.optim.RMSprop(
                self.params,
                momentum=self.solver.MOMENTUM
            )
        elif self.solver.OPTIMIZER == 'ADAM':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.solver.BASE_LR
            )
        else:
            optimizer = None
        return optimizer

```


#### get_params_list

解析了`model`中各个操作层所包含的参数，将权重、偏置、gn三大类参数分别进行打包。

```Python
    def get_params_list(self):
        for key, value in self.model.named_parameters():
            if value.requires_grad:
                if 'bias' in key:
                    self.bias_params_list.append(value)
                elif key in self.gn_param_nameset:
                    self.gn_params_list.append(value)
                else:
                    self.nonbias_params_list.append(value)
            else:
                logging_rank('{} does not need grad.'.format(key), local_rank=self.local_rank)
```


#### get_params

对三大类参数的权重衰减因子和学习率倍率进行了差异化设置。

```Python
    def get_params(self):
        self.params += [
            {'params': self.nonbias_params_list,
             'lr': 0,
             'weight_decay': self.solver.WEIGHT_DECAY,
             'lr_scale': 1},
            {'params': self.bias_params_list,
             'lr': 0 * (self.solver.BIAS_DOUBLE_LR + 1),
             'weight_decay': self.solver.WEIGHT_DECAY if self.solver.BIAS_WEIGHT_DECAY else 0,
             'lr_scale': self.solver.BIAS_DOUBLE_LR + 1},
            {'params': self.gn_params_list,
             'lr': 0,
             'weight_decay': self.solver.WEIGHT_DECAY_GN,
             'lr_scale': 1}
        ]
```

* 卷积、全连接层偏置：通常情况下，偏置参数的学习率倍率是权重学习倍率的2倍，且不需要权重衰减，您可以根据您的需要对偏置进行权重衰减。

* gn参数：gn操作的参数默认不进行权重衰减，您可以根据您的需要开启权重衰减。

在此Pet设置模型中所有参数的学习率为0，除偏置外所有参数的学习率倍率为1，这只是一个简单的学习率初始化，在模型训练过程的每一次迭代中，学习率调度器会根据学习率调度策略对所有参数的学习率进行调整。

当您需要对网络中的某些特定参数进行差异化优化设置时，例如使用[Cascade-RCNN]()算法训练`rcnn`模型时，三个阶段的`RCNN`网络的权重和偏置分贝被赋予不同的学习率倍率，您可以在优化器中通过参数索引的形式尽心该操作，详情请见[教程]()。


### 优化算法

Pet提供了**随机梯度下降**（SGD）、**均方根支持**（RMSPROP）和**自适应矩估计**（ADAM）三种优化算法，可以满足绝大部分卷积神经网络的优化。`get_params`将模型中所有参数打包，在此Pet调用`torch`提供的优化算法接口，将所有参数和动量因子作为参数输入优化器，代码实现见`build`函数。


## 学习率调度器

在优化器构建完毕后，还需要构建学习率调度器，学习率调度器会根据您在配置系统中`SOLVER`部分设定的学习率变化策略在训练过程的每一次迭代中计算新的基础学习率，并对模型中的不同参数调整其差异化学习率，优化器的完整代码请参考[$/pet/utils/lr_scheduler.py]()。

学习率调度器的实现规则与流程如下：

* 根据学习率变化策略与当前迭代数，计算当前这一次迭代所对应的学习率；

* 根据优化器内的差异化设置，将学习率乘以不同的倍率赋值给不同的参数。

当您需要使用其他的学习率变化策略时，您需要在学习率调度器内遵循以上的代码实现标准，将您的改进加入Pet。


### 初始化

学习率调度器在Pet中对应`LearningRateScheduler`这一具体的Python类，接收`optimizer`和`solver`作为输入，`LearningRateScheduler`的主要成员函数包括`get_lr`、`update_learning_rate`以及`step`。在了解`LearningRateScheduler`的功能函数之前，我们首先对`Optimizer`类进行初始化：

```Python
class LearningRateScheduler(object):
    def __init__(self, optimizer, solver, start_iter=1, iter_per_epoch=-1, local_rank=0):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer

        self.solver = solver
        assert self.solver.LR_POLICY in ['STEP', 'COSINE', 'POLY']
        assert self.solver.WARM_UP_METHOD in ['CONSTANT', 'LINEAR']
        self.base_lr = self.solver.BASE_LR
        self.new_lr = self.base_lr

        self.iteration = start_iter
        self.iter_per_epoch = iter_per_epoch
        self.local_rank = local_rank

        if 'MAX_ITER' in self.solver:
            self.max_iter = self.solver.MAX_ITER
            self.warm_up_iters = self.solver.WARM_UP_ITERS
            self.steps = self.solver.STEPS  # only useful for step policy
        else:
            assert self.iter_per_epoch > 0  # need to specify the iter_per_epoch
            self.conver_epoch2iter()
```

* 初始化学习率调度器的第一步是要先检查`optimizer`是否具有一个真正的优化器结构体，这是Pet的一种异常预警机制，可以在您需要扩展新的代码时，提醒您在训练脚本中可能遗忘或错误地定义优化器，这种预警机制被广泛运用在目前的深度学习平台和算法工程中。

* 在获取`optimizer`和`solver`并将它们设置为学习率调度器的成员变量之后，还需要通过Pet的异常预警机制检查您在配置系统的`solver`模块中配置的字段是否是Pet所支的持学习率衰减和预热策略。

* 学习率调度器的主要全局成员变量如下表所示，这些全局成员变量将在整个深度学习模型训练过程中被用来调整每一次迭代的学习率：

| 成员变量       | 含义                        |
| :-----------: | :------------------------: |
| base_lr       | 基础学习率                   |
| new_Lr        | 当前学习率                   |
| iteration     | 当前迭代数                   |
| max_iter      | 最大迭代数                   |
| warm_up_iters | 用于学习率预热的迭代数         |
| steps         | 阶段下降策略中的学习率下降迭代数 | 

NOTE

#############

尽管Pet在记录日志时根据不同的视觉任务支持iter与epoch两种迭代计数单位，但是在学习率调度器内只使用迭代数作为调度学习的率的计数规则。

Rcnn任务使用迭代数作为日志计数单位，但对于姿态分析、单阶段目标检测、图像分类、图像分割等训练全数据迭代次数较多的计算机视觉任务，学习率调度器在初始化时将全数据迭代数通过`conver_epoch2iter`函数转化成最大迭代数，通过检查迭代次数来调整学习率。

详细代码请见[conver_epoch2iter]()

#############


### step

在对学习率调度器进行初始化之后，需要构建学习率调度器在迭代过程中的成员函数`step`，通过`step`函数可以在每一次迭代中，根据学习率优化策略以及当前迭代数计算学习率，并将当前学习率分配给网络模型中不同的参数。

```Python
    def step(self, cur_iter=None):
        if cur_iter is None:
            cur_iter = self.iteration + 1
        self.iteration = cur_iter

        # update learning rate
        self.new_lr = self.get_lr()
        self.update_learning_rate()
```

#### get_lr

根据配置系统中设置的学习率变化策略，在每一次迭代中计算当前的学习学习率，Pet将模型训练过程中学习率变化分为两个阶段：预热阶段和下降阶段。

##### 学习率预热策略(Learning rate warming up)

Pet将学习率预热策略收纳于学习率调度器中，提供了**连续**（CONSTANT）和**线性**（LINEAR）两种学习率预热策略。

在当前深度学习模型的训练过程中，批量优化技术已经成为一种通用的训练方法，但是小批量的数据不足以代表整个用于训练的数据集的统计分布，当学习率设置不合理时，模型的优化方向可能并不是全局最优，这可能导致模型在迭代优化过程中出现局部最优或者是不收敛的情况。学习率预热策略在训练的开始阶段将学习率保持在一个比较小的水平，并在最大预热迭代次数之内使学习率缓慢增长，保证模型在优化的最开始不会偏向错误的方向。


##### 学习率下降策略

Pet为深度卷积神经网络模型的训练和优化提供了**阶段下降**、**余弦下降**、**复数下降**三种学习率下降策略，他们在配置系统的`SOLVER`模块中对应的字段分别是`STEP`、`COSINE`和`POLY`，在学习率预热之后，学习率变成`SOLVER`中设置的基础学习率，并随着迭代次数的增长按照策略下降。

```Python
    def get_lr(self):
        new_lr = self.base_lr
        if self.iteration <= self.warm_up_iters:  # warm up
            if self.solver.WARM_UP_METHOD == 'CONSTANT':
                warmup_factor = self.solver.WARM_UP_FACTOR
            elif self.solver.WARM_UP_METHOD == 'LINEAR':
                alpha = self.iteration / self.warm_up_iters
                warmup_factor = self.solver.WARM_UP_FACTOR * (1 - alpha) + alpha
            else:
                raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(self.solver.WARM_UP_METHOD))
            new_lr = self.base_lr * warmup_factor
        elif self.iteration > self.warm_up_iters:
            if self.solver.LR_POLICY == 'STEP':
                new_lr = self.base_lr * self.solver.GAMMA ** bisect_right(self.steps, self.iteration)
            elif self.solver.LR_POLICY == 'COSINE':
                actual_iter = self.max_iter - self.warm_up_iters  # except warm up
                new_lr = 0.5 * self.base_lr * (
                    np.cos((self.iteration - self.warm_up_iters - 1) * np.pi / actual_iter) + 1.0)
            elif self.solver.LR_POLICY == 'POLY':
                actual_iter = self.max_iter - self.warm_up_iters  # except warm up
                new_lr = self.base_lr * (
                    (1. - float(self.iteration - self.warm_up_iters - 1) / actual_iter) ** self.solver.LR_POW)
            else:
                raise KeyError('Unknown SOLVER.LR_POLICY: {}'.format(self.solver.LR_POLICY))
        return new_lr
```


#### update_learning_rate

更新模型中所有参数的学习率，偏置、权重、gn参数因为不同的学习率倍率获得不同的学习率。

```Python
    def update_learning_rate(self):
        """Update learning rate
        """
        cur_lr = self.optimizer.param_groups[0]['lr']
        if cur_lr != self.new_lr:
            ratio = _get_lr_change_ratio(cur_lr, self.new_lr)
            if ratio > self.solver.LOG_LR_CHANGE_THRESHOLD and self.new_lr >= 1e-7:
                logging_rank('Changing learning rate {:.6f} -> {:.6f}'.format(cur_lr, self.new_lr),
                             local_rank=self.local_rank)
            # Update learning rate, note that different parameter may have different learning rate
            for ind, param_group in enumerate(self.optimizer.param_groups):
                if 'lr_scale' in param_group:
                    lr_scale = param_group['lr_scale']
                else:
                    lr_scale = 1
                param_group['lr'] = self.new_lr * lr_scale
```


## 高效计算

深度学习模型以庞大的数据规模作为基础，学习能够普遍表达目标特征的网络参数，这一过程十分漫长，一个深度学习模型训练的时间动辄十数个小时，某些大型模型甚至需要数天的时间才能训练完毕。已经有许多研究致力于提升深度学习模型的计算效率，各大深度学习框架的开发团队也不断地推出加速计算方法。Pet持续跟进PyTorch提供的最新接口与方法，将分布式数据并行以及混合精度训练技术应用在深度学习研究中。


### 分布式数据并行（DistributedDataParallel）

Pet使用了基于`torch.distributed`工具包的分布式数据并行模型来训练所有计算机视觉模型，在模型层级上实现分布式数据并行，关于torch的分布式工具包，请参阅[torch.distributed](https://pytorch.org/docs/stable/distributed.html#distributed-basics)。`DistributedDataParallel`通过在批次维度上将数据切片并分发到不同设备上来实现模型的并行计算。模型被赋值按到每个主机或者GPU，每个模型备份在前向计算过程中负责处理部分输入数据，在反向传播过程中，多个设备上的梯度将取平均值。

`Distributed`有单线程多GPU和多线程单GPU两种使用方法。单线程多GPU模式在每个主机/节点上生成一个线程，该线程调用所有的GPU计算资源；多线程单GPU模式在每个主机/节点上生成多个线程，每个线程在单个GPU上运行。Pet使用后者作为提高模型计算效率的方式，这是目前使用PyTorch进行数据并行训练的最快方法，适用于单节点（多GPU）和多节点数据并行训练。对于单节点多GPU数据并行训练，它被证明比PyTorch提供的另外一种数据并行训练工具`torch.nn.DataParallel`快得多。

############################
NOTE

Pet使用NCCL作为跨GPU通信框架，在多线程单GPU模式下使用NCCL可以达到最快的计算速度，nccl同样可以支持单节点多GPU模式。使用`torch.nn.parallel.DistributedDateParallel`需要注意以下几点：

* 网络在进行分布式部署之前，其中所有的参数必须被注册，在进行分布式部署之后模型中新增加的参数将无效，同样也不能移除任何参数。

* 当使用了分布式并行工具之后，在数据载入时也需要使用多线程数据载入器，详情请见[数据载入]()。

* 当使用混合精度训练技术时，Pet采用英伟达提供的APEX驱动来加快计算。
############################


### 混合精度训练

深度学习在人工智能领域的进步对计算资源的需求越来越庞大，功能强大的神经网络需要GPU、TPU等处理器强大算力的支持，然而在实际使用中，设备的功耗、算力等实际条件处处限制了神经网络的体量，大型神经网络的训练也非常耗时。轻量化网络、模型剪枝等模型量化技术一直是研究的热点，但对模型体量的减弱必然在一定程度上对深度神经网络的精度带来损失，为了在不损失精度的前提下加快神经网络的训练与测试速度，同时减小模型容量，混合精度训练技术于2018年被百度与英伟达联合推出。

混合精度训练技术是要在不损失模型精度的情况下，使用FP32与FP16两种计算精度共同训练模型，最后将模型参数以半精度浮点数存储，在英伟达[官方网站](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/)上您可以了解到更多关于混合精度训练的详细信息。

在混合精度训练技术问世之前，深度神经网络一直是以32比特浮点数（float32）来进行前向与反向计算，被称为单精度浮点计算（FP32），双精度浮点计算（FP64）也常常被使用。CUDA编程语言支持半精度计算的FP16数据类型，而英伟达GPU的架构很早就提供了FP16的数值表示，在混合精度训练技术出现之后，神经网络的计算效率得到了巨大的提升，使用混合精度训练技术将在两方面收获极大的便利：

* 对内存空间需求的大幅减少：相比单精度与双精度，半精度浮点格式（FP16）使用16比特来表达数值，节省的内存空间可以用来训练体量更加庞大的神经网络模型，或者使用更大的批量数（batch size）。

* 缩短训练与测试时间：GPU内部的计算执行速度对内存以及带宽非常敏感，半精度计算将访问的字节数减半，从而减少了在某些内存限制层中花费的时间。与单精度相比，英伟达GPU提供了8倍的半精度计算吞吐量，从而加速了数学限制层的计算速度，详情可参考英伟达[Volta](https://devblogs.nvidia.com/inside-volta/)架构的特性介绍。

混合精度训练技术的核心是保证模型训练过程中的数值、参数以及两种张亮对应的梯度从FP32量化到FP16的过程中不会形成过大的信息丢失而导致精度损失，这其中涉及了单精度权重副本、损失量化和个别层的强制单精度训练等关节细节，详情请参考ICLR2018论文[MIXED PRECISION TRAINING](https://arxiv.org/abs/1710.03740v3)。一个完整的混合精度训练迭代过程分为以下几个步骤：

* 1、生成权重的半精度副本；

* 2、使用半精度的数值和权重进行前向计算；

* 3、将得到的损失乘以比例因子S；

* 4、使用半精度的数值、权重和它们的梯度进行反向传播传播；

* 5、将权重梯度乘以除以比例因子S；

* 6、使用权重截断、权重衰减有选择地处理权重梯度；

* 7、更新主副本中的权重为单精度。

使用混合精度计算技术训练计算机视觉模型得到的效果如下表所示（摘自论文的实验结果），在ImageNet分类、Faster-RCNN、SSD三个计算机视觉任务的精度均不低于单精度训练，甚至略有提升。

不同卷积神经网络在ILSVRC12验证集上的Top-1精度。

| Model        | FP32   | Mixed Precision |
| :----------: | :----: | :-------------: |
| AlexNet      | 56.77% | 56.93%          |
| VGG-D        | 65.40% | 65.43%          |
| GoogleNet    | 68.33% | 68.43%          |
| Inception-v1 | 70.03% | 70.02%          |
| Resnet50     | 73.61% | 73.75%          |

Faster-RCNN与SSD在混合精度训练时在PASCAL VOC 2007测试集上的mAP，比例因子S=8，Faster-RCNN在VOC 2007上训练，SSD在VOC 2007-2012上训练。

| Method       | FP32 baseline | Mixed Precision without loss-scaling | Mixed Precision with loss-scaling |
| :----------: | :-----------: | :----------------------------------: | :-------------------------------: |
| Faster-RCNN  | 69.1%         | 68.6%                                | 69.7%                             |
| Multibox SSD | 76.9%         | -                                    | 77.1%                             |


我们也在Pet上进行了一系列混合精度训练的实验，在此给出了不同任务在单精度与混合精度训练模式下的精度、内存与速度对比。

姿态估计任务中，以`simple_R-50-1x64d-D3K4C256_256x192_adam_1x`为实验配置，混合精度训练与单精度训练的模型性能对比。

``batch_size=256``

|                    | mAP        | 内存占用    | 迭代耗时      | 数据耗时     |
| :----------------: |:----------:|:----------:|:-----------:|:-----------:|
| 1080Ti             | 71.6       | 4,739MB    | 0.3002      | 0.0055      |
| 1080Ti-amp         | 71.7       | 2,977MB    | 0.3002      | 0.0055      |
| titan-xp           | --         | 4,739MB    | 0.3002      | 0.0055      |
| titan-xp-amp       | --         | 4,739MB    | 0.3002      | 0.0055      |

``batch_size=1024``

|                    | mAP        | 内存占用    | 迭代耗时      | 数据耗时     |
|--------------------|:----------:|:----------:|:-----------:|:-----------:|
| 1080Ti             | 71.6       | 4,739MB    | 0.3002      | 0.0055      |
| 1080Ti-amp         | 71.7       | 2,977MB    | 0.3002      | 0.0055      |
| titan-xp           | --         | 4,739MB    | 0.3002      | 0.0055      |
| titan-xp-amp       | --         | 4,739MB    | 0.3002      | 0.0055      |

SSD算法以`ssd_VGG16_300x300_1x`为实验配置，混合精度训练与单精度训练的模型性能对比。

``batch_size=64``

|                    | mAP        | 内存占用    | 迭代耗时      | 数据耗时     |
|--------------------|:----------:|:----------:|:-----------:|:-----------:|
| 1080Ti             | 25.0       | 3,305MB    | 0.3002      | 0.0055      |
| 1080Ti-amp         | --         | 2,641MB    | 0.3420      | 0.0120      |
| titan-xp           | --         | 4,739MB    | 0.3002      | 0.0055      |
| titan-xp-amp       | --         | 4,739MB    | 0.3002      | 0.0055      |

``batch_size=256``

|                    | mAP        | 内存占用    | 迭代耗时      | 数据耗时     |
|--------------------|:----------:|:----------:|:-----------:|:-----------:|
| 1080Ti             | 25.0       | 3,305MB    | 0.3002      | 0.0055      |
| 1080Ti-amp         | --         | 2,977MB    | 0.3002      | 0.0055      |
| titan-xp           | --         | 4,739MB    | 0.3002      | 0.0055      |
| titan-xp-amp       | --         | 4,739MB    | 0.3002      | 0.0055      |


## 使用案例


### 优化器与学习率调度器

Pet在训练过程中通过调用`Optimizer`类与`LearningRateScheduler`类来构建优化器，代码如下：

```Python
from pet.utils.optimizer import Optimizer
from pet.utils.lr_scheduler import LearningRateScheduler

optimizer = Optimizer(model, cfg.SOLVER, local_rank=args.local_rank).build()
scheduler = LearningRateScheduler(optimizer, cfg.SOLVER, start_iter=0, iter_per_epoch=cfg.TRAIN.ITER_PER_EPOCH,
                                  local_rank=args.local_rank)
```


### 分布式数据并行

Pet通过如下代码完成模型在多个GPU上的分布式数据并行。设置`cfg.DEVICE`为`cuda`保证使用GPU进行训练，local_rank、world_size。

```Python
    args.device = torch.device(cfg.DEVICE)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        args.world_size = torch.distributed.get_world_size()
    else:
        args.world_size = 1
        args.local_rank = 0
        cfg.NUM_GPUS = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) if cfg.DEVICE == 'cuda' else 1
        cfg.TRAIN.LOADER_THREADS *= cfg.NUM_GPUS
        cfg.TEST.LOADER_THREADS *= cfg.NUM_GPUS
        cfg.TEST.IMS_PER_GPU *= cfg.NUM_GPUS
    
    # Model Distributed
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
```


### 混合精度计算

在分类、单人体姿态估计和单阶段目标检测任务的混合精度训练，需要下面几个步骤：

在构建好网络模型之后，我们首先对混合精度训练过程中的各种关键设置使用英伟达提供的[APEX](https://github.com/nvidia/apex)驱动进行初始化，APEX是英伟达为PyTorch开发的主要用于优化自动混合精度计算与分布式训练的驱动，APEX的[文档](https://nvidia.github.io/apex/)中有详细的特性功能介绍。控制进行混合精度训练的超参数请见配置系统中的[cfg.SOLVER.AMP]()，主要内容包括：

* 训练过程中使用的计算精度；

* 是否强制BatchNorm层使用单精度计算；

* 损失量化的方式，支持动态调整量化策略以及静态量化策略

```Python
    from apex import amp

    if cfg.SOLVER.AMP.ENABLED:
        # Create Amp for mixed precision training
        model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.SOLVER.AMP.OPT_LEVEL,
                                          keep_batchnorm_fp32=cfg.SOLVER.AMP.KEEP_BN_FP32,
                                          loss_scale=cfg.SOLVER.AMP.LOSS_SCALE)
```

Pet在混合精度训练的情况下使用APEX提供的`DistributedDataParallel`进行分布式训练。

```Python
    from apex.parallel import DistributedDataParallel

    # Model Distributed
    if args.distributed:
        if cfg.SOLVER.AMP.ENABLED:
            model = DistributedDataParallel(model)  # use apex.parallel
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank
            )
```

## 便利与规范

基于Pet对优化迭代操作所设置的一系列规范化实现，可以方便地定制出不同的优化器与学习率调度器来满足您训练深度学习模型时的需求。使用优化器与学习率调度器，您将在训练深度学习模型时获得以下便利：

* 您可以在Pet所支持的所有视觉任务中使用优化器和学习率调度器来训练您的网络模型，我们根据不同的视觉任务与算法提供了一些建议性优化策略与配置，因视觉任务的差异而导致的代码差异将不再是您的困扰。

* 优化器和学习率调度器提供了丰富的优化算法以及学习率调度策略，在此我们支持真正有价值的优化迭代方法，让您可以高效率地进行深度学习研究。

当您需要对深度学习模型的优化与迭代方法进行丰富拓展时，您需要完全遵循Pet的代码实现标准来改进优化器与学习率调度器，欢迎您将有价值的方法和代码提交到我们的github，我们十分感谢任何对于发展Pet有益的贡献。


## 参考文献

[1] YuXin Wu and Kaiming He. Group normalization. CVPR 2018.

[2] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich
Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaev, Ganesh
Venkatesh, and others. 2017. MIXED PRECISION TRAINING. ICLR 2018.
