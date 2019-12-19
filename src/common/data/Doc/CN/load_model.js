export let load_model_data = {
    key: 'load_model',
    dataSource: [
        {title:'模型加载与保存'},
        {text:"模型的加载与保存对网络训练十分重要，Pet定义了一个类`CheckPointer`用于相关功能的封装。在进行模型训练时，加载的模型分为之前已有的检查点模型、从其他任务中迁移过来的模型和进行随机初始化的模型，您可以设置`cfg.TRAIN.WEIGHTS`中的对应变量选择对应参数加载方式。模型加载与保存的处理内容包括了模型的网络参数还有模型训练所需要的优化器和学习率调度器。\n",className:'segmentation'},
        {part_title:'CheckPointer'},
        {text:"在您要进行模型参数加载和保存时，`CheckPointer`可以帮您将这一过程封装起来。在完成CheckPointer类的实例化之后，只需调用成员函数就能完成相应的模型加载或保存功能。完整代码请参考[$Pet/pet/utils/checkpointer.py](https://github.com/BUPT-PRIV/Pet/blob/master/pet/utils/checkpointer.py)。\n" +
                "\n" +
                "模型加载与保存的实现过程如下：\n"},
        {
            ul:[
                '完成该类的实例化，确认模型参数的加载方式；',
                '通过调用类中的成员函数进行模型参数、优化器、学习率调节器的更新；',
                '完成一次迭代后，将训练出的模型参数、优化器、学习率调节器设置作为最新一次的结果进行更新。'
            ]
        },
        {h3_title:'Init'},
        {text:"`CheckPointer`的初始化用于确定模型参数的加载方式，通过对`weights_path`和`resume`的初始化来实现，进行初始化需要传入`ckpt`、`weights_path`、`auto_resume`、`local_rank`等参数。\n"},
        {
            ul:[
                '`ckpt`：进行参数加载的模型所在的路径。',
                '`weights_path`：迁移参数的模型所在路径。',
                '`auto_resume`：作为是否使用参数加载检查点模型的标志变量，在为True时选择最近检查点模型的权重参数进行初始化。',
                '`local_rank`：在多GPU训练时使用，过滤多个GPU上模型的重复信息，仅输出`local_rank`所指定的GPU上的模型的相关信息。'
            ]
        },
        {text:"```Python\n" +
                "class CheckPointer(object):\n" +
                "    def __init__(self, ckpt, weights_path=None, auto_resume=True, local_rank=0):\n" +
                "        self.ckpt = ckpt\n" +
                "        self.weights_path = weights_path\n" +
                "        self.auto_resume = auto_resume\n" +
                "        self.local_rank = local_rank\n" +
                "\n" +
                "        self.mismatch_keys = set()\n" +
                "        self.resume = self.get_model_latest()\n" +
                "        if self.weights_path:\n" +
                "            self.checkpoint = self._load_file()\n" +
                "```\n" +
                "\n" +
                "`weights_path`和`resume`两个变量在执行模型加载功能函数`load_model`时被用到，这里介绍他们与不同加载方式的对应关系：\n"},
        {
            ul:[
                '`resume`为True：加载检查点模型；',
                '`resume`为False且`weights_path`为True：加载预训练模型；',
                '`resume`、`weights_path`均为`False`：加载直接随机初始化后的网络参数。'
            ]
        },
        {
            h4_block:[
                {h4_title:'Get_model_latest'},
                {text:"用于初始化`resume`，在初始化中被调用，通过确认类的`ckpt`成员变量下是否有目标模型`model_latest.pth`来赋予不同的值。\n"},
                {h4_title:'_Load_file'},
                {text:'用于将模型文件中的参数赋给`self.checkpoint`。\n'}
            ]
        },
        {h3_title:'load_model'},
        {text:"```Python\n" +
                "    def load_model(self, model, convert_conv1=False):\n" +
                "        if self.resume:\n" +
                "            weights_dict = self.checkpoint.pop('model')\n" +
                "            weights_dict = strip_prefix_if_present(weights_dict, prefix='module.')\n" +
                "            model_state_dict = model.state_dict()\n" +
                "            model_state_dict, self.mismatch_keys = align_and_update_state_dicts(model_state_dict, weights_dict,\n" +
                "                                                                                self.local_rank)\n" +
                "            model.load_state_dict(model_state_dict)\n" +
                "            logging_rank('Resuming from weights: {}.'.format(self.weights_path), local_rank=self.local_rank)\n" +
                "        else:\n" +
                "            if self.weights_path:\n" +
                "                weights_dict = self.checkpoint\n" +
                "                weights_dict = strip_prefix_if_present(weights_dict, prefix='module.')\n" +
                "                weights_dict = self.weight_mapping(weights_dict)    # only for pre-training\n" +
                "                if convert_conv1:   # only for pre-training\n" +
                "                    weights_dict = self.convert_conv1_rgb2bgr(weights_dict)\n" +
                "                model_state_dict = model.state_dict()\n" +
                "                model_state_dict, self.mismatch_keys = align_and_update_state_dicts(model_state_dict, weights_dict,\n" +
                "                                                                                    self.local_rank)\n" +
                "                model.load_state_dict(model_state_dict)\n" +
                "                logging_rank('Pre-training on weights: {}.'.format(self.weights_path), local_rank=self.local_rank)\n" +
                "            else:\n" +
                "                logging_rank('Training from scratch.', local_rank=self.local_rank)\n" +
                "        return model\n" +
                "```\n" +
                "\n" +
                '模型的三种参数加载方式在`load_model`成员函数中实现，这里调用了`weight_mapping`、`convert_conv1_rgb2bgr`等成员函数和其他外部函数，下面介绍函数中三种加载方式的实现过程。\n'},
        {text:'1、加载最近检查点的模型参数的执行过程\n'},
        {
          ul:[
              '1.`checkpoint`中的网络参数赋给`weights_dict`存储；',
              '2.使用`strip_prefix_if_present`函数去除`weights_dict`的参数前缀名；',
              '3.将模型参数中的优化器以外的参数组成字典返回给新的参数字典`model_state_dict`；',
              '4.使用`align_and_update_state_dicts`函数将参数映射到模型对应的参数字典`model_state_dict`中；',
              '5.将参数字典`model_state_dict`中的参数对应加载到模型；',
              '6.最后使用`logging_rank`函数打印模型参数的加载方式。',
          ]
        },
        {text:'2、加载预训练模型参数的执行过程:\n'},
        {
            ul:[
                '1.`checkpoint`中的参数赋给`weights_dict`存储；',
                '2.使用`strip_prefix_if_present`函数去除`weights_dict`的参数前缀名；',
                '3.使用`weight_mapping`将预训练模型中参数的前缀替换为Pet中定义模型参数的前缀；',
                '4.使用`convert_conv1_rgb2bgr`函数将卷积层的参数通道进行转换；',
                '5.使用`align_and_update_state_dicts`函数将参数映射到模型的参数字典；',
                '6.将参数字典`model_state_dict`中的参数对应加载到模型；',
                '7.最后使用`logging_rank`函数打印模型参数的加载方式。'
            ]
        },
        {text:'3、参数随机初始化的执行过程\n'},
        {ul:'1.使用`logging_rank`函数打印模型参数的加载方式。'},
        {h3_title:'load_optimizer'},
        {text:"在加载最近检查点的模型参数时，将其中的优化器参数取出，更新当前优化器。\n" +
                "\n" +
                "```\n" +
                "    def load_optimizer(self, optimizer):\n" +
                "        if self.resume:\n" +
                "            optimizer.load_state_dict(self.checkpoint.pop('optimizer'))\n" +
                "            logging_rank('Loading optimizer done.', local_rank=self.local_rank)\n" +
                "        else:\n" +
                "            logging_rank('Initializing optimizer done.', local_rank=self.local_rank)\n" +
                "        return optimizer\n" +
                "```\n"},
        {h3_title:'load_scheduler'},
        {text:"与`load_optimizer`功能相似，用于更新当前的学习率调节器。\n" +
                "\n" +
                "```Python\n" +
                "    def load_scheduler(self, scheduler):\n" +
                "        if self.resume:\n" +
                "            scheduler.load_state_dict(self.checkpoint.pop('scheduler'))\n" +
                "            logging_rank('Loading scheduler done.', local_rank=self.local_rank)\n" +
                "        else:\n" +
                "            logging_rank('Initializing scheduler done.', local_rank=self.local_rank)\n" +
                "        return scheduler\n" +
                "```\n"},
        {h3_title:'save'},
        {text:'定义了一个中间变量`save_dict`，用于保存模型本次训练的模型参数、优化器和学习率调节器设置，并将其存入成员变量`ckpt`路径下，打印模型的保存状态和位置。调用该函数需要传入`model`、`optimizer`、`scheduler`、`copy_latest`和`infix`等参数。\n' +
                '\n' +
                '```Python\n' +
                '    def save(self, model, optimizer=None, scheduler=None, copy_latest=True, infix=\'iter\'):\n' +
                '        save_dict = {\'model\': model.state_dict()}\n' +
                '        if optimizer is not None:\n' +
                '            save_dict[\'optimizer\'] = optimizer.state_dict()\n' +
                '        if scheduler is not None:\n' +
                '            save_dict[\'scheduler\'] = scheduler.state_dict()\n' +
                '\n' +
                '        torch.save(save_dict, os.path.join(self.ckpt, \'model_latest.pth\'))\n' +
                '        logg_sstr = \'Saving checkpoint done.\'\n' +
                '        if copy_latest and scheduler:\n' +
                '            shutil.copyfile(os.path.join(self.ckpt, \'model_latest.pth\'),\n' +
                '                            os.path.join(self.ckpt, \'model_{}{}.pth\'.format(infix, str(scheduler.iteration))))\n' +
                '            logg_sstr += \' And copy "model_latest.pth" to "model_{}{}.pth".\'.format(infix, str(scheduler.iteration))\n' +
                '        logging_rank(logg_sstr, local_rank=self.local_rank)\n' +
                '```\n'},
        {table:{
                titles:['成员变量','含义'],
                data:[["model","当前迭代下的模型参数"],["optimizer","当前迭代下的优化器设置"],["scheduler","当前迭代下的学习率调节器"],["copy_latest","是否进行模型参数文件复制操作的标志变量"],["infix","当前迭代数"],]
            }
            , className:'table_2'},

        {part_title: '便利与规范'},
        {text:"基于以上标准，Pet对模型参数加载与保存系统进行了封装，在训练和测试网络时获得以下便利：\n"},
        {
            ul:[
                '便捷高效地实现多种模型参数加载方式的转换和模型参数的保存，加快工程开发周期。',
                '简单方便地使用模型参数加载与保存系统，来获取包括学习率调度器、优化器的设置参数。'
            ]
        },
        {text:"如果您准备对我们的模型参数加载与保存系统进行拓展，您需要完全遵循Pet的代码实现标准来进行修改和添加，欢迎您将有价值的代码和意见提交到我们的github，我们十分感谢任何对于发展Pet有益的贡献。\n"}
    ],
    dataNav:[
        {
            'CheckPointer':[
                'Init',
                'load_model',
                'load_optimizer',
                'load_scheduler',
                'save'
            ]
        },
        '便利与规范'
    ]
}