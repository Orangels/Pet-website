export let log_system_data = {
    key:'log_system',
    dataSource: [
        {title:'日志系统'},
        {text:'日志模块是模型训练与测试阶段的辅助模块，它可以在控制台中输出信息以帮助您在训练与测试模型时能够监控模型的学习状态。Pet将这一功能封装成一个通用的模块，包括主要的`TrianingLogger`和`TestingLogger`两个Python类以及两个辅助类`SmoothedValue`和`AverageMeter`。您可以在训练或测试网络时将对应的类实例化，用于状态的输出。\n'},
        {part_title:'TrainingLogger'},
        {text:'在搭建网络模型的训练流程时，可以实例化此类，调用类中方法，实现控制台信息的输出，来监控训练过程。\n'},
        {h3_title:'初始化'},
        {text:'在训练脚本中使用的是`TrainingLogger`这一类，该类接收`cfg_filename`、`scheduler`、`log_period`作为参数，主要成员函数包括`get_stats`、`update_stats`、`log_stats`以及多个记录时间所用的方法，在实例化该类时，首先进行初始化：\n' +
                '\n' +
                '```Python\n' +
                '    class TrainingLogger(object):\n' +
                '        def __init__(self, cfg_filename, scheduler=None, log_period=20):\n' +
                '            self.cfg_filename = cfg_filename\n' +
                '            self.scheduler = scheduler\n' +
                '            self.log_period = log_period\n' +
                '\n' +
                '            self.data_timer = Timer()\n' +
                '            self.iter_timer = Timer()\n' +
                '\n' +
                '            def create_smoothed_value():\n' +
                '                return SmoothedValue(self.log_period)\n' +
                '\n' +
                '            self.smoothed_losses = defaultdict(create_smoothed_value)\n' +
                '            self.smoothed_metrics = defaultdict(create_smoothed_value)\n' +
                '            self.smoothed_total_loss = SmoothedValue(self.log_period)\n' +
                '```\n'},
        {
            ul:['`scheduler`是学习率优化方式类，包含了迭代次数，全数据训练次数等等参数',
                '`log_period`是打印信息的迭代间隔数，指的是每过`log_period`个迭代之后打印一次训练状态',
                '`SmoothedValue`是将loss进行平滑处理的一个辅助类，能够返回中位数、平均数等'
            ]
        },
        {text:'`TrainingLogger`类实例化后主要使用的是`update_stats`和`log_stats`，`get_stats`则是用于前面两个方法中。\n'},
        {h3_title:'get_stats'},
        {text:'```Python\n' +
                '        def get_stats(self, cur_iter, lr):\n' +
                '            eta_seconds = self.iter_timer.average_time * (self.scheduler.max_iter - cur_iter)\n' +
                '            eta = str(datetime.timedelta(seconds=int(eta_seconds)))\n' +
                '            stats = OrderedDict(cfg_filename=self.cfg_filename)\n' +
                '            if self.scheduler.iter_per_epoch == -1:\n' +
                '                stats[\'iter\'] = cur_iter\n' +
                '                stats[\'max_iter\'] = self.scheduler.max_iter\n' +
                '            else:\n' +
                '                stats[\'epoch\'] = cur_iter // self.scheduler.iter_per_epoch + 1\n' +
                '                stats[\'max_epoch\'] = self.scheduler.max_iter // self.scheduler.iter_per_epoch\n' +
                '                stats[\'iter\'] = cur_iter % self.scheduler.iter_per_epoch\n' +
                '                stats[\'max_iter\'] = self.scheduler.iter_per_epoch\n' +
                '            stats[\'lr\'] = lr\n' +
                '            stats[\'eta\'] = eta\n' +
                '            stats[\'data_time\'] = self.data_timer.average_time\n' +
                '            stats[\'data_time_d\'] = self.data_timer.diff\n' +
                '            stats[\'iter_time\'] = self.iter_timer.average_time\n' +
                '            stats[\'iter_time_d\'] = self.iter_timer.diff\n' +
                '            stats[\'total_loss\'] = self.smoothed_total_loss.avg\n' +
                '            stats[\'total_loss_m\'] = self.smoothed_total_loss.median\n' +
                '\n' +
                '            metrics = []\n' +
                '            for k, v in self.smoothed_metrics.items():\n' +
                '                metrics.append((k, str(v.median) + \' \' + str(v.avg)))\n' +
                '            stats[\'metrics\'] = OrderedDict(metrics)\n' +
                '\n' +
                '            losses = []\n' +
                '            for k, v in self.smoothed_losses.items():\n' +
                '                losses.append((k, str(v.median) + \' \' + str(v.avg)))\n' +
                '            stats[\'losses\'] = OrderedDict(losses)\n' +
                '\n' +
                '            return stats\n' +
                '```\n' +
                '\n' +
                '在训练阶段，输出`outputs`以字典结构呈现，包含了`metrics`和`losses`两个子字典，在该函数中将其加入`stats`列表中。\n'},
        {ul:'`stats`中的关键字如下表所示：'},
        {table:{
                titles:['关键字','值'],
                data:[["epoch","当前所处的全数据迭代数"],["max_epoch","最大全数据迭代数"],["iter","当前所处的迭代数"],["max_iter","最大迭代数"],["lr","学习率"],["eta","预计剩余迭代需要的时间"],["data_time","处理数据时用的平均时间"],["data_time_d","处理数据时用的总时间"],["iter_time","迭代周期中每个迭代用的平均时间"],["iter_time_d","迭代周期用的总时间"],["total_loss","损失值的平均数"],["total_loss_m","损失值的中位数"],["metrics","输出的任务评价指标，例如\"classes\":[top1, top5]"],["losses","任务中的各个损失值，同样以字典的形式存储\n"]]
            }
            , className:'table_1'},
        {h3_title:'update_stats'},
        {text:'```Python\n' +
                '    def update_stats(self, output, distributed=True, world_size=1):\n' +
                '        total_loss = 0\n' +
                '        for k, loss in output[\'losses\'].items():\n' +
                '            total_loss += loss\n' +
                '            loss_data = loss.data\n' +
                '            if distributed:\n' +
                '                loss_data = reduce_tensor(loss_data, world_size=world_size)\n' +
                '            self.smoothed_losses[k].update(loss_data)\n' +
                '        output[\'total_loss\'] = total_loss  # add the total loss for back propagation\n' +
                '        self.smoothed_total_loss.update(total_loss.data)\n' +
                '\n' +
                '        for k, metric in output[\'metrics\'].items():\n' +
                '            metric = metric.mean(dim=0, keepdim=True)\n' +
                '            self.smoothed_metrics[k].update(metric.data[0])\n' +
                '```\n' +
                '\n' +
                '将损失值记录更新，输出数组的维度压缩后保存至`outputs[total_loss]`用于反传更新参数。\n'},
        {h3_title:'log_stats'},
        {text:'```Python\n' +
                '        def log_stats(self, cur_iter, lr, skip_metrics=False, skip_losses=False, suffix=None):\n' +
                '            """Log the tracked statistics."""\n' +
                '            if self.scheduler.iter_per_epoch == -1:\n' +
                '                log_flag = not cur_iter % self.log_period\n' +
                '            else:\n' +
                '                log_flag = not (cur_iter % self.scheduler.iter_per_epoch) % self.log_period\n' +
                '            if log_flag:\n' +
                '                stats = self.get_stats(cur_iter, lr)\n' +
                '                lines = \'[Training][{}]\'.format(stats[\'cfg_filename\'])\n' +
                '                if \'epoch\' in stats.keys():\n' +
                '                    lines += \'[epoch: {}/{}]\'.format(stats[\'epoch\'], stats[\'max_epoch\'])\n' +
                '                lines += \'[iter: {}/{}]\'.format(stats[\'iter\'], stats[\'max_iter\'])\n' +
                '                lines += \'[lr: {:.6f}][eta: {}]\\n\'.format(stats[\'lr\'], stats[\'eta\'])\n' +
                '\n' +
                '                lines += \'\\t  total_loss: {:.6f} ({:.6f}), \'.format(stats[\'total_loss_m\'], stats[\'total_loss\'])\n' +
                '                lines += \'iter_time: {:.4f} ({:.4f}), data_time: {:.4f} ({:.4f})\\n\'. \\\n' +
                '                    format(stats[\'iter_time_d\'], stats[\'iter_time\'], stats[\'data_time_d\'], stats[\'data_time\'])\n' +
                '\n' +
                '                if stats[\'metrics\'] and not skip_metrics:\n' +
                '                    lines += \'\\t  \' + \', \'.join(\'{}: {:.4f} ({:.4f})\'.format(k, float(v.split(\' \')[0]), \n' +
                '                                                                            float(v.split(\' \')[1])) \n' +
                '                                                for k, v in stats[\'metrics\'].items()) + \'\\n\'\n' +
                '                if stats[\'losses\'] and not skip_losses:\n' +
                '                    lines += \'\\t  \' + \', \'.join(\'{}: {:.6f} ({:.6f})\'.format(k, float(v.split(\' \')[0]), \n' +
                '                                                                            float(v.split(\' \')[1])) \n' +
                '                                                for k, v in stats[\'losses\'].items()) + \'\\n\'\n' +
                '                if suffix is not None:\n' +
                '                    lines += suffix + \'\\n\'\n' +
                '                print(lines[:-1])  # remove last new line\n' +
                '            return None\n' +
                '```\n' +
                '\n' +
                '该函数用于在控制台打印显示训练的状态信息，包括当前迭代数、当前全数据迭代数、损失值、时间、学习率等等，可以通过设置`skip_metrics`、`skip_losses`来选择是否显示评价指标和损失值，也可以通过设置`suffix`来进行信息的补充说明。\n'},
        {h3_title:'时间函数'},
        {text:'```Python\n' +
                '        def data_tic(self):\n' +
                '            self.data_timer.tic()\n' +
                '\n' +
                '        def data_toc(self):\n' +
                '            return self.data_timer.toc(average=False)\n' +
                '\n' +
                '        def iter_tic(self):\n' +
                '            self.iter_timer.tic()\n' +
                '\n' +
                '        def iter_toc(self):\n' +
                '            return self.iter_timer.toc(average=False)\n' +
                '\n' +
                '        def reset_timer(self):\n' +
                '            self.data_timer.reset()\n' +
                '            self.iter_timer.reset()\n' +
                '```\n' +
                '\n' +
                '用于记录开始、结束、迭代和数据处理时间。\n'},
        {part_title: 'TestingLogger类'},
        {text:'该类与TrainingLogger相差无几，区别在于使用的阶段不同，该类在测试阶段使用，主要成员函数包含时间函数和`log_stats`。用途与TrainingLogger相同。\n'},
        {part_title:'使用案例'},
        {text:'这里简单介绍一下在训练脚本中，日志模块的使用。\n' +
                '\n' +
                '首先，引入`TrainingLogger`类：\n' +
                '\n' +
                '```Python\n' +
                '    from pet.utils.logger import TrainingLogger\n' +
                '```\n' +
                '\n' +
                '在训练脚本中实例化：\n' +
                '\n' +
                '```Python\n' +
                '    logger = TrainingLogger(args.cfg_file.split(\'/\')[-1], scheduler=scheduler, log_period=cfg.DISPLAY_ITER)\n' +
                '```\n' +
                '\n' +
                '传入配置文件、学习率优化类和需要显示的迭代周期即可。\n' +
                '\n' +
                '在对应过程调用相应的成员函数：\n' +
                '\n' +
                '```Python\n' +
                '        for iteration, (images, targets, _) in enumerate(loader, cur_iter):\n' +
                '            logger.iter_tic()\n' +
                '            logger.data_tic()\n' +
                '\n' +
                '            scheduler.step()    # adjust learning rate\n' +
                '            optimizer.zero_grad()\n' +
                '\n' +
                '            images = images.to(device)\n' +
                '            targets = [target.to(device) for target in targets]\n' +
                '            logger.data_toc()\n' +
                '\n' +
                '            outputs = model(images, targets)\n' +
                '\n' +
                '            logger.update_stats(outputs, args.distributed, args.world_size)\n' +
                '            loss = outputs[\'total_loss\']\n' +
                '            loss.backward()\n' +
                '            optimizer.step()\n' +
                '\n' +
                '            if args.local_rank == 0:\n' +
                '                logger.log_stats(scheduler.iteration, scheduler.new_lr)\n' +
                '\n' +
                '                # Save model\n' +
                '                if cfg.SOLVER.SNAPSHOT_ITERS > 0 and (iteration + 1) % cfg.SOLVER.SNAPSHOT_ITERS == 0:\n' +
                '                    checkpointer.save(model, optimizer, scheduler, copy_latest=True, infix=\'iter\')\n' +
                '            logger.iter_toc()\n' +
                '```\n' +
                '\n' +
                '在迭代开始时，使用时间函数`logger.iter_tic()`，`logger.data_tic()`记录开始时间，分别在对应流程结束时调用`logger.data_toc()`，`logger.iter_toc()`记录结束时间；在得到模型输出outputs之后调用`logger.log_stats`将信息显示到控制台，进行训练状态的监控。\n'},
        {note:[{
                text:'Pet使用了PyTorch提供的分布式数据并行模式进行高效的深度学习计算，当使用多块GPU进行模型训练时，`TrainingLogger`会在打印每块GPU上模型的训练状态。为了使日志系统简洁地记录模型的训练状态，Pet通过设置`local_rank`这一参数来确保日志系统只记录第一块GPU上模型的状态，`local_rank`通常被设置为0。\n'
            }]},
        {part_title:'便利与规范'},
        {text:'基于Pet的代码实现标准，我们对日志系统进行了封装，您可以在训练和测试网络时获得以下便利：\n'},
        {
            ul:['您可以随时监督训练或测试的状态，及时发现出错的过程，以及时调整。','您可以很简单方便地使用日志系统，不需要过多的代码编写。']
        },
        {text:'如果您需要对我们的日志系统进行拓展，您需要完全遵循Pet的代码实现标准来进行修改和添加，欢迎您将有价值的代码和意见提交到我们的github，我们十分感谢任何对于发展Pet有益的贡献。\n'}
    ],
    dataNav:[
        {
            'TrainingLogger':['初始化','get_stats','update_stats','log_stats','时间函数']
        },
        'TestingLogger类','使用案例','便利与规范'
    ]
}