export let log_system_data = {
    key:'Log System',
    dataSource: [
        {title:'Log System'},
        {text:'The log module is an auxiliary module for the model training and testing phase. It can output information in the console to help you monitor the state of the model. Pet encapsulates these functions into `TrianingLogger` and `TestingLogger` Python classes, as well as two auxiliary classes `SmoothedValue` and `AverageMeter`. You can instantiate the class to output the status of model while training or testing the networks.\n'},
        {part_title:'TrainingLogger'},
        {text:'When constructing the training process, you can instantiate this class, call the methods and implement the output of the console information to monitor the training process.\n'},
        {h3_title:'Init'},
        {text:'The training script uses the `TrainingLogger` class, which receives `cfg_filename`, `scheduler`, `log_period` as arguments. The main member functions include `get_stats`, `update_stats`, `log_stats`, and multiple time counting functions, when instantiating the class, first do the initialization:\n' +
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
            ul:['`scheduler` is a learning rate optimization class, including the number of iterations, the number of training epochs, etc.',
                '* `log_period` is the iteration interval of the printed information, which means that the training state is printed after every `log_period` iteration.',
                '`SmoothedValue` is an auxiliary class that smoothes loss, which returns the median, average, etc.'
            ]
        },
        {text:'The instantiated `TrainingLogger` class uses `update_stats`, `log_stats` as main fhunctions, and `get_stats` is used in the two former methods.\n'},
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
                'In the training phase, the output is presented in a dictionary structure, containing two sub-dictionaries of `metrics` and `losses`, which are added to the `stats` list in this function.\n'},
        {ul:'The keywords in `stats` are shown in the following table: '},
        {table:{
                titles:['Keyword','Value'],
                data:[["epoch","Current data iteration number"],
                    ["max_epoch","Maximum Full Data Iteration Number"],
                    ["iter","The number of iterations currently"],
                    ["max_iter","Maximum Iterations"],
                    ["lr","learning rate"],
                    ["eta","Estimated time required for remaining iterations"],
                    ["data_time","Average time spent processing data"],
                    ["data_time_d","Total time spent processing data"],
                    ["iter_time","Average time per iteration in the iteration cycle"],
                    ["iter_time_d"," Total time for the iteration period"],
                    ["total_loss","Average of loss values"],
                    ["total_loss_m","Median loss value"],
                    ["metrics","Output task evaluation metrics, such as `\"classes\":[top1, top5]`"],
                    ["losses","The loss values in individual tasks are also stored in the form of a dictionary\n"]]
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
                'Update the loss values, the output array are saved to `outputs[\'total_loss]` with compressed dimensions for parameters updating.\n'},
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
                'This function is used to print the status information of the training process in the console, including current iteration number, current epoch, loss value, time cost, learning rate, etc. `Skp_metrics` and `skip_losses` can be set to choose whether to display the evaluation index and loss value, or `suffix` can be set to supplement the information.\n'},
        {h3_title:'Time Function'},
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
                'It is used to record start, end, iteration, and data processing time.\n'},
        {part_title: 'TestingLogger'},
        {text:'`TestingLogger` is similar to TrainingLogger in that it is used in different phases. The main member functions include time function and `log_stats`. The purpose is the same as TrainingLogger.\n'},
        {part_title:'Example'},
        {text:'Here is a brief introduction on how to use `TrainingLogger` in the training script.\n' +
                '\n' +
                'First, introduce the `TrainingLogger` class:\n' +
                '\n' +
                '```Python\n' +
                '    from pet.utils.logger import TrainingLogger\n' +
                '```\n' +
                '\n' +
                'Instantiat in the training script:\n' +
                '\n' +
                '```Python\n' +
                '    logger = TrainingLogger(args.cfg_file.split(\'/\')[-1], scheduler=scheduler, log_period=cfg.DISPLAY_ITER)\n' +
                '```\n' +
                '\n' +
                'Provide the configuration file, the learning rate optimization class, and the iteration cycle that needs to be displayed.\n' +
                '\n' +
                'Call the member functions in the corresponding procedure:：\n' +
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
                'At the beginning of the iteration, use the time function `logger.iter_tic()`, `logger.data_tic()` to record the start time,  and call `logger.data_toc()`, `logger.iter_toc()` to record the end time; after getting the model outputs, call `logger.log_stats` to display the information in the console for monitoring the training status.\n'},
        {note:[{
                text:'Pet uses the distributed data parallel mode provided by PyTorch for efficient computation. When using multiple GPUs to  train, `TrainingLogger` will prints the training status of the model on each GPU. To make recording concisely, Pet ensures that the log system records only the state of the model on the first GPU by setting the `local_rank` parameter, which is usually set to 0.\n'
            }]},
        {part_title:'Convenience and Specification'},
        {text:'Based on the Pet\'s code implementation standard, we encapsulate the logging system so you can get the following conveniences when training and testing your network:\n'},
        {
            ul:['You can monitor the status of training or testing process, find out the errors adjust it in time.',
                'You can use the logging system very easily and without too much code.']
        },
        {text:'If you want to extend our log system, you need to follow Pet\'s code implementation standards to modify. You are welcome to submit valuable code and comments to our github, we are very grateful for any benefit to the development of Pet\'s contribution.\n'}
    ],
    dataNav:[
        {
            'TrainingLogger':['Init','get_stats','update_stats','log_stats','Time Function']
        },
        'TestingLogger','Example','Convenience and Specification'
    ]
}