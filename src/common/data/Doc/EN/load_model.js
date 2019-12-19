export let load_model_data = {
    key: 'load_model',
    dataSource: [
        {title:'Model Loading and Saving'},
        {text:'Loading and saving model is import for model training, Pet defines a class `CheckPointer` for encapsulating related functions. During the model training process, the loaded model is divided into the existing checkpoint and the finetuned model, also we can train from scratch. You can set the corresponding variables in `cfg.TRAIN.WEIGHTS`, and choose the initialization mode. The targets of model loading and saving includes the network parameters of the model, the optimizer and the learning rate scheduler needed \n',className:'segmentation'},
        {part_title:'CheckPointer'},
        {text:'When you want to load and save model parameters, `CheckPointer` can help you encapsulate this process. After the instantiation of the CheckPointer class is completed, the corresponding model loading or saving functions can be completed by calling the member functions. Refer to [$Pet/pet/utils/checkpointer.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/pet/utils/checkpointer.py) for the complete code.\n' +
                "\n" +
                'The implementation process of model loading and saving is as follows:\n'},
        {
            ul:[
                'Completing the instantiation of this class and confirms the loading mode of model parameters.',
                'The model parameters, optimizer and learning rate scheduler are updated by calling the member functions in the class.',
                'After the completion of each iteration, the trained model parameters, optimizer and learning rate scheduler were updated as the latest results.'
            ]
        },
        {h3_title:'Init'},
        {text:"The initialization of `CheckPointer` determines the loading mode of model parameters. It is implemented by setting `weights_path` and `resume`. Initialization requires `ckpt`, `weights_path`, `auto_resume`, `local_rank`, etc.\n"},
        {
            ul:[
                '`ckpt`: the path of the model for parameter loading.',
                '`weights_path`: the path of the model for finetune.',
                '`auto_resume`: as a flag for automatic loading the checkpoint, the weight parameters of the nearest checkpoint model are selected to initialize for True.',
                '`local_rank`: used in multi-GPU training to filter repeated information of models when using multiple GPUs, and only output relevant information of models on GPU specified by local_rank.'
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
                '`weights_path `and `resume` are used in the model loading function `load_model`, and their corresponding relations with different loading modes are introduced here.\n'},
        {
            ul:[
                '`resume=True`: load the model parameters from checkpoint;',
                '`resume=False` and `weights_path=True`: load model parameters from pre-trained model;',
                '`resume=False`and `weights_path=False`: load no parameters and train from scratch.'
            ]
        },
        {
            h4_block:[
                {h4_title:'get_model_latest'},
                {text:"Used to initialize `resume`, which is called during initialization, different values are assigned by confirming whether there is a target model naming `model_latest.pth` under the member variable of `ckpt`.\n"},
                {h4_title:'_load_file'},
                {text:'`checkpoint` is used to assign the parameters in the model file to `self.checkpoint`.\n'}
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
                'The three parameter loading modes of the model are implemented in the `load_model` function. The functions such as `weight_mapping`, `convert_conv1_rgb2bgr`, and other external functions are called here. The implementation process of the three loading modes in the function is described below.\n'},
        {text:'1. Steps of loading model parameters from the nearest checkpoint:\n'},
        {
            ul:[
                '1.The network parameters in `checkpoint` are assigned to `weights_dict` for storage;',
                '2.Use the `strip_prefix_if_present` function to remove the parameter prefix name of `weights_dict\';',
                '3.Return the model parameters(except parameters of optimizer) to the new parameter dictionary `model_state_dict` in the form of dictionary',
                '4.Using `align_and_update_state_dicts` function to map the parameters to the corresponding parameter dictionary `model_state_dict\'.',
                '5.Loaded parameters of `model_state_dict` to the model.',
                '6.`logging_rank` outputs the loading mode of model parameters.',
            ]
        },
        {text:'2. Steps of loading model parameters from the pre-trained model:\n'},
        {
            ul:[
                '1.`weights_dict` stores parameters of `checkpoint`;',
                '2.`strip_prefix_if_present` removes parameter prefix name `weights_dict`;',
                '3.`weight_mapping` replace the prefix of parameters in the pre-training model with the prefix of Pet defined model parameters;',
                '4.Convert parameter channels of the convolution layer by `convert_conv1_rgb2bgr` ;',
                '5.`align_and_update_state_dicts` maps parameters to the corresponding model\'s parameter dictionary;',
                '6.Load parameters in `model_state_dict` to the model;',
                '7.`logging_rank` outputs the loading mode of model parameters.'
            ]
        },
        {text:'3. Random initialize parameters\n'},
        {ul:'1.`logging_rank` outputs the loading mode of model parameters.'},
        {h3_title:'load_optimizer'},
        {text:"When the model parameters of the nearest checkpoint are loaded, the optimizer parameters are taken out to update the current optimizer.\n" +
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
        {text:"Similar to the feature of `load_optimizer`, the optimizer is used to update current learning rate scheduler.\n" +
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
        {text:'An intermediate variable `save_dict` is defined to save model parameters, optimizer and learning rate scheduler settings of this training process, and store them in the path specified by `ckpt` to print the saved state and file position of the model. Calling this function requires passing parameters such as `model`、`optimizer`、`scheduler`、`copy_latest` and `infix`.\n' +
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
                titles:['Member variable','Meaning'],
                data:[["model","model in the current iteration"],
                    ["optimizer","optimizer states in the current iteration"],
                    ["scheduler","learning rate scheduler in the current iteration"],
                    ["copy_latest","flag variable for whether to copy the model parameters file"],
                    ["infix","current iterations"],]
            }
            , className:'table_2'},

        {part_title: 'Convenience and Specification'},
        {text:"Based on the above standards, Pet encapsulats the model parameter loading and saving system, and obtains the following advantages in training and testing the network:\n"},
        {
            ul:[
                'Convenient and efficient transformation of various model parameter loading modes and storage of model parameters can accelerate the engineering development period.',
                'Easily use the model parameter loading and saving system to get the configurations of learning rate schduler and optimizer。'
            ]
        },
        {text:"If you are ready to expand our model parameter loading and saving system, you need to completely follow the code implementation standards of Pet to modify and extend. You are welcome to submit valuable code and comments to our github. We appreciate any useful contributions to the development of Pet.\n"}
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
        'Convenience and Specification'
    ]
}