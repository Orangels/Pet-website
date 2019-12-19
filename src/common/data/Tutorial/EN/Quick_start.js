export let Quick_start = {
    key: 'Quick_start',
    dataSource: [
        {title:'Getting Started: Train a classification model on CIFAR10 dataset\n', className:'title_1'},
        {text:'According to quick start, you can learn the brief steps of using Pet to train and test a classifier on the CIFAR10 data set. For more information, click the following links:\n'},
        {
            ul:[
                'Details of building a classifier on ImageNet dataset in [ImageNet Classification Tutorials](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E5%9C%A8ImageNet%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E8%AE%AD%E7%BB%83%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B.md).',
                'Introduction of system components of Pet in [Documents](https://github.com/BUPT-PRIV/Pet-DOC/tree/master/%E6%9E%B6%E6%9E%84/component-collects).',
                'See the detailed training and testing code in [$Pet/tools/cls/train_net.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/tools/cls/train_net.py) and [$Pet/tools/cls/test_net.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/tools/cls/test_net.py).',
            ]
        },
        {text:'Now, let\'s get start to build the classifier：\n'},
        {part_title:'Data Preparation'},
        {ul:'Download：Before training the model，you first download the [python version of CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) locally and decompression. The contents and categories of CIFAR10 dataset are as follows:'},
        {img:'cifar10_pic'},
        {ul:'Data root：Store the data directly or in the form of a soft links under the *$Pet/data* file path, The file structure is as follows:'},
        {yaml:'```\n' +
                '    cifar\n' +
                '      |--cifar-10-batches-py\n' +
                '        |--data_batch_1\n' +
                '        |--data_batch_2\n' +
                '        |--data_batch_3\n' +
                '        |--data_batch_4\n' +
                '        |--data_batch_5\n' +
                '        |--test_batch\n' +
                '        ...\n' +
                '```\n'},
        {text:'Establish the current data path `CIFAR` to `$Pet/data` soft link:\n'},
        {shell:'```\n' +
                '    ln -s $CIFAR $Pet/data\n' +
                '```\n'},
        {ul:'Register dataset：Pet has already difined the root path of CIFAR10 dataset in [$Pet/pet/utils/data/catalog.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/pet/utils/data/dataset_catalog.py)，please turn to [Data Preparation](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%9E%B6%E6%9E%84/component-collects/%E6%95%B0%E6%8D%AE%E5%88%B6%E5%A4%87.md) for more information of dataset registration。'},
        {part_title: 'Pre-train model'},
        {text:'\'If you want to fine-tuning the model, please the pre-train model under `$Pet/weights`. As CIFAR10 classification is the simple and basic classfication task, there is no need to use pre-train model, so Pet just train the model from scratch. \n'},
        {part_title:'Training and testing a classifier'},
        {text:'When using Pet to train and test a classifier on CIFAR10, it is necessary to specify a `yaml` file, which contains all the hyper parameters used in training and testing. Here we take [$Pet/cfgs/cls/cifar/resnext29-8x64d_cifar10.yaml](https://github.com/BUPT-PRIV/Pet-dev/blob/master/cfgs/cls/cifar/resnext29-8x64d_cifar10.yaml) as an example.\n'},
        {text:'Training commands：\n'},
        {shell:'```\n' +
                '     CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/cls/train_net.py \n' +
                '     --cfg cfgs/cls/cifar/resnext29-8x64d_cifar10.yaml\n' +
                '```\n'},

        {text:'During training progress，following information will be logged in the console 。\n'},
        {text:'```\n' +
                '    [Training][resnext29-8x64d_cifar10.yaml][epoch: 1/300][iter: 20/196][lr: 0.100000][eta: 1 day, 1:28:29]\n' +
                '          total_loss: 2.264606 (2.198514), iter_time: 0.4097 (1.5602), data_time: 0.0019 (0.0021)\n' +
                '          acc1: 22.2656 (19.7266), acc5: 71.4844 (70.0586)\n' +
                '    [Training][resnext29-8x64d_cifar10.yaml][epoch: 1/300][iter: 40/196][lr: 0.100000][eta: 15:49:45]\n' +
                '          total_loss: 1.991784 (2.129857), iter_time: 0.4080 (0.9698), data_time: 0.0022 (0.0021)\n' +
                '          acc1: 23.4375 (22.0312), acc5: 81.2500 (74.6777)\n' +
                '    [Training][resnext29-8x64d_cifar10.yaml][epoch: 1/300][iter: 60/196][lr: 0.100000][eta: 12:43:38]\n' +
                '          total_loss: 1.797434 (2.069275), iter_time: 0.4100 (0.7800), data_time: 0.0020 (0.0021)\n' +
                '          acc1: 30.8594 (24.0299), acc5: 84.7656 (77.0508)\n' +
                '    [Training][resnext29-8x64d_cifar10.yaml][epoch: 1/300][iter: 80/196][lr: 0.100000][eta: 11:11:46]\n' +
                '          total_loss: 1.910455 (2.017684), iter_time: 0.4116 (0.6864), data_time: 0.0020 (0.0021)\n' +
                '          acc1: 27.7344 (25.7080), acc5: 82.0312 (78.8721)\n' +
                '          ......\n' +
                '```\n'},
        {text:'After the training, the lastest checkpoint and the best model will be saved to the `$Pet/ckpts/cls/cifar` path. Next we can test the classification model we have trained. \n' +
                '\n' +
                'Testing command：\n'},
        {shell:'```\n' +
                '    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/cls/test_net.py \n' +
                '    --cfg cfgs/cls/cifar/resnext29-8x64d_cifar10.yaml\n' +
                '```\n'},
        {text:'Testing result and log information：\n'},
        {shell:'```\n' +
                '    INFO:pet.utils.misc:[Testing][range:1-250 of 250][249/250][120.903s = 120.747s + 0.154s + 0.002s][eta: 0:02:00][acc1:\n' +
                '    77.48% | acc5: 93.75%]\n' +
                '    INFO:pet.utils.misc:[Testing][range:1-250 of 250][250/250][121.293s = 121.138s + 0.154s + 0.002s][eta: 0:00:00][acc1:\n' +
                '    77.44% | acc5: 93.75%]\n' +
                '    INFO:pet.utils.misc:val_top1: 77.4360% | val_top5: 93.7520%\n' +
                '```\n'}
    ]
}


