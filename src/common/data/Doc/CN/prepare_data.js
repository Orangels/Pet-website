export let prepare_data = {
    key:'prepare_data',
    dataSource:[
        {title:'数据制备', className:'title_1'},
        {text:'数据制备组件包括了**数据集源文件的准备**以及**数据集注册**两个部分。遵循数据制备的标准即可在Pet中使用数据集进行模型的训练和测试。\n',},
        {part_title:'数据集格式标准'},
        {text:'Pet对所有视觉任务指定了一套标准的数据格式风格，配合高度一致的数据载入组件的工作方式与风格，确保能够高效的在数据集、模型上进行组合与切换。Pet对数据集格式的要求主要包括数据集源文件的文件结构和标注信息的存储形式两个方面。\n' +
                '\n' +
                '根据计算机视觉任务的不同，数据集的规模、标注形式也不尽相同，遵循最小化差异原则，Pet对于数据集格式的标准被划分为两种基本风格：\n'},
        {
            ul:['[ImageNet](http://www.image-net.org/challenges/LSVRC/)数据集风格主导的分类任务数据集格式标准。',
                '[MSCOCO](http://cocodataset.org/#home)数据集风格主导的实例分析任务的数据集格式标准。'
            ]
        },
        {h3_title:'分类标准'},
        {text:'目前最流行的分类数据集当属`ImageNet`数据集，大量的视觉任务都依赖在`ImageNet`数据集上训练得到的分类模型进行迁移学习。因此对于分类任务，Pet目前主要提供`ImageNet`风格的分类数据集格式标准，对于[CIFAR](http://www.cs.toronto.edu/~kriz/cifar.html)数据集的训练与测试，Pet将之视为简单的快速开始，并未给予过多的关注。'},
        {text:'训练数据文件夹的结构示例:'},
        {text:'```\n' +
                '  └─(用户定义的数据集名称)\n' +
                '    └─train\n' +
                '      ├─n0001\n' +
                '      | ├─Image_00001.JPEG\n' +
                '      | ...\n' +
                '      | └─Image_99999.JPEG\n' +
                '      ...\n' +
                '      └─n9999\n' +
                '        ├─Image_00001.JPEG\n' +
                '        ...\n' +
                '        └─Image_99999.JPEG\n' +
                '```\n' +
                '\n' +
                '文件结构中的`n0001`、`n9999`是存放图片数据的文件夹，用来表示其路径下数据的类别(使用`label_mapping.txt`文件映射到对应类别)。\n' +
                '在Pet中，默认将数据集路径设在`$Pet/data/`文件夹下，通过建立软连接的方式实现，这里给出`ILSVRC2017`加到`$Pet/data/`中的示例。\n' +
                '\n' +
                '```\n' +
                '  data\n' +
                '  └─(用户定义的数据集名称)\n' +
                '    ├─train\n' +
                '    ├─val\n' +
                '    └─test\n' +
                '```'},
        {text:'如果需要使用Pet在其他的分类数据集上进行研究，请先参考ImageNet的数据格式制备数据集，将自己的数据集通过加入到`$Pet/data`下，并参考[数据集注册](#数据集注册)完成数据的注册。'},
        {h3_title:'实例分析标准'},
        {text:'`rcnn`，`ssd`，`pose`三个视觉模块包含了目标检测、实例分割、多人姿态估计、多人密集姿态估计、多人人体部位分析等基于实例的视觉任务的实现，而`MSCOCO`数据集是目前应用最广泛的实例级综合数据集，因此Pet中这三个视觉模块的数据制备组件以`MSCOCO2017`数据集的文件结构以及标注风格为主体数据集格式标准，并且使用高度统一格式的数据集能够充分发挥[cocoapi](https://github.com/cocodataset/cocoapi)的巨大便利性，对数据集进行高效的解析。\n'},
        {text:'实例分析是Pet所有支持的视觉任务中，包含子任务最多的计算机视觉任务，所有实例分析任务在Pet下的数据制备标准均以MSCOCO2017为参考。MSCOCO数据集是目前最为流行的目标检测数据集，cocoapi也为数据集的解析、统计分析、可视化分析与算法评估提供了极大便利。\n'},
        {text:'目标检测、实例分割、姿态估计是`MSCOCO`数据集所提供的几种官方标注，[人体密集姿态估计](http://densepose.org/)（DensePose）标注是由Facebook提供的MSCOCO数据集的扩充标注，选择了MSCOCO数据集中部分图片进行了人体密集姿态标注。它们是Pet所支持的重要功能，在此将几种实例分析任务的数据制备标准统一说明。\n'},
        {text:'MSCOCO数据集的标注文件的标准格式请见[COCO官方文档](http://cocodataset.org/#format-data)。\n'},
        {text:'根据Facebook开源的标注文件，人体密集姿态估计任务的标注文件`densepose_{dataset_name}_train/val/test.json`的格式与`MSCOCO`数据集相同，都包含有`images`、`categories`、`annotations`三个部分来分别存储图片、类别以及以实例为单位的标注信息，标注信息所包含的内容以及格式如下所示：\n'},
        {text:'```\n' +
                '    {\n' +
                '      "id": int,\n' +
                '      "iscrowd": 0 or 1,\n' +
                '      "category_id": int,\n' +
                '      "area": float,\n' +
                '      "num_keypoints": int,\n' +
                '      "bbox": [x, y, w, h],\n' +
                '      "keypoints": [x1, y1, v1, x2, y2, v2, ...],\n' +
                '      "segmentation": RLE or [polygon],\n' +
                '      "dp_x": [x1, x2, ...],\n' +
                '      "dp_y": [y1, y2, ...],\n' +
                '      "dp_U": [U1, U2, ...],\n' +
                '      "dp_V": [V1, V2, ...],\n' +
                '      "dp_I": [I1, I2, ...],\n' +
                '      "dp_masks":  RLE or [polygon],\n' +
                '      "image_id": int\n' +
                '    }\n' +
                '```\n' +
                '\n' +
                '在DensePose-COCO数据集标注内容中，包含了实例分析任务所有的标注内容，但对于目标检测、实例分割和人体姿态估计任务来说，只需要必要的外接框和任务对应的标注内容。\n'},
        {text:'由于Pet采用`cocoapi`进行数据集解析，因此如果您需要在私人数据集上进行模型训练以及测试，则需要生成相应的`JSON`文件，`JSON`文件包含标注信息且格式与`MSCOCO`数据集中相应任务的标注文件相同。\n'},
        {text:'参考`MSCOCO2017`数据集，Pet为实例分析的数据源文件规定了如下标准的文件结构：\n' +
                '\n' +
                '```\n' +
                '  └─MSCOCO2017(dataset name)\n' +
                '    ├─annotations(from annotations_trainval2017.zip)\n' +
                '    | ├─instances_train2017.json\n' +
                '    | ├─instances_val2017.json\n' +
                '    | ├─person_keypoints_train2017.json\n' +
                '    | ├─person_keypoints_val2017.json\n' +
                '    | ├─densepose_{dataset_name}_train.json\n' +
                '    | ├─densepose_{dataset_name}_val.json\n' +
                '    | ├─densepose_{dataset_name}_test.json\n' +
                '    | ├─image_info_test2017.json\n' +
                '    | └─image_info_test-dev2017.json\n' +
                '    ├─train2017\n' +
                '    | ├─000000000009.JPEG\n' +
                '    | ...\n' +
                '    | └─000000581929.JPEG\n' +
                '    ├─val2017\n' +
                '    | ├─000000000139.jpg\n' +
                '    | ...\n' +
                '    | └─000000581781.jpg\n' +
                '    ├─test2017\n' +
                '    | ├─000000000001.jpg\n' +
                '    | ...\n' +
                '    | └─000000581918.jpg\n' +
                '    └─annotations(from image_info_test2017.zip)\n' +
                '```\n'},
        {text:'上面的文件结构包括了检测、实例分割、姿态估计和人体密集姿态任务在内的数据文件结构。如果需要在`MSCOCO2017`数据集上训练实例分析任务模型，请下载数据集源文件并按此文件结构放置，如果您想在其他公开数据集或是自己的私人数据集上进行模型训练和测试，在生成相应的`JSON`标注文件后，也需要按上面的文件结构来放置数据集的源文件。\n'},
        {part_title:'数据集注册'},
        {text:'按照标准完成了的数据集制备之后，还需要在Pet中对数据集进行注册，才可以Pet中使用数据集进行模型的训练以及测试。\n'},
        {ul:'首先需要将数据集源文件软连接到`$Pet/data/`路径中，以MSCOCO数据集为例，通过如下指令建立源文件在Pet下的数据软连接：'},
        {shell:'ln -s /home/dataset_dir/MSCOCO2017  $Pet/data/coco\n'},
        {ul:'完成数据集源文件的软连接后，需要进一步在Pet中对数据集进行声明。在[$Pet/utils/data/catalog.py](https://github.com/BUPT-PRIV/Pet/blob/master/pet/utils/data/dataset_catalog.py)中指定您数据集的图片文件、标注文件的路径，并设置数据集对应的关键词，如下所示：\n'},
        {text:'```\n' +
                '    \'coco_2017_train\': {\n' +
                '        _IM_DIR:\n' +
                '            _DATA_DIR + \'/coco/images/train2017\',\n' +
                '        _ANN_FN:\n' +
                '            _DATA_DIR + \'/coco/annotations/instances_train2017.json\',\n' +
                '    },\n' +
                '    \'coco_2017_val\': {\n' +
                '        _IM_DIR:\n' +
                '            _DATA_DIR + \'/coco/images/val2017\',\n' +
                '        _ANN_FN:\n' +
                '            _DATA_DIR + \'/coco/annotations/instances_val2017.json\',\n' +
                '    },\n' +
                '    \'coco_2017_test\': {\n' +
                '        _IM_DIR:\n' +
                '            _DATA_DIR + \'/coco/images/test2017\',\n' +
                '        _ANN_FN:\n' +
                '            _DATA_DIR + \'/coco/annotations/image_info_test2017.json\',\n' +
                '    },\n' +
                '    \'coco_2017_test-dev\': {\n' +
                '        _IM_DIR:\n' +
                '            _DATA_DIR + \'/coco/images/test2017\',\n' +
                '        _ANN_FN:\n' +
                '            _DATA_DIR + \'/coco/annotations/image_info_test-dev2017.json\',\n' +
                '    }\n' +
                '```\n'},
        {part_title:'Pet在不同数据集上进行的研究'},
        {table:{
                titles:['视觉任务','数据集'],
                data:[["图像分类","cifar、ImageNet"],["语义分割","ADE2017"],["目标检测、实例分析","MSCOCO2017、VOC PASCAL、Densepose-COCO、MHP-v2、CIHP"],["重识别","DukeMTMC、Market1501、VehicleID"],["姿态估计","MSCOCO2017 keypoints"],]
            }
            , className:'table_pre'},
        {text:'遵循Pet所制定的数据制备组件的标准，我们在许多开源数据集上进行了大量的实验，已经在不同的任务上都训练出了最高水平精度的模型，同时提供这些模型的训练参数配置以及模型下载。\n'},
        {text:'语义分割、目标检测、重识别任务下的一些数据集的标注格式可能与MSCOCO2017有很大不同，无法直接使用`cocoapi`进行标注信息的读取，因此需要将这些数据集的标注格式转换为COCO风格。Pet提供了一些数据集转换工具，可以将[VOC PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/)、[CityScapes](https://www.cityscapes-dataset.com/)风格的标注文件转换成cocoapi可以读取的COCO-json标注，详情请见[convert_xml_to_json.py](https://github.com/BUPT-PRIV/Pet/blob/master/tools/rcnn/scripts/convert_xml_to_json.py)、[convert_cityscapes_to_coco.py](https://github.com/BUPT-PRIV/Pet/blob/master/tools/rcnn/scripts/convert_cityscapes_to_coco.py)。如果您需要[ADE2017](http://groups.csail.mit.edu/vision/datasets/ADE20K/)、[Market1501](http://www.liangzheng.com.cn/Datasets.html)等数据集经过标准化格式转换之后的标注，可以联系[我们](http://www.petcv.net/Contact_us)。\n'}

    ],
    dataNav:[
        {
            '数据集格式标准':['分类标准','实例分析标准']
        },
        '数据集注册',
        'Pet在不同数据集上进行的研究'

    ]
};