export let prepare_data = {
    key:'prepare_data',
    dataSource:[
        {title:'Data Preparation', className:'title_1'},
        {text:'The data preparation component includes two parts: ** dataset source file preparation** and **dataset registration**. The data set can be used in Pet to train and test the model according to the standards of data preparation.\n',},
        // {text: '[Dataset Format Standard](#Dataset_Format_Standard)\n' +
        //         '\n' +
        //         '>&#8195;&#8195;[classification standard](#classification_standard)\n' +
        //         '\n' +
        //         '>&#8195;&#8195;[example analysis standard](#example_analysis_standard)\n' +
        //         '\n' +
        //         '[Dataset registration](#Dataset_registration)\n' +
        //         '\n' +
        //         '[Models on different datasets](#Models_on_different_datasets)\n'},
        {part_title:'Dataset Format Standard'},
        {text:'Pet assigns a standard data format style to all visual tasks, working with a highly consistent data loading component to ensure that it can be efficiently combined and switched between datasets and models. Pet\'s requirements for dataset format mainly include two aspects: the file structure of the dataset source file and the storage form of the annotation information.\n' +
                '\n' +
                'According to the different tasks of computer vision, the size and labeling form of the data set are not the same. Following the principle of minimizing difference, Pet\'s standard for data set format is divided into two basic styles:\n'},
        {ul:'[ImageNet](http://www.image-net.org/challenges/LSVRC/)Dataset style-led classification task dataset format standard.'},
        {ul:'[MSCOCO](http://cocodataset.org/#home)Dataset format standard for other tasks dominated by dataset styles.'},

        {h3_title:'Classification Standard'},
        {text:'At present, the most popular classified dataset is the `ImageNet` dataset. A large number of visual tasks rely on the classification model trained on the `ImageNet` dataset for migration learning. Therefore, for the classification task, Pet currently mainly provides the "ImageNet" style classification dataset format standard. For the training and testing of the [Cifar](http://www.cs.toronto.edu/~kriz/cifar.html) dataset, Pet regards it as a simple quick start and does not give too much attention.\n',className:'segmentation'},
        {text:'Example of the structure of the training data folder:'},
        {text:'```\n' +
                '  └─(Dataset name defined by User)\n' +
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
                'The `n0001` and `n9999` in the file structure are folders for storing image data, which are used to indicate the category of data under its path (using the \'`label_mapping.txt` file to map to the corresponding category).In Pet, the dataset path is set to the `$Pet/data/` folder by default, and the user implements the soft connection. Here is an example of adding `ILSVRC2017` to `$Pet/data/`.'+
                '\n' +
                '```\n' +
                '  data\n' +
                '  └─(Dataset name defined by User)\n' +
                '    ├─train\n' +
                '    ├─val\n' +
                '    └─test\n' +
                '```'},
        {text:'If you need to use Pet to research on other classified datasets, please refer to ImageNet\'s data format to prepare the dataset, add your own dataset to `$Pet/data`, and refer to [dataset registration] (# Data set registration) completes the registration of the data.'},


        {h3_title: 'Example Analysis Standard'},
        {text:'The three visual modules of `rcnn`, `ssd`, `pose` include instance-based tasks such as object detection, instance segmentation, multi-person pose estimation, multi-person densepose estimation, and multi-person body part analysis. The `MSCOCO` dataset is currently the most widely used instance-level comprehensive dataset, so the data preparation components of the three visual modules in Pet are based on the file structure of the `MSCOCO2017` dataset and the labeling style as the main dataset format standard. The data set with highly uniform format can also take the great convenience of [cocoapi](https://github.com/cocodataset/cocoapi) and efficiently parse the dataset.\n'},
        {text:'Instance analyze is a computer vision task with the largest number of sub-tasks among all the visual tasks supported by Pet. The data preparation standard of all instance analyszing tasks under Pet is based on MSCOCOCO2017. MSCOCO dataset is the most popular target detection data set at present. Cocoapi also provides great convenience for dataset parsing, statistical analysis, visual analysis and algorithm evaluation.'},
        {text:'Object detection, instance segmentation, and pose estimation are several official annotations provided by the `MSCOCO` dataset. [Dense Human Pose Estimation](http://densepose.org/) (DensePose) is an extension subset of MSCOCO dataset provided by Facebook, Selected pictures in the MSCOCO dataset are annotated with densepose. They are important functions supported by Pet. Here are the unified data preparation standards of several instance analyzing task .'},
        {text:'The standard format of the MSCOCO dataset annotation file can be found in [COCO Official Document](http://cocodataset.org/#format-data).'},
        {text:'According to the Facebook\'s open source annotation file, the format of the annotation file `densepose_{dataset_name}_train/val/test.json` of the human body densepose estimation task is the same as the `MSCOCO` dataset, including `images`, `categories`, `annotations` three sections are used to store images, categories, and annotation information in instances. The content and format of the annotation information are as follows:'},
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
                '```'},
        {text:'In the DensePose-COCO dataset annotation content, all the annotation content of the instance analysis task is included, but for the target detection, instance segmentation and human pose estimation tasks, only the necessary external frame and the corresponding annotation content of the task are needed.'},
        {text:'Since Pet uses `cocoapi` for dataset parsing, if you need to perform model training and testing on a private dataset, you need to generate the corresponding `JSON` file, the `JSON` file contains the annotation information and the format `MSCOCO` dataset. The label file for the corresponding task is the same.'},
        {text:'Referring to the `MSCOCO2017` dataset, Pet specifies the following standard file structure for the data source files for instance analysis:'},
        {text:'```\n' +
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
                '```'},
        {text:'The above file structure includes data file structure including detection, instance segmentation, pose estimation, and human body intensive gesture tasks. If you need to train the instance analysis task model on the `MSCOCO2017` dataset, download the dataset source file and place it according to this file structure. If you want to train and test the model on other public datasets or your own private dataset, After generating the corresponding `JSON` annotation file, you also need to place the source file of the dataset according to the above file structure.'},

        {part_title:'Dataset Registration'},
        {text:'After the data set is completed according to the standard, the data set needs to be registered in the Pet, and the data set can be used in the Pet to train and test the model.'},
        {ul:'First, you need to softly connect the dataset source file to the `$Pet/data/` path. Take the MSCOCO data set as an example, and establish the data soft connection of the source file under Pet by the following command:'},
        {shell:'ln -s /home/dataset_dir/MSCOCO2017  $Pet/pet/data/coco\n'},
        {ul:'After completing the soft connection of the dataset source file, the data sets need to be further declared in Pet. In [$Pet/utils/data/catalog.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/pet/utils/data/dataset_catalog.py), specify the path of the image file and annotated file of your dataset, and set the keywords corresponding to the dataset as follows:'},
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
        {part_title:'Pet\'s Research on Different Datasets'},
        {table:{
                titles:['Visual Mission ','Dataset '],
                data:[["Image Classification","cifar、ImageNet"],["Semantic Segmentation","ADE2017"],["Target detection, case analysis","MSCOCO2017、VOC PASCAL、Densepose-COCO、MHP-v2、CIHP"],["Re-identification","DukeMTMC、Market1501、VehicleID"],["Attitude Estimation ","MSCOCO2017 keypoints"],]
            }
            , className:'table_pre'},
        {text:'Following Pet\'s standards for data preparation components, we have done a lot of experiments on many open source datasets. We have trained models with the stae-of-art accuracy on different tasks, and provided training configuration and download links for these models.\n'},
        {text:'The annotation format of some datasets under the tasks of semantics segmentation, target detection and Re-recognition may be very different from that of MSCOCOCOCO2017. It is impossible to read the annotation information directly using `cocoapi`. Therefore, it is necessary to convert the annotation format of these data sets into COCO style.'},
        {text:'The annotation format of some datasets under semantic segmentation, target detection, and re-identification tasks may be very different from MSCOCO2017. It is not possible to directly use the `cocoapi` to read the annotation information. Therefore, the annotation format of these datasets needs to be converted to COCO. style. Pet provides some conversion tools to convert [VOC PASCAL] (http://host.robots.ox.ac.uk/pascal/VOC/), [CityScapes] (https://www.cityscapes-dataset.com/) style annotation files into COCO-json annotations that can be read by cocoapi. For details, see [convert_xml_to_json](), [cconvert_cityscapes_to_coco.py](). If you need the annotations of datasets such as [ADE2017] (http://groups.csail.mit.edu/vision/datasets/ADE20K/), [Market1501]()after standardized format conversion, you can contact [us]().Pet provides some dataset conversion tools, which can be [VOC PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/), [CityScapes](https://www.cityscapes-dataset). The com/ style annotation file is converted to a COCO-json annotation that can be read by cocoapi. See [convert_xml_to_json.py](), [convert_cityscapes_to_coco.py]() for details. If you need [AD2017](http://groups.csail.mit.edu/vision/datasets/ADE20K/), [Market1501](http://www.liangzheng.com.cn/Datasets.html) and other datasets after the standardized format conversion, you can contact [us](http://www.petcv.net/Contact_us).\n'}

    ],
    dataNav:[
        {
            'Dataset Format Standard':[
                'Classification Standard',
                'Example Analysis Standard'
            ]
        },
        'Dataset Registration',
        'Pet\'s Research on Different Datasets',
    ]
};