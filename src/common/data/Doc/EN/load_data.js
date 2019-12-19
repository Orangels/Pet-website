export let load_data_data = {
    key: 'Data Loading',
    dataSource: [
        {title:'Data Loading'},
        {text:'Data loading is a key part of the deep learning model training process. Pet provides a complete and efficient loading and preprocessing loading method for the data loading process. In use, Pet completes the data loading process by calling the two functions `build_dataset` and `make_train_data_loader`.\n'},
        {part_title:'build_dataset'},
        {text:'Take the dataloader in ssd as an example. First, Pet calls the function `build_dataset` during training to get the data set and data and useful information. At the same time, according to the requirements in the configuration file, call `build_transforms` to define the preprocessing method of the data.\n'},
        {text:'```Python\n' +
                '    def build_dataset(dataset_list, is_train=True, local_rank=0):\n' +
                '        if not isinstance(dataset_list, (list, tuple)):\n' +
                '            raise RuntimeError(\n' +
                '                "dataset_list should be a list of strings, got {}".format(dataset_list)\n' +
                '            )\n' +
                '        for dataset_name in dataset_list:\n' +
                '            assert contains(dataset_name), \'Unknown dataset name: {}\'.format(dataset_name)\n' +
                '            assert os.path.exists(get_im_dir(dataset_name)), \'Im dir \\\'{}\\\' not found\'.format(get_im_dir(dataset_name))\n' +
                '            logging_rank(\'Creating: {}\'.format(dataset_name), local_rank=local_rank)\n' +
                '\n' +
                '        transforms = build_transforms(cfg, is_train)\n' +
                '        datasets = []\n' +
                '        for dataset_name in dataset_list:\n' +
                '            args = {}\n' +
                '            args[\'root\'] = get_im_dir(dataset_name)\n' +
                '            args[\'ann_file\'] = get_ann_fn(dataset_name)\n' +
                '            args[\'remove_images_without_annotations\'] = is_train\n' +
                '            ann_types = (\'bbox\',)\n' +
                '            args[\'ann_types\'] = ann_types\n' +
                '            args[\'transforms\'] = transforms\n' +
                '            # make dataset from factory\n' +
                '            dataset = D.COCODataset(**args)\n' +
                '            datasets.append(dataset)\n' +
                '\n' +
                '        # for training, concatenate all datasets into a single one\n' +
                '        dataset = datasets[0]\n' +
                '        if len(datasets) > 1:\n' +
                '            dataset = D.ConcatDataset(datasets)\n' +
                '\n' +
                '        return dataset\n' +
                '```\n'},
        {h3_title:'make_train_data_loader'},
        {text:'After the data set is built, the data load of the `tatch.utils.data.DataLoader` is loaded by the data provided by Torch to complete the data loading of the Pet, and the data is preprocessed. Pet provides users with a rich image preprocessing method. For details, refer to [transforms](#transforms):\n'},
        {text:'```Python\n' +
                '    def make_train_data_loader(datasets, is_distributed=False, start_iter=0):\n' +
                '        images_per_gpu = cfg.TRAIN.IMS_PER_GPU\n' +
                '        shuffle = True\n' +
                '        num_iters = cfg.SOLVER.MAX_ITER\n' +
                '\n' +
                '        # group images which have similar aspect ratio. In this case, we only\n' +
                '        # group in two cases: those with width / height > 1, and the other way around,\n' +
                '        # but the code supports more general grouping strategy\n' +
                '        aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []\n' +
                '\n' +
                '        sampler = make_data_sampler(datasets, shuffle, is_distributed)\n' +
                '        batch_sampler = make_batch_data_sampler(\n' +
                '            datasets, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter\n' +
                '        )\n' +
                '        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)\n' +
                '        num_workers = cfg.TRAIN.LOADER_THREADS\n' +
                '        data_loader = torch.utils.data.DataLoader(\n' +
                '            datasets,\n' +
                '            num_workers=num_workers,\n' +
                '            batch_sampler=batch_sampler,\n' +
                '            collate_fn=collator,\n' +
                '        )\n' +
                '\n' +
                '        return data_loader\n' +
                ' ```\n'},
        {text:'It should be noted that the ssd, pose and other tasks are determined and consistent when the data is loaded. However, due to the particularity of the two-stage target detection task rcnn, Pet will separately propose the data sampler under the rcnn task to function. The form of `make_batch_data_sampler` is placed under [$Pet/pet/rcnn/datasets/dataset.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/pet/rcnn/datasets/dataset.py). The reason is that in the implementation of the two-stage detector, we need to scale the largest graph in each batch and scale it to determine the required tensor dimension, which is the basis for scaling the rest of the images in the same batch. On the right side and the lower side, the edge-filling operation is performed. In order to reduce the increase in the amount of calculation caused by the trimming, we divide the picture into two cases: width greater than height and height greater than width. The same type of picture is collected into the same batch for data loading.\n' },
        {text:'```Python\n' +
                '    def make_batch_data_sampler(\n' +
                '        dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0\n' +
                '    ):\n' +
                '        if aspect_grouping:\n' +
                '            if not isinstance(aspect_grouping, (list, tuple)):\n' +
                '                aspect_grouping = [aspect_grouping]\n' +
                '            aspect_ratios = _compute_aspect_ratios(dataset)\n' +
                '            group_ids = _quantize(aspect_ratios, aspect_grouping)\n' +
                '            batch_sampler = samplers.GroupedBatchSampler(\n' +
                '                sampler, group_ids, images_per_batch, drop_uneven=False\n' +
                '            )\n' +
                '        else:\n' +
                '            batch_sampler = torch.utils.data.sampler.BatchSampler(\n' +
                '                sampler, images_per_batch, drop_last=False\n' +
                '            )\n' +
                '        if num_iters is not None:\n' +
                '            batch_sampler = samplers.IterationBasedBatchSampler(\n' +
                '                batch_sampler, num_iters, start_iter\n' +
                '            )\n' +
                '        return batch_sampler\n' +
                '```\n' +
                '\n' +
                'Pet puts the data load component shared by all tasks under `$Pet/pet/utils/data` and contains the following:\n' +
                '\n' +
                '```\n' +
                '  ├─utils\n' +
                '    ├─data\n' +
                '      ├─datasets\n' +
                '      ├─samplers\n' +
                '      ├─structures\n' +
                '      ├─transforms\n' +
                '      ├─__init__.py\n' +
                '      ├─collate_batch.py\n' +
                '      ├─dataset_catalog.py\n' +
                '```\n'},
        {part_title: 'Datasets'},
        {text:'`datasets` provides three kinds of data reading methods for classification, target detection and instance analysis for different tasks. It is presented in three separate classes: `ImageFolderList`, `COCODataset` and `COCOInstanceDataset`. In the implementation process of subsequent different tasks, the image and annotation data are read and loaded by inheriting these two classes.\n'},
        {h3_title:'ImageFolderList'},
        {text:'Pet provides the image `ImageFolderList` for obtaining image and lable for the data loading of the image classification task. After the user stores the data according to the format required by the classification task in the data preparation, the data loading process of the classification task is called.\n'},
        {h3_title:'Initialization'},
        {text:'```Python\n' +
                '    class ImageFolderList(DatasetFolder):\n' +
                '        def __init__(self, root_list, transform=None, target_transform=None,\n' +
                '                     loader=default_loader):\n' +
                '            if not isinstance(root_list, (list, tuple)):\n' +
                '                raise RuntimeError(\n' +
                '                    "dataset_list should be a list of strings, got {}".format(dataset_list)\n' +
                '                )\n' +
                '\n' +
                '            super(ImageFolderList, self).__init__(root_list[0], loader, IMG_EXTENSIONS,\n' +
                '                                                  transform=transform,\n' +
                '                                                  target_transform=target_transform)\n' +
                '            if len(root_list) > 1:\n' +
                '                for root in root_list[1:]:\n' +
                '                    classes, class_to_idx = self._find_classes(root)\n' +
                '                    for k in class_to_idx.keys():\n' +
                '                        class_to_idx[k] += len(self.classes)\n' +
                '                    samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)\n' +
                '                    self.classes += classes\n' +
                '                    self.class_to_idx.update(class_to_idx)\n' +
                '                    self.samples += samples\n' +
                '            self.targets = [s[1] for s in self.samples]\n' +
                '            self.imgs = self.samples\n' +
                '```\n'},
        {h3_title:'COCODataset'},
        {text:'For target detection tasks such as ssd and rcnn, Pet provides image-level annotation file parsing and reading class `COCODataset` to complete the loading of image-level annotations during data loading.\n'},
        {
            h4_block:[
                {h4_title:'Init'},
                {text:'First we initialize `COCODataset`: `COCODataset` inherits the class `torchvision.datasets.coco.CocoDetection`, and the initialization parameters include:\n'},
                {
                    ul:['`ann_file`: label file path',
                        '`root`: image data path',
                        '`remove_images_without_annotations`: whether to remove files without annotation information',
                        '`ann_types`: annotation file type',
                        '`transforms`: preprocessing method'
                    ]
                },
                {text:'```Python\n' +
                        '  class COCODataset(torchvision.datasets.coco.CocoDetection):\n' +
                        '      def __init__(\n' +
                        '          self, ann_file, root, remove_images_without_annotations, ann_types,   transforms=None\n' +
                        '      ):\n' +
                        '          super(COCODataset, self).__init__(root, ann_file)\n' +
                        '          # sort indices for reproducible results\n' +
                        '          self.ids = sorted(self.ids)\n' +
                        '\n' +
                        '          # filter images without detection annotations\n' +
                        '          if remove_images_without_annotations:\n' +
                        '              ids = []\n' +
                        '              for img_id in self.ids:\n' +
                        '                  ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)\n' +
                        '                  anno = self.coco.loadAnns(ann_ids)\n' +
                        '                  if has_valid_annotation(anno):\n' +
                        '                      ids.append(img_id)\n' +
                        '              self.ids = ids\n' +
                        '\n' +
                        '          self.json_category_id_to_contiguous_id = {\n' +
                        '              v: i + 1 for i, v in enumerate(self.coco.getCatIds())\n' +
                        '          }\n' +
                        '          self.contiguous_category_id_to_json_id = {\n' +
                        '              v: k for k, v in self.json_category_id_to_contiguous_id.items()\n' +
                        '          }\n' +
                        '          self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}\n' +
                        '          self.ann_types = ann_types\n' +
                        '          self._transforms = transforms\n' +
                        '```\n'},
                {h4_title:'__getitem__'},
                {text:'Parse and obtain the annotation information of each image in the dataset. For current mainstream algorithms such as MaskRCNN, there will be key points or instance split branches. Pet will obtain images, image annotations and image indexes according to the task switch state in the configuration file.\n'},
                {text:'```Python\n' +
                        '    def __getitem__(self, idx):\n' +
                        '        img, anno = super(COCODataset, self).__getitem__(idx)\n' +
                        '\n' +
                        '        # filter crowd annotations\n' +
                        '        # TODO might be better to add an extra field\n' +
                        '        anno = [obj for obj in anno if obj["iscrowd"] == 0]\n' +
                        '\n' +
                        '        boxes = [obj["bbox"] for obj in anno]\n' +
                        '        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes\n' +
                        '        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")\n' +
                        '\n' +
                        '        classes = [obj["category_id"] for obj in anno]\n' +
                        '        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]\n' +
                        '        classes = torch.tensor(classes)\n' +
                        '        target.add_field("labels", classes)\n' +
                        '\n' +
                        '        if \'segm\' in self.ann_types:\n' +
                        '            masks = [obj["segmentation"] for obj in anno]\n' +
                        '            masks = SegmentationMask(masks, img.size, mode=\'poly\')\n' +
                        '            target.add_field("masks", masks)\n' +
                        '\n' +
                        '        if \'keypoint\' in self.ann_types:\n' +
                        '            if anno and "keypoints" in anno[0]:\n' +
                        '                keypoints = [obj["keypoints"] for obj in anno]\n' +
                        '                keypoints = PersonKeypoints(keypoints, img.size)\n' +
                        '                target.add_field("keypoints", keypoints)\n' +
                        '\n' +
                        '        target = target.clip_to_image(remove_empty=True)\n' +
                        '\n' +
                        '        if self._transforms is not None:\n' +
                        '            img, target = self._transforms(img, target)\n' +
                        '\n' +
                        '        return img, target, idx\n' +
                        '```\n'}
            ]
        },
        {h3_title:'COCOInstanceDataset'},
        {text:'`COCOInstanceDataset` is similar to `COCODataset`, except that it inherits the `torch.utils.data.Dataset` class to complete the fetching of images and annotations for the instance.\n' +
                '\n' +
                '```Python\n' +
                '    class COCOInstanceDataset(data.Dataset):\n' +
                '        def __init__(self, ann_file, root, bbox_file, separate_instance_kpts_annotations,\n' +
                '                     transforms=None):\n' +
                '```\n'},
        {part_title:'Samplers'},
        {text:'`samplers` implements data sampling methods based on batch, iteration, distributed and random sampling based on the inheritance of `torch.utils.data.sampler`.\n'},
        {
            ul:'`DistributedSampler`: A sampler that limits data loading to a subset of datasets. Its advantages are reflected in the use of the class `torch.nn.parallel.DistributedDataParallel`. In this case, each process can pass the distributed sampler instance as a data loader sampler and load a subset of the original data set. Its initialization parameters include:'
        },
        {
            table:{
                titles:['Parameter','Parameter Explanation'],
                data:[
                    ['dataset','data set for sampling'],['num_replicas','Number of processes participating in distributed training'],['rank','Sequence number of the current process in all processes']
                ]
            }
        },
        {
          ul:'`GroupedBatchSampler`: Packages the sampler to produce a minibatch, which forces the elements from the same group to appear in batch_size size. At the same time, the mini-batch provided by the sampler will sample as much as possible according to the requirements of the original sampler. Its initialization parameters include:'
        },
        {
            table:{
                titles:['Parameter','Parameter Explanation'],
                data:[
                    ['sampler','Basic sampler for sampling'],['batch_size','Size of mini-batch'],['drop_uneven','When set to True, the sampler will ignore batches smaller than batch_size']
                ]
            }
        },
        {
            ul:['`IterationBasedBatchSampler`: Packs the BatchSampler and resamples it until it has finished sampling the specified number of iterations.',
                '`RangeSampler`: Randomly sample the data set.'
            ]
        },
        {part_title:'Structure'},
        {
            ul:[
                '`structures` defines the handling and conversion of different task annotations. The processing methods for various types of labels such as detection frames, key points, and split masks are encapsulated into Python classes.',
                '`BoxList`: This class represents a set of bounding boxes that are stored as a set of Tensor for Nx4. In order to uniquely determine the bounding box corresponding to each image, we also store the corresponding image size. At the same time, the class contains additional information specific to each bounding box, such as labels.',
                '`ImageList`: Saves images of different sizes in the list as a single tensor structure. The principle is to fill the image to the same size and store the original size of each image.',
                '`PersonKeypoints`: This class inherits the `Keypoints` class, which completes the scaling of the key points of the human body. At the same time, it defines the key point mapping for the horizontal flip operation commonly used in the detection of human key points, and realizes the reading of keypoints during data loading. Conversion.',
                '`HeatMapKeypoints`: This class is used to generate human key point heat maps. In contrast to `PersonKeypoints`, this class contains instance level human key operations.',
                '`KeypointsInstance`: encapsulates all basic operations for human instance keypoint detection tasks into one class.',
                '`BinaryMaskList`: A binary mask used to process all targets in the split task image.',
                '`PolygonInstance`: Contains a polygon that represents a single instance target mask. Its instantiated object can be a collection of polygons.',
                '`PolygonList`: Used to process all targets marked in polygons in the split task picture.',
                '`SegmentationMask`: It is used to store the split label information of all the targets in the image, including the binary mask and the polygon mask, and complete the fusion process of the annotation information extracted by `BinaryMaskList` and `PolygonList`.'
            ]
        },
        {part_title:'Transforms'},
        {text:'`transforms` provides a wealth of two image data preprocessing methods for object and instance. Pet encapsulates the preprocessing of each image and instance into a Python class and instantiates it in subsequent preprocessing.\n'},
        {ul:'Image preprocessing method for target detection:'},
        {
            table:{
                titles:['Pretreatment','Use'],
                data:[
                    ['Compose','Combine all the pre-processing methods you need to use'],
                    ['Resize','Scale image size'],
                    ['RandomHorizontalFlip','Horizontal flip (image) of images'],
                    ['ColorJitter','Dithering on image brightness, saturation, contrast'],
                    ['ToTensor','Convert images to tensor form'],
                    ['Normalize','Normalize the image (minus the mean except the variance)'],
                    ['RandomErasing','Replace pixel values in a rectangular area of the image with random values'],
                    ['SSD_ToTensor','Convert umpy-styled images to tensor form in ssd task'],
                    ['SSD_Distort','Randomly modify the image brightness, saturation, and contrast within the ssd task'],
                    ['SSD_Mirror','Horizontal flip (image) of images in ssd task'],
                    ['SSD_Init','Converting rgb image channels to bgr in ssd task'],
                    ['SSD_Resize','Scaling the image size in the ssd task'],
                    ['SSD_Normalize','Normalize the image in the ssd task (minus the mean except the variance)'],
                    ['SSD_CROP_EXPAND','The ssd task first randomly crops the image, then the cropped image is randomly complemented with 0 pixels']
                ]
            }
        },
        {ul:'Image preprocessing method for instance analysis:'},
        {
            table:{
                titles:['Pretreatment','Use'],
                data:[
                    ['Xy2Xyz','Decompose labels in the form of COCO key points x, y, v into x, y, z and vx, vy, vz'],
                    ['Box2CS','Representation of transforming bounding boxes into center points and scales'],
                    ['Scale','Scale the instance'],
                    ['Rotate','Rotating an instance'],
                    ['Flip','Horizontal flipping of images (mirror)'],
                    ['Affine','Rotate, flip, etc. applied to images via affine'],
                    ['InstanceOperate','For instance operations such as generating heatmap'],
                    ['BGR_Normalize','Channel conversion and normalization of instance images'],
                    ['ExtractTargetTensor','Extract the required tensor property values from the dataset class in the pose task for the pass of the __getitem__ function'],
                ]
            }
        },
        {part_title:'Use Cases'},
        {h3_title:'build_dataset'},
        {text:'Take the dataloader in ssd as an example. First, Pet calls the function `build_dataset` during training to get the data set and data and useful information. At the same time, according to the requirements in the configuration file, call `build_transforms` to define the data preprocessing.\n' +
                '\n' +
                '```Python\n' +
                '    def build_dataset(dataset_list, is_train=True, local_rank=0):\n' +
                '        if not isinstance(dataset_list, (list, tuple)):\n' +
                '            raise RuntimeError(\n' +
                '                "dataset_list should be a list of strings, got {}".format(dataset_list)\n' +
                '            )\n' +
                '        for dataset_name in dataset_list:\n' +
                '            assert contains(dataset_name), \'Unknown dataset name: {}\'.format(dataset_name)\n' +
                '            assert os.path.exists(get_im_dir(dataset_name)), \'Im dir \\\'{}\\\' not found\'.format(get_im_dir(dataset_name))\n' +
                '            logging_rank(\'Creating: {}\'.format(dataset_name), local_rank=local_rank)\n' +
                '\n' +
                '        transforms = build_transforms(cfg, is_train)\n' +
                '        datasets = []\n' +
                '        for dataset_name in dataset_list:\n' +
                '            args = {}\n' +
                '            args[\'root\'] = get_im_dir(dataset_name)\n' +
                '            args[\'ann_file\'] = get_ann_fn(dataset_name)\n' +
                '            args[\'remove_images_without_annotations\'] = is_train\n' +
                '            ann_types = (\'bbox\',)\n' +
                '            args[\'ann_types\'] = ann_types\n' +
                '            args[\'transforms\'] = transforms\n' +
                '            # make dataset from factory\n' +
                '            dataset = D.COCODataset(**args)\n' +
                '            datasets.append(dataset)\n' +
                '\n' +
                '        # for training, concatenate all datasets into a single one\n' +
                '        dataset = datasets[0]\n' +
                '        if len(datasets) > 1:\n' +
                '            dataset = D.ConcatDataset(datasets)\n' +
                '\n' +
                '        return dataset\n' +
                '```\n'},
        {h3_title:'Use Cases'},
        {text:'The above two functions are directly called in the training script to complete the overall data loading function. At the same time, Pet supports the use of `torch.distributed` to implement data parallelism, and uses multi-threading to complete data loading during data loading.\n' +
                ' \n' +
                ' ```Python\n' +
                '   # Create training dataset and loader\n' +
                '    train_set = build_dataset(cfg.TRAIN.DATASETS, is_train=True, local_rank=args.local_rank)\n' +
                '    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if args.distributed else None\n' +
                '    ims_per_gpu = int(cfg.TRAIN.BATCH_SIZE / args.world_size)\n' +
                '    collator = BatchCollator(0)\n' +
                '    train_loader = torch.utils.data.DataLoader(\n' +
                '        train_set,\n' +
                '        ims_per_gpu,\n' +
                '        sampler=train_sampler,\n' +
                '        num_workers=cfg.TRAIN.LOADER_THREADS,\n' +
                '        drop_last=True,\n' +
                '        collate_fn=collator,\n' +
                '        pin_memory=True\n' +
                '    )\n' +
                '```\n'}
    ],
    dataNav:[
        'build_dataset',
        'make_train_data_loader',
        {
            'Datasets':[
                'ImageFolderList',

                'COCODataset',

                'COCOInstanceDataset',
            ]
        },
        'Samplers','Structure','Transforms','Use Cases'
    ]
}