export let load_data_data = {
    key: 'load_data_data',
    dataSource: [
        {title:'数据加载'},
        {text:'数据加载是在深度学习模型训练过程中的关键环节，Pet为数据加载过程提供了完整且高效的读取加预处理的加载方式。在使用时，Pet通过调用`build_dataset`和`make_train_data_loader`两个函数完成数据的加载过程。\n'},
        {part_title:'build_dataset'},
        {text:'以ssd中的dataloader使用为例，首先Pet在训练过程中调用`build_dataset`这一函数得到所需使用的数据集及数据及中的有用信息。同时按照配置文件中的要求调用`build_transforms`定义数据的预处理方式。\n'},
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
        {text:'在数据集构建完毕后利用torch提供的数据加载类`torch.utils.data.DataLoader`完成Pet的数据载入，同时对数据进行预处理。Pet为用户提供丰富的图像预处理方式，详情参考[transforms](#transforms)介绍：\n'},
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
        {text:'需要注意的是，ssd、pose等任务在数据载入时图片大小均确定且一致，但由于两阶段目标检测任务rcnn的特殊性，Pet将rcnn任务下的数据采样器实现单独提出，以函数`make_batch_data_sampler`的形式置于[$Pet/pet/rcnn/datasets/dataset.py](https://github.com/BUPT-PRIV/Pet/blob/master/pet/rcnn/datasets/dataset.py)下。其原因为在两阶段检测器的实现过程中，我们需要以每个batch中最大的图为基准，将其按比例缩放来确定所需张量维度，对于同一批次内其余图片在缩放的基础上在右侧和下侧进行补边操作，为了减少补边带来的计算量的增加，我们将图片分为**宽大于高**与**高大于宽**两种情况，将属于同一种类型的图片采集到同一个批次中进行数据加载。\n' },
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
                'Pet将所有任务共用的数据加载组件置于`$Pet/pet/utils/data`下，包含以下内容：\n' +
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
        {text:'针对不同任务的特殊性，`datasets`提供了针对分类、目标检测和实例分析的三种数据读取方式，以`ImageFolderList`,`COCODataset`和`COCOInstanceDataset`三个独立的类来呈现，在后续不同任务的实现过程中，通过继承这两个类完成图像和标注数据的读取与载入。\n'},
        {h3_title:'ImageFolderList'},
        {text:'Pet针对图像分类任务的数据加载提供了获取图像和lable的类`ImageFolderList`,用户通过按照数据制备中分类任务要求的格式进行数据存放后，调用此类完成分类任务的数据加载过程。\n'},
        {h3_title:'初始化'},
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
        {text:'Pet针对目标检测任务如ssd和rcnn，提供了图像级标注文件解析读取的类`COCODataset`完成数据加载过程中图像级标注的载入。\n'},
        {
            h4_block:[
                {h4_title:'Init'},
                {text:'首先我们对`COCODataset`进行初始化：`COCODataset`继承了`torchvision.datasets.coco.CocoDetection`这一类，定义初始化参数包括：\n'},
                {
                    ul:['`ann_file`:标注文件路径','`root`:图像数据路径','`remove_images_without_annotations`:是否去除无标注信息的文件',
                        '`ann_types`:标注文件类型','`transforms`:预处理方式'
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
                {text:'解析并获取数据集中每张图片的标注信息，针对现在主流算法如MaskRCNN会有关键点或实例分割的分支，Pet会根据配置文件中任务开关状态来获取图像、图像标注以及图像索引。\n'},
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
        {text:'`COCOInstanceDataset`与`COCODataset`类似，区别在于其继承了`torch.utils.data.Dataset`类完成针对实例的图像和标注的获取。\n' +
                '\n' +
                '```Python\n' +
                '    class COCOInstanceDataset(data.Dataset):\n' +
                '        def __init__(self, ann_file, root, bbox_file, separate_instance_kpts_annotations,\n' +
                '                     transforms=None):\n' +
                '```\n'},
        {part_title:'Samplers'},
        {text:'`samplers`在继承`torch.utils.data.sampler`的基础上实现了基于batch、iteration、分布式和随机采样等数据采样方式。\n'},
        {
            ul:'`DistributedSampler`:将数据加载限制为数据集子集的采样器。它的优势体现在和类  `torch.nn.parallel.DistributedDataParallel`结合使用。在这种情况下，每个进程可以将分布式采样器实例作为数据加载器采样器传递并加载原始数据集的一个子集。其初始化参数包括:'
        },
        {
            table:{
                titles:['参数','参数解释'],
                data:[
                    ['dataset','用于采样的数据集'],['num_replicas','参与分布式训练的进程数'],['rank','当前进程在所有进程中的序号']
                ]
            }
        },
        {
            ul:'`GroupedBatchSampler`:将采样器打包产生一个minibatch，强制将来自同一组的元素转为按照batch_size的大小输出。同时，该采样器提供的mini-batch将尽可能按照原始采样器的要求进行采样。其初始化参数包括:'
        },
        {
            table:{
                titles:['参数','参数解释'],
                data:[
                    ['sampler','用于采样的基础采样器'],['batch_size','mini-batch的大小'],['drop_uneven','当其设置为True时，采样器将忽略大小小于batch_size的批次']
                ]
            }
        },
        {
            ul:['`IterationBasedBatchSampler`：将BatchSampler打包，从中重新采样，直到完成对指定的迭代次数采样完毕为止。',
                '`RangeSampler`：对数据集进行随机采样。'
            ]
        },
        {part_title:'Structure'},
        {
            ul:[
                '`structures`定义了对不同任务标注的处理及转换方式。对各类型标注如检测框、关键点、分割掩模等的处理方式均封装为Python类。',
                '`BoxList`:这一类表示一组边界框，这些边界框存储为Nx4的一组Tensor。为了唯一确定与每张图像相应的边界框，我们还存储了对应的图像尺寸。同时类中包含特定于每个边界框的额外信息，例如标签等。',
                '`ImageList`:将列表中可能大小不同的图像保存为单个张量的结构。其原理是将图像补边到相同的大小，并将每个图像的原始进行大小存储。',
                '`PersonKeypoints`：这一类继承了`Keypoints`类，完成对人体关键点的缩放，同时针对人体关键点检测中常用的水平翻转操作定义了关键点映射，实现数据加载过程中keypoints信息读取的转换。',
                '`HeatMapKeypoints`：这一类用于生成人体关键点热图，与`PersonKeypoints`有区别的是，此类中包含实例级别的人体关键点操作。',
                '`KeypointsInstance`：将针对人体实例关键点检测任务的所有基本操作封装为一类。',
                '`BinaryMaskList`：用于处理分割任务中，图像上所有目标的二值掩模。',
                '`PolygonInstance`：包含表示单个实例目标掩模的多边形。其实例化对象可以为一组多边形的集合。',
                '`PolygonList`：用于处理分割任务中，图像上以多边形形式标注的所有目标。',
                '`SegmentationMask`：用于存储图像中所有目标的分割标注信息，其中包括二值掩模和多边形掩模，完成`BinaryMaskList`和`PolygonList`提取出的标注信息的融合过程。'
            ]
        },
        {part_title:'Transforms'},
        {text:'`transforms`提供了丰富的针对object和instance的两种图像数据预处理方式。Pet将每一种图像和实例的预处理操作封装为一个Python类，并在后续预处理时实例化。\n'},
        {ul:'针对目标检测的图像预处理方式'},
        {
            table:{
                titles:['预处理操作','用途'],
                data:[
                    ['Compose','将所有需要使用的预处理方式结合'],
                    ['Resize','对图片尺寸进行缩放'],
                    ['RandomHorizontalFlip','对图片进行水平翻转(镜像)'],
                    ['ColorJitter','对图像亮度、饱和度、对比度的抖动'],
                    ['ToTensor','将图片转换为张量形式'],
                    ['Normalize','对图片进行归一化(减均值除方差)'],
                    ['RandomErasing','把图像中一块矩形区域中的像素值替换为随机值'],
                    ['SSD_ToTensor','ssd任务中将numpy形式存储的图片转换为张量形式'],
                    ['SSD_Distort','ssd任务中对图像亮度、饱和度、对比度进行一定范围内的随机修改'],
                    ['SSD_Mirror','ssd任务中对图片进行水平翻转(镜像)'],
                    ['SSD_Init','ssd任务中将rgb形式的图像通道转为bgr'],
                    ['SSD_Resize','ssd任务中对图片尺寸进行缩放'],
                    ['SSD_Normalize','ssd任务中对图片进行归一化(减均值除方差)'],
                    ['SSD_CROP_EXPAND','ssd任务中对图像先进行随机裁剪，后将裁剪过的图片用0像素进行随机补边']
                ]
            }
        },
        {ul:'针对实例分析的图像预处理方式'},
        {
            table:{
                titles:['预处理操作','用途'],
                data:[
                    ['Xy2Xyz','将COCO关键点x,y,v形式的标注分解为x,y,z和vx,vy,vz'],
                    ['Box2CS','将边界框转化为中心点和尺度的表示方法'],
                    ['Scale','对实例进行尺度变换'],
                    ['Rotate','对实例进行旋转'],
                    ['Flip','对实例进行水平翻装(镜像)'],
                    ['Affine','旋转、翻转等操作通过affine应用于图片上'],
                    ['InstanceOperate','对于实例的操作如产生heatmap等'],
                    ['BGR_Normalize','对实例图像进行通道转换和归一化'],
                    ['ExtractTargetTensor','在pose任务中提取dataset类中所需的tensor属性值以用于__getitem__函数的传递'],
                ]
            }
        },
        {part_title:'使用案例'},
        {h3_title:'build_dataset'},
        {text:'以ssd中的dataloader使用为例，首先Pet在训练过程中调用`build_dataset`这一函数得到所需使用的数据集和数据集中的有效信息。同时按照配置文件中的要求调用`build_transforms`定义数据的预处理\n' +
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
        {h3_title:'使用案例'},
        {text:'在训练脚本中直接调用以上两个函数完成整体的数据加载功能，同时Pet支持使用`torch.distributed`实现数据并行，在数据载入时使用多线程完成数据加载。\n' +
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
        'Samplers','Structure','Transforms','使用案例'
    ]
}