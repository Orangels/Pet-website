export let Make_DataSet = {
    key: 'Make_DataSet',
    dataSource: [
        {title:'数据集准备'},
        {text:'本教程介绍了如何从各大公开数据集的官方网站上下载数据集源文件并按照Pet关于数据准备的标准准备好数据集用于训练以及测试，同时Pet还提供相关数据集的样本与标注可视化。\n'},
        {part_title:'MSCOCO数据集'},
        {h3_title:'数据集准备'},
        {text:'微软发布的MSCOCO数据库是一个大型图像数据集，专为对象检测、分割、人体关键点检测、语义分割和字幕生成而设计。\n'},
        {img:'introduce_coco'},
        {text:'MSCOCO数据库的网址是:\n'},
        {
            ul: [
            'MSCOCO数据集主页：[http://mscoco.org/](http://mscoco.org/)',
                'Github网址：[https://github.com/cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)',
            ]
        },
        {text:'[cocoapi](https://github.com/cocodataset/cocoapi)提供了Matlab，Python和Lua的调用接口。该API可以提供完整的图像标签数据的加载, 分析和可视化。此外，网站还提供了数据相关的文章，教程等。\n' +
                '在使用MSCOCO数据库提供的API和demo之前, 需要首先下载MSCOCO的图像和标签数据（类别标志、类别数量区分、像素级的分割等），pet需要从MSCOCO中下载以下内容：\n'},
        {
            table:{
                titles:['文件名','大小','SHA-1'],
                data:[
                    ['[train2017.zip](http://images.cocodataset.org/zips/train2017.zip)','18GB','10ad623668ab00c62c096f0ed636d6aff41faca5'],
                    ['[val2017.zip](http://images.cocodataset.org/zips/val2017.zip)','778MB','4950dc9d00dbe1c933ee0170f5797584351d2a41'],
                    ['[annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)','241MB','8551ee4bb5860311e79dace7e79cb91e432e78b3'],
                    ['[stuff_annotations_trainval2017.zip](http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip)','401MB','e7aa0f7515c07e23873a9f71d9095b06bcea3e12']
                ],
                text_type:2
            }
        },
        {
            ul:[
                '图像数据下载到`$Pet/data/coco/images/`文件夹中',
                '标签数据下载到`$Pet/data/coco/annotations/`文件夹中'
            ]
        },
        {text:'Pet为用户提供了下载并提取MSCOCO数据集的脚本，用户可以通过下载并运行[download_coco.py]()完成coco数据集的准备。用户通过在终端中运行如下命令完成数据集的处理。\n'},
        {shell:'```\n' +
                ' python coco.py –dataset_dir $download path -target_dir $save path\n' +
                '```\n'},
        {h3_title: '数据可视化'},
        {text:'Pet可以对COCO数据的标注进行可视化：\n'},
        {img:'Data_prepare_000000016823'},
        {img:'VisualizationCOCO'},
        {part_title:'DensePose-COCO数据集'},
        {h3_title:'数据集准备'},
        {text:'FAIR提供了基于COCO的密集人体姿态估计数据集，该数据集是将一个RGB图像的所有人体像素映射到人体的3D表面。本教程帮助您下载Densepose-COCO数据集并按照要求存放。首先需要将MSCOCO数据集的图片下载完成，具体参照[MSCOCO数据集准备](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87.md#mscoco%E6%95%B0%E6%8D%AE%E9%9B%86)\n'},
        {img:'D046BCDCA28DDD9F8E9E468912D41733C154F794_size130_w1080_h550'},
        {text:'Pet需要从[DensePose-COCO](http://densepose.org/)中下载并解压下列文件：\n'},
        {
            ul:[
                '图像数据下载到 $Pet/data/DensePose/images/ 文件夹中',
                '标签数据下载到 $Pet/data/DensePose/annotations/ 文件夹中'
            ]
        },
        {
            table:{
                titles:['文件名','大小'],
                data:[
                    ['[densepose_coco_2014_train.json](https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.jso)','923MB'],
                    ['[densepose_coco_2014_valminusminival.json](https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.jso)','202MB'],
                    ['[densepose_coco_2014_minival.json](https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.jso)','923MB'],
                    ['[densepose_coco_2014_test.json](https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.jso)','202MB']
                ],
                text_type:2
            }
        },
        {text:'Pet为用户提供了下载并提取DensePose数据集的脚本，用户可以通过下载并运行[download_densepose.py]()完成coco数据集的准备。用户通过在终端中运行如下命令完成数据集的处理。\n'},
        {shell:'```\n' +
                'python densepose.py –dataset_dir $download path -target_dir $save path\n' +
                '```\n'},
        {h3_title:'数据集可视化'},
        {img:'Data_papre000000162581_UV'},
        {part_title:'ADE20K数据集'},
        {h3_title:'数据集准备'},
        {text:'[ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)是一个以场景解析数据集，该数据集包含22210幅图片，并附有150个类别对象的标注信息。本教程帮助您下载ADE20K并按照pet要求的标注格式进行数据集的转换。\n'},
        {img:'introduce_ade20k'},
        {text:'  Pet需要从ADE20K中下载并解压下列文件：\n'},
        {
            ul:[
                '图像数据下载到 $Pet/data/ade20k/images/ 文件夹中',
                '标签数据下载到 $Pet/data/ade20k/annotations/ 文件夹中'
            ]
        },
        {
            table:{
                titles:['文件名','大小'],
                data:[
                    ['[ADEChallengeData2016.zip](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)','923MB'],
                    ['[release_test.zip](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip)','202MB'],
                ],
                text_type:2
            }
        },
        {text:'Pet为用户提供了下载并提取ADE20K数据集的脚本，用户可以通过下载并运行ade20k.py完成ADE20K数据集的准备。用户通过在终端中运行如下命令完成数据集的处理。\n'},
        {shell:'```\n' +
                ' python ade20k.py –download_dir $download path -target_dir $save path\n' +
                '```\n'},
        {h3_title:'数据可视化'},
        {img:'VisualizationADE'},
        {part_title:'Cityscapes数据集'},
        {h3_title:'数据集准备'},
        {text:'[Cityscapes](https://www.cityscapes-dataset.com/)是一个侧重于城市街景语义理解的数据集。其中包含了从50个不同城市的街道场景中录制的视频序列，数据集包含20 000帧图像的粗略标注，还有5 000帧高质量像素级标注。本教程帮助您下载Cityscapes数据集，并将其设置为Pet所需格式。\n'},
        {img:'introduce_cityscapes'},
        {text:'Pet需要从Cityscapes中下载并解压下列文件：\n'},
        {
            ul:[
                '图像数据下载到 $Pet/data/cityscapes/images/ 文件夹中',
                '标签数据下载到 $Pet/data/cityscapes/annotations/ 文件夹中'
            ]
        },
        {
            table:{
                titles:['文件名','大小'],
                data:[
                    ['[leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3)','253MB'],
                    ['[gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1)','12GB'],
                ],
                text_type:2
            }
        },
        {text:'Pet为用户提供了下载并提取Cityscapes数据集的脚本，用户可以通过下载并运行[cityscapes.py]()完成Cityscapes数据集的准备。用户通过在终端中运行如下命令完成数据集的处理。\n'},
        {shell:'```\n' +
                'python cityscapes.py –dataset_dir $download path -target_dir $save path\n' +
                '```\n'},
        {h3_title:'数据集可视化'},
        {img:'VisualizationCityScape'},
        {part_title:'CIHP数据集'},
        {h3_title:'数据集准备'},
        {text:'[CIHP](http://sysu-hcp.net/lip/overview.php)是一个目标建立在多人体解析研究的数据集，数据集中收集了多人体实例级的图像，并提供了实例级人体解析的标注。CIHP数据集包含28280个训练样本、5000张验证集和5000张测试集，共有38280个多人图像。本教程帮助您下载CIHP数据集并按照pet要求的标注格式进行数据集的转换。\n'},
        {img:'introduce_cihp'},
        {text:'Pet需要从CIHP中下载并解压下列文件：\n'},
        {
            ul:[
                '图像数据下载到 $Pet/data/CIHP/images/ 文件夹中',
                '标签数据下载到 $Pet/data/CIHP/annotations/ 文件夹中'
            ]
        },
        {
            table:{
                titles:['文件名','大小'],
                data:[
                    ['[instance-level_human_parsing.tar.gz](https://pan.baidu.com/s/1nvqmZBN#list/path=%2Fsharelink2787269280-523292635003760%2FLIP%2FCIHP&parentPath=%2Fsharelink2787269280-523292635003760)','1.89GB'],
                ],
                text_type:2
            }
        },
        {text:'Pet为用户提供了下载并提取CIHP数据集的脚本，用户可以通过下载并运行cihp.py完成CIHP数据集的准备。用户通过在终端中运行如下命令完成数据集的处理。\n'},
        {shell:'```\n' +
                ' python cihp.py –dataset_dir $download path -target_dir $save path\n' +
                '```\n'},
        {h3_title:'数据集可视化'},
        {img:'VisualizationCIHP'}

    ]
}