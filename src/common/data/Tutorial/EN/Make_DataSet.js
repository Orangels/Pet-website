export let Make_DataSet = {
    key: 'Make_DataSet',
    dataSource: [
        {title:'Prepare Datasets'},
        {text:'This tutorial introduces how to download source files from official websites of public datasets and prepare data sets for training and testing according to Pet\'s standards on data preparation. Pet also provides visualization examples and annotations of relevant datasets.\n'},
        {part_title:'MSCOCO'},
        {h3_title:'Dataset Preparation'},
        {text:'Microsoft\'s MSCOCO database is a large image dataset designed for object detection, instance segmentation, human keypoint detection, semantic segmentation and caption generation.\n'},
        {img:'introduce_coco'},
        {text:'The website of the MSCOCO database::\n'},
        {
            ul: [
                'MSCOCO dataset homepage：[http://mscoco.org/](http://mscoco.org/)',
                'Github：[https://github.com/cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)'
            ]
        },
        {text:'[cocoapi](https://github.com/cocodataset/cocoapi) provides APIs of MATLAB, Python and Lua. The APIs can provide complete loading, analysis and visualization of image label data. In addition, the website also provides data related articles, tutorials and so on.\n' +
                'Before using API and demo provided by MSCOCO database, it is necessary to download MSCOCO image and label data (category flag, category number differentiation, pixel level segmentation, etc.). Pet needs to download the following contents from MSCOCO:\n'},
        {
            table:{
                titles:['File name','Size','SHA-1'],
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
                'Download image data to `$Pet/data/coco/images/`.',
                'Download annotation data to `$Pet/data/coco/annotations/`.'
            ]
        },
        {text:'Pet provides a script for users to download and extract MSCOCO dataset. Users can complete the preparation of coco dataset by downloading and running [download_coco.py](). The user completes the dataset processing by running the following commands in the terminal.\n'},
        {shell:'```\n' +
                ' python3 coco.py –dataset_dir $download path -target_dir $save path\n' +
                '```\n'},
        {h3_title: 'Visualization'},
        {img:'Data_prepare_000000016823'},
        {part_title:'DensePose-COCO'},
        {h3_title:'Dataset Preparation'},
        {text:'FAIR provides a dense human pose estimation data set based on COCO, which maps all human pixels of a RGB image to the 3D surface of the human body. This tutorial helps you download Densepose-COCO dataset and store them as required. First, we need to download the image of MSCOCO dataset, referring specifically to [MSCOCO dataset preparation](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87/Prepare%20Datasets.md#mscoco)\n'},
        {img:'D046BCDCA28DDD9F8E9E468912D41733C154F794_size130_w1080_h550'},
        {text:'Pet needs to download the following contents from [DensePose-COCO](http://densepose.org/)：'},
        {
            ul:[
                'Download image data to `$Pet/data/DensePose/images`',
                'Download annotation data to `$Pet/data/DensePose/annotations/`'
            ]
        },
        {
            table:{
                titles:['File name','Size'],
                data:[
                    ['[densepose_coco_2014_train.json](https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.jso)','923MB'],
                    ['[densepose_coco_2014_valminusminival.json](https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.jso)','202MB'],
                    ['[densepose_coco_2014_minival.json](https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.jso)','923MB'],
                    ['[densepose_coco_2014_test.json](https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.jso)','202MB']
                ],
                text_type:2
            }
        },
        {text:'Pet provides a script for users to download and extract DensePose dataset. You can complete the preparation of coco dataset by downloading and running [download_densepose.py]() and data set processing by running the following command in the terminal.\n'},
        {shell:'```\n' +
                'python3 densepose.py –dataset_dir $download path -target_dir $save path\n' +
                '```\n'},
        {h3_title:'Visualization'},
        {img:'Data_papre000000162581_UV'},
        {part_title:'ADE20K'},
        {h3_title:'Dataset Preparation'},
        {text:'[ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)is a scene parsing dataset, which contains 22210 pictures with annotations of 150 category objects. This tutorial helps you download ADE20K and convert dataset according to the annotation format required by Pet.\n'},
        {img:'introduce_ade20k'},
        {text:'Pet needs to download and extract the following files from ADE20K:\n'},
        {
            ul:[
                'Download image data to `$Pet/data/ade20k/images/`',
                'Download annotation data to `$Pet/data/ade20k/annotations/`'
            ]
        },
        {
            table:{
                titles:['File name','Size'],
                data:[
                    ['[ADEChallengeData2016.zip](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)','923MB'],
                    ['[release_test.zip](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip)','202MB'],
                ],
                text_type:2
            }
        },
        {text:'Pet provides a script for users to download and extract ADE20K dataset. Users can complete the preparation of ADE20K dataset by downloading and running ade20k.py and the dataset processing by running the following commands in the terminal.\n'},
        {shell:'```\n' +
                ' python3 ade20k.py –download_dir $download path -target_dir $save path\n' +
                '```\n'},
        {h3_title:'Visualization'},
        {img:'VisualizationADE'},
        {part_title:'Cityscapes'},
        {h3_title:'Dataset Preparation'},
        {text:'[Cityscapes](https://www.cityscapes-dataset.com/) is a dataset that focuses on the semantic understanding of urban street scenes. It contains video sequences recorded from street scenes in 50 different cities. The dataset contains rough annotations of 20,000 images and 5,000 high-quality pixel-level annotations. This tutorial helps you download the Cityscapes dataset and set it to the format that Pet needs.\n'},
        {img:'introduce_cityscapes'},
        {text:'Pet needs to download and extract the following files from Cityscapes:\n'},
        {
            ul:[
                'Download image data to `$Pet/data/cityscapes/images/`',
                'Download annotation data to `$Pet/data/cityscapes/annotations/`'
            ]
        },
        {
            table:{
                titles:['File name','Size'],
                data:[
                    ['[leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3)','253MB'],
                    ['[gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1)','12GB'],
                ],
                text_type:2
            }
        },
        {text:'Pet provides a script for users to download and extract Cityscapes dataset. Users can complete the preparation of Cityscapes data sets by downloading and running [cityscapes.py]() and the data set processing by running the following command in the terminal.\n'},
        {shell:'```\n' +
                'python3 cityscapes.py –dataset_dir $download path -target_dir $save path\n' +
                '```\n'},
        {h3_title:'Visualization'},
        {img:'VisualizationCityScape'},
        {part_title:'CIHP'},
        {h3_title:'Dataset Preparation'},
        {text:'[CIHP](http://sysu-hcp.net/lip/overview.php) is a data set which aims at the research of human body analysis. The dataset collects the image of human body at and provides the label of human body analysis at instance level. The CIHP dataset contains 28280 training samples, 5000 validation sets and 5000 test sets, with a total of more than 38,280 images. This tutorial helps you download CIHP dataset and convert them into the annotated format required by Pet.\n'},
        {img:'introduce_cihp'},
        {text:'Pet needs to download and extract the following files from CIHP:\n'},
        {
            ul:[
                'Download image data to `$Pet/data/CIHP/images/`',
                'Download annotation data to `$Pet/data/CIHP/annotations/`'
            ]
        },
        {
            table:{
                titles:['File name','Size'],
                data:[
                    ['[instance-level_human_parsing.tar.gz](https://pan.baidu.com/s/1nvqmZBN#list/path=%2Fsharelink2787269280-523292635003760%2FLIP%2FCIHP&parentPath=%2Fsharelink2787269280-523292635003760)','1.89GB'],
                ],
                text_type:2
            }
        },
        {text:'Pet provides a script for users to download and extract CIHP data sets. Users can complete the preparation of CIHP data sets by downloading and running cihp.py. The user completes the data set processing by running the following commands in the terminal.\n'},
        {shell:'```\n' +
                ' python3 cihp.py –dataset_dir $download path -target_dir $save path\n' +
                '```\n'},
        {h3_title:'Visualization'},
        {text:'```Python\n' +
                ' def vis_mask(img, mask, bbox_color, show_parss=False):\n' +
                '     """Visualizes a single binary mask."""\n' +
                '     img = img.astype(np.float32)\n' +
                '     idx = np.nonzero(mask)\n' +
                '\n' +
                '     border_color = cfg.VIS.SHOW_SEGMS.BORDER_COLOR\n' +
                '     border_thick = cfg.VIS.SHOW_SEGMS.BORDER_THICK\n' +
                '\n' +
                '     mask_color = bbox_color if \n' +
                '```\n'},
        {img:'VisualizationCIHP'}

    ]
}