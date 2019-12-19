export let Home_model_zoo_component_data = [
    {
        'Classification':{
            title:'Classification',
            text:['Categorize input images into','pre-defined classes'],
        },
        'Detection':{
            title:'Detection',
            text:['Locate and label objects ','in target image'],
        },
        'Segmentation':{
            title:'Segmentation',
            text:['Pixel-level semantic segmentation']
        },
        'Human_Pose':{
            title:'Keypoints',
            text:['Recover the position of ','human joints in image']
        },
        'Face':{
            title:'Face',
            text:['Face recognition']
            // text:['Keypoints in face']
        },
        'Parsing':{
            title:'Parsing',
            text:['Instance-level human','part analysis']
        },
        'Dense_Pose':{
            title:'Dense Pose',
            text:['Mapping all human pixels','to the 3D surface']
        }
    },
    {
        'Classification':{
            title:'分类',
            text:['根据预定义的类别为输入图像分配类别标签']
        },
        'Detection':{
            title:'检测',
            text:['在图像中找到并识别图像中物体的位置和类别']
        },
        'Segmentation':{
            title:'语义分割',
            text:['将类别标签逐像素的分配给图像的每个像素点']
        },
        'Human_Pose':{
            title:'关键点',
            text:['在视频或图像中恢复人体关节点的位置']
        },
        'Face':{
            title:'人脸',
            // text:['人脸检测、人脸关键点定位、人脸识别']
            text:['人脸识别']
        },
        'Parsing':{
            title:'人体部位分割',
            text:['图像中人体各个部分的实例级分析']
        },
        'Dense_Pose':{
            title:'密集姿态',
            text:['将2D 图像坐标映射到 3D 人体表面实现动态人物的精确定位和姿态估计']
        }
    }
]

