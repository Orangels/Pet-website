
// charts
export let Classification_Cifar_charts_data = {
    'FCNN':[
        [1,55,9,56,0.46,18,6,"e2e_faster_rcnn_A-26-1x64d-FPN_1x_pytorch"],
        [2,25,11,21,0.65,34,9,"优"],
        [3,56,7,63,0.3,14,5,"良"],
        [4,33,7,29,0.33,16,6,"优"],
        [5,42,24,44,0.76,40,16,"优"],
        [6,82,58,90,1.77,68,33,"良"],
        [7,74,49,77,1.46,48,27,"良"],
        [8,78,55,80,1.29,59,29,"良"],
        [9,267,216,280,4.8,108,64,"重度污染"],
        [10,185,127,216,2.52,61,27,"中度污染"],
        [11,39,19,38,0.57,31,15,"优"],
        [12,41,11,40,0.43,21,7,"优"],
        [13,64,38,74,1.04,46,22,"良"],
        [14,108,79,120,1.7,75,41,"轻度污染"],
        [15,108,63,116,1.48,44,26,"轻度污染"],
        [16,33,6,29,0.34,13,5,"优"],
        [17,94,66,110,1.54,62,31,"良"],
        [18,186,142,192,3.88,93,79,"中度污染"],
        [19,57,31,54,0.96,32,14,"良"],
    ],
    'SSD':[
        [1,26,37,27,1.163,27,13,"优"],
        [2,85,62,71,1.195,60,8,"良"],
        [3,78,38,74,1.363,37,7,"良"],
        [4,21,21,36,0.634,40,9,"优"],
        [5,41,42,46,0.915,81,13,"优"],
        [6,56,52,69,1.067,92,16,"良"],
        [7,64,30,28,0.924,51,2,"良"],
        [8,55,48,74,1.236,75,26,"良"],
        [9,76,85,113,1.237,114,27,"良"],
        [10,91,81,104,1.041,56,40,"良"],
        [11,84,39,60,0.964,25,11,"良"],
        [12,64,51,101,0.862,58,23,"良"],
        [13,70,69,120,1.198,65,36,"良"],
        [14,77,105,178,2.549,64,16,"良"],
        [15,109,68,87,0.996,74,29,"轻度污染"],
        [16,73,68,97,0.905,51,34,"良"],
        [17,54,27,47,0.592,53,12,"良"],
        [18,51,61,97,0.811,65,19,"良"],
        [19,91,71,121,1.374,43,18,"良"],
        [20,73,102,182,2.787,44,19,"良"],
        [21,73,50,76,0.717,31,20,"良"],
        [22,84,94,140,2.238,68,18,"良"],
        [23,93,77,104,1.165,53,7,"良"],
        [24,99,130,227,3.97,55,15,"良"],
        [25,146,84,139,1.094,40,17,"轻度污染"],
        [26,113,108,137,1.481,48,15,"轻度污染"],
        [27,81,48,62,1.619,26,3,"良"],
        [28,56,48,68,1.336,37,9,"良"],
        [29,82,92,174,3.29,0,13,"良"],
        [30,106,116,188,3.628,101,16,"轻度污染"],
        [31,118,50,0,1.383,76,11,"轻度污染"]
    ],
    'YOLO3':[
        [1,91,45,125,0.82,34,23,"良"],
        [2,65,27,78,0.86,45,29,"良"],
        [3,83,60,84,1.09,73,27,"良"],
        [4,109,81,121,1.28,68,51,"轻度污染"],
        [5,106,77,114,1.07,55,51,"轻度污染"],
        [6,109,81,121,1.28,68,51,"轻度污染"],
        [7,106,77,114,1.07,55,51,"轻度污染"],
        [8,89,65,78,0.86,51,26,"良"],
        [9,53,33,47,0.64,50,17,"良"],
        [10,80,55,80,1.01,75,24,"良"],
        [11,117,81,124,1.03,45,24,"轻度污染"],
        [12,99,71,142,1.1,62,42,"良"],
        [13,95,69,130,1.28,74,50,"良"],
        [14,116,87,131,1.47,84,40,"轻度污染"],
        [15,108,80,121,1.3,85,37,"轻度污染"],
    ]
};

export let Classification_Image_charts_data = {
    // vec top1 top5 flops params  backbone note
    'mobilenet':[
        [4050,74.37,91.87,569, 4.24, 'mobilenet-V1-1.0'],
        [5900,77.36,93.57,317,2.59, 'mobilenet-v1-0.75'],
        [9000,77.86,93.46,150,1.34, 'mobilenet-v1-0.5']
    ],
    'crossnet':[
        [3000,77.86,94.01,569,4.24,'crossnet 1'],
        [5500,79.22,94.64,317,2.59,'crossnet 2'],
        [8300,70.94,89.83,150,1.34,'crossnet 3'],
    ],
    'vgg':[
        [940,74.65,92.08,569,4.24,'vgg13'],
        [760,77.67,93.82,317,2.59,'vgg16'],
        [640,70.94,89.83,150,1.34,'vgg19']
    ],
    'resnet':[
        [4100,79.22,94.64,569,4.24,'resnet18'],
        [1200,70.94,89.83,317,2.59,'resnet50'],
        [680,74.65,92.08,150,1.34,'resnet101']
    ],
    '32/64X4d':[
        [830,77.67,93.82,569,4.24,'resnet 50 32X4d'],
        [470,79.22,94.64,317,2.59,'resnet 101 32X4d'],
        [280,70.94,89.83,150,1.34,'resnet 101 64X4d']
    ],
    'aligned_resnet':[
        [810,74.65,92.08,569,4.24,'aligned_resnet18'],
        [450,79.22,94.64,317,2.59,'aligned_resnet50'],
        [250,79.22,94.64,150,1.34,'aligned_resnet101']
    ],
    'Se-resnet':[
        [1020,70.94,89.83,569,4.24,'Se-resnet18'],
        [690,74.65,92.08,317,2.59,'Se-resnet50'],
        [560,77.67,93.82,150,1.34,'Se-resnet101'],
    ]
};

//table
export let Classification_Cifar_table_data =  [
    {
        key: '1',
        Network: 'mobilenet-v1',
        Flops: '569',
        Params:4.24,
        Top1:29.1,
        Top5:10.1,
    },
    {
        key: '2',
        Network: 'mobilenet-v1-0.75',
        Flops: '317',
        Params:2.59,
        Top1:31.6,
        Top5:11.8
    }, {
        key: '3',
        Network: 'mobilenet-v1-0.5',
        Flops: 150,
        Params:1.34,
        Top1:36.7,
        Top5:15.1
    }, {
        key: '4',
        Network: 'mobilenet-v1-0.25',
        Flops: 41,
        Params:0.47,
        Top1:50.2,
        Top5:25.8
    }, {
        key: '5',
        Network: 'mobilenet-v1-swish',
        Flops: 569,
        Params:4.24,
        Top1:25.8,
        Top5:8.30
    },{
        key: '6',
        Network: 'mobilenet-v2',
        Flops: 300,
        Params:3.4,
        Top1:28.3,
        Top5:null
    },{
        key: '7',
        Network: 'mobilenet-v2',
        Flops: 585,
        Params:6.9,
        Top1:25.3,
        Top5:null
    },
    {
        key: '8',
        Network: 'shufflenet-2x-g3-se',
        Flops: 304,
        Params:3.18,
        Top1:33.78,
        Top5:null
    }, {
        key: '9',
        Network: 'shufflenet-2x-g3',
        Flops: 351,
        Params:3.48,
        Top1:32.98,
        Top5:null
    },
    {
        key: '10',
        Network: 'shufflenet-1.5x-g3',
        Flops: 351,
        Params:3.48,
        Top1:32.98,
        Top5:null
    }, {
        key: '11',
        Network: 'shufflenet-1x-g8',
        Flops: 351,
        Params:3.48,
        Top1:33.30,
        Top5:null
    },{
        key: '12',
        Network: 'resnet110',
        Flops: 247,
        Params:1.72,
        Top1:5.56,
        Top5:null
    }, {
        key: '13',
        Network: 'resnet110-mixup',
        Flops: 247,
        Params:1.72,
        Top1:4.99,
        Top5:null
    },
    {
        key: '14',
        Network: 'resnext29_8x64d-mixup',
        Flops: 5387,
        Params:34.5,
        Top1:2.92,
        Top5:null
    }, {
        key: '15',
        Network: 'resnext29_8x64d',
        Flops: 5387,
        Params:34.5,
        Top1:3.91,
        Top5:null
    },{
        key: '16',
        Network: 'crossnet62-e48e05',
        Flops: 304,
        Params:3.18,
        Top1:33.78,
        Top5:null
    },{
        key: '17',
        Network: 'crossnet62-e64e05',
        Flops: 351,
        Params:3.48,
        Top1:32.98,
        Top5:null
    },{
        key: '18',
        Network: 'crossnet47',
        Flops: 351,
        Params:3.48,
        Top1:32.98,
        Top5:null
    },{
        key: '19',
        Network: 'crossnet47-dropout',
        Flops: 351,
        Params:3.48,
        Top1:33.30,
        Top5:null
    },{
        key: '20',
        Network: 'nasnet-A',
        Flops: 304,
        Params:3.18,
        Top1:33.78,
        Top5:null
    },{
        key: '21',
        Network: 'nasnet-B',
        Flops: 351,
        Params:3.48,
        Top1:32.98,
        Top5:null
    },{
        key: '22',
        Network: 'nasnet-c',
        Flops: 351,
        Params:3.48,
        Top1:32.98,
        Top5:null
    },{
        key: '23',
        Network: 'pnasnet-A',
        Flops: 351,
        Params:3.48,
        Top1:33.30,
        Top5:null
    },
];

export let Classification_Image_table_data = [
    {
        key: '1',
        Network: 'mobilenet-V1-1.0',
        Flops: 569,
        Params:4.24,
        Top1:74.37,
        Top5:91.87,
        note:null
    },
    {
        key: '2',
        Network: 'mobilenet-v1-0.75',
        Flops: 317,
        Params:2.59,
        Top1:77.36,
        Top5:93.57,
        note:null
    }, {
        key: '3',
        Network: 'mobilenet-v1-0.5',
        Flops: 150,
        Params:1.34,
        Top1:77.86,
        Top5:93.46,
        note:null
    }, {
        key: '4',
        Network: 'crossnet 1',
        Flops: 569,
        Params:4.24,
        Top1:78.34,
        Top5:94.01,
        note:null
    }, {
        key: '5',
        Network: 'crossnet 2',
        Flops: 317,
        Params:2.59,
        Top1:79.22,
        Top5:94.64,
        note:null
    },{
        key: '6',
        Network: 'crossnet 3',
        Flops: 150,
        Params:1.34,
        Top1:70.94,
        Top5:89.83,
        note:null
    },
    {
        key: '7',
        Network: 'vgg13',
        Flops: 569,
        Params:4.24,
        Top1:74.65,
        Top5:92.08,
        note:null
    }, {
        key: '8',
        Network: 'vgg16',
        Flops: 317,
        Params:2.59,
        Top1:77.67,
        Top5:93.82,
        note:null
    },
    {
        key: '9',
        Network: 'vgg19',
        Flops: 150,
        Params:1.34,
        Top1:70.94,
        Top5:89.83,
        note:null
    }, {
        key: '10',
        Network: 'resnet18',
        Flops: 569,
        Params:4.24,
        Top1:79.22,
        Top5:94.64,
        note:null
    }, {
        key: '11',
        Network: 'resnet50',
        Flops: 317,
        Params:2.59,
        Top1:70.94,
        Top5:89.83,
        note:null
    },
    {
        key: '12',
        Network: 'resnet101',
        Flops: 150,
        Params:1.34,
        Top1:74.65,
        Top5:92.08,
        note:null
    }, {
        key: '13',
        Network: 'resnet 50 32X4d',
        Flops: 569,
        Params:4.24,
        Top1:77.67,
        Top5:93.82,
        note:null
    },{
        key: '14',
        Network: 'resnet 101 32X4d',
        Flops: 317,
        Params:2.59,
        Top1:79.22,
        Top5:94.64,
        note:null
    },{
        key: '15',
        Network: 'resnet 101 64X4d',
        Flops: 150,
        Params:1.34,
        Top1:70.94,
        Top5:89.83,
        note:null
    },{
        key: '16',
        Network: 'aligned_resnet18',
        Flops: 569,
        Params:4.24,
        Top1:74.65,
        Top5:92.08,
        note:null
    },{
        key: '17',
        Network: 'aligned_resnet50',
        Flops: 317,
        Params:2.59,
        Top1:79.22,
        Top5:94.64,
        note:null
    },{
        key: '18',
        Network: 'aligned_resnet101',
        Flops: 150,
        Params:1.34,
        Top1:79.22,
        Top5:94.64,
        note:null
    },{
        key: '19',
        Network: 'Se-resnet18',
        Flops: 569,
        Params:4.24,
        Top1:70.94,
        Top5:89.83,
        note:null
    },{
        key: '20',
        Network: 'Se-resnet50',
        Flops: 317,
        Params:2.59,
        Top1:74.65,
        Top5:92.08,
        note:null
    },{
        key: '21',
        Network: 'Se-resnet101',
        Flops: 150,
        Params:1.34,
        Top1:77.67,
        Top5:93.82,
        note:null
    },
];

export let Classification_Image_detail_table_data = [
    {key: '1', Network: 'ResNet18_v1 1', Params: 4.24, 'Flops': '569', Top1: 70.93, Top5: 89.92, note: null},
    {key: '2', Network: 'ResNet34_v1 1', Params: 2.59, 'Flops': '317', Top1: 74.37, Top5: 91.87, note: null},
    {key: '3', Network: 'ResNet50_v1 1', Params: 1.34, 'Flops': '150', Top1: 77.36, Top5: 93.57, note: null},
    {key: '4', Network: 'ResNet50_v1_int8 1', Params: 4.24, 'Flops': '569', Top1: 76.86, Top5: 93.46, note: null},
    {key: '5', Network: 'ResNet101_v1 1', Params: 2.59, 'Flops': '317', Top1: 78.34, Top5: 94.01, note: null},
    {key: '6', Network: 'ResNet152_v1 1', Params: 1.34, 'Flops': '150', Top1: 79.22, Top5: 94.64, note: null},
    {key: '7', Network: 'ResNet18_v1b 1', Params: 4.24, 'Flops': '569', Top1: 70.94, Top5: 89.83, note: null},
    {key: '8', Network: 'ResNet34_v1b 1', Params: 2.59, 'Flops': '317', Top1: 74.65, Top5: 92.08, note: null},
    {key: '9', Network: 'ResNet50_v1b 1', Params: 1.34, 'Flops': '150', Top1: 77.67, Top5: 93.82, note: null},
    {key: '10', Network: 'ResNet50_v1b_gn 1', Params: 4.24, 'Flops': '569', Top1: 77.36, Top5: 93.59, note: null},
    {key: '11', Network: 'ResNet101_v1b 1', Params: 2.59, 'Flops': '317', Top1: 79.2, Top5: 94.61, note: null},
    {key: '12', Network: 'ResNet152_v1b 1', Params: 1.34, 'Flops': '150', Top1: 79.69, Top5: 94.74, note: null},
    {key: '13', Network: 'ResNet50_v1c 1', Params: 4.24, 'Flops': '569', Top1: 78.03, Top5: 94.09, note: null},
    {key: '14', Network: 'ResNet101_v1c 1', Params: 2.59, 'Flops': '317', Top1: 79.6, Top5: 94.75, note: null},
    {key: '15', Network: 'ResNet152_v1c 1', Params: 1.34, 'Flops': '150', Top1: 80.01, Top5: 94.96, note: null},
    {key: '16', Network: 'ResNet50_v1d 1', Params: 4.24, 'Flops': '569', Top1: 79.15, Top5: 94.58, note: null},
    {key: '17', Network: 'ResNet50_v1d 1', Params: 2.59, 'Flops': '317', Top1: 78.48, Top5: 94.2, note: null},
    {key: '18', Network: 'ResNet101_v1d 1', Params: 1.34, 'Flops': '150', Top1: 80.51, Top5: 95.12, note: null},
    {key: '19', Network: 'ResNet101_v1d 1', Params: 4.24, 'Flops': '569', Top1: 79.78, Top5: 94.8, note: null},
    {key: '20', Network: 'ResNet152_v1d 1', Params: 2.59, 'Flops': '317', Top1: 80.61, Top5: 95.34, note: null},
    {key: '21', Network: 'ResNet152_v1d 1', Params: 1.34, 'Flops': '150', Top1: 80.26, Top5: 95, note: null},
    {key: '22', Network: 'ResNet18_v2 2', Params: 4.24, 'Flops': '569', Top1: 71, Top5: 89.92, note: null},
    {key: '23', Network: 'ResNet34_v2 2', Params: 2.59, 'Flops': '317', Top1: 74.4, Top5: 92.08, note: null},
    {key: '24', Network: 'ResNet50_v2 2', Params: 1.34, 'Flops': '150', Top1: 77.11, Top5: 93.43, note: null},
    {key: '25', Network: 'ResNet101_v2 2', Params: 4.24, 'Flops': '569', Top1: 78.53, Top5: 94.17, note: null},
    {key: '26', Network: 'ResNet152_v2 2', Params: 2.59, 'Flops': '317', Top1: 79.32, Top5: 94.53, note: null},
    {key: '27', Network: 'ResNext50_32x4d 12', Params: 1.34, 'Flops': '150', Top1: 80.37, Top5: 95.06, note: null},
    {key: '28', Network: 'ResNext101_32x4d 12', Params: 4.24, 'Flops': '569', Top1: 80.69, Top5: 95.17, note: null},
    {key: '29', Network: 'ResNext101_64x4d_v1 12', Params: 2.59, 'Flops': '317', Top1: 79.95, Top5: 94.93, note: null},
    {key: '30', Network: 'SE_ResNext50_32x4d ', Params: 2.59, 'Flops': '150', Top1: 80.91, Top5: 95.39, note: null},
    {key: '31', Network: 'SE_ResNext101_32x4d', Params: 1.34, 'Flops': '317', Top1: 81.01, Top5: 80.91, note: null},
    {key: '32', Network: 'SE_ResNext101_64x4d', Params: 4.24, 'Flops': '150', Top1: 80.91, Top5: 80.91, note: null}
];

export let Classification_Cifar_detail_table_data = [];