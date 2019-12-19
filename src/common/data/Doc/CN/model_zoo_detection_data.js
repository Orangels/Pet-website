// charts
export let Detect_VOC_charts_data = {
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

export let Detect_COCO_charts_data = {
    // vec top1 top5 flops params  backbone note
    // inference_time box_AP mask_AP train_mem backbone
    'Faster R-CNN':[
        [13.6,36.4,null,3.8,'R-50-FPN'],
        [11.9,38.5,null,5.7,'R-101-FPN'],
        [10.3,40.1,null,6.9,'ResneXt-101-32x4d-FPN'],
    ],
    'Mask R-CNN':[
        [10.2,37.4,34.2,3.8,'R-50-FPN'],
        [9.5,39.4,35.9,5.8,'R-101-FPN'],
        [8.3,41.1,37.1,7.1,'ResneXt-101-32x4d-FPN'],
    ],
    'DCN':[
        [7.7,41.1,37.2,4.5,'R-50-FPN'],
        [6.5,43.2,38.7,6.4,'R-101-FPN'],
        [6.6,43.4,null,7.1,'ResneXt-101-32x4d-FPN'],
    ],
    'GN':[
        [5.4,39.6,36.1,7.2,'R-50-FPN'],
        [4.8,41.5,37,8.9,'R-101-FPN'],
        [4.1,41.9,37.3,9.7,'ResneXt-101-32x4d-FPN'],
    ],
    'RetinaNet':[
        [8,35.7,null,6.8,'R-50-FPN'],
        [10.4,37.8,null,5.3,'R-101-FPN'],
        [9.3,39,null,6.7,'ResneXt-101-32x4d-FPN'],
    ],
    'RetinaNet +DCN*':[
        [7,37.7,null,7.5,'R-50-FPN'],
        [9.4,39.8,null,5.8,'R-101-FPN'],
        [8.3,40.8,null,7.3,'ResneXt-101-32x4d-FPN'],
    ],
    'RetinaNet +GN *':[
        [8.2,37.8,null,7.6,'R-50-FPN'],
        [10.6,39.8,null,5.9,'R-101-FPN'],
        [9.5,41.8,null,7.4,'ResneXt-101-32x4d-FPN'],
    ],
    'Cascade':[
        [7.6,40.9,35.5,5.1,'R-50-FPN'],
        [6.8,42.6,37,7.2,'R-101-FPN'],
        [6.6,44.4,38.2,8.4,'ResneXt-101-32x4d-FPN'],
    ],
    'Cascade + DCN':[
        [6.6,40.9,38.5,5.6,'R-50-FPN'],
        [5.8,42.6,39.7,7.9,'R-101-FPN'],
        [5.6,44.4,41.3,9.2,'ResneXt-101-32x4d-FPN'],
    ],
    'Cascade + DCN + GN':[
        [6.3,41.9,38,5.8,'R-50-FPN'],
        [5.5,43.6,39.3,8.2,'R-101-FPN'],
        [5.3,45.4,41.3,40.8,'ResneXt-101-32x4d-FPN'],
    ],
    'FCOS':[
        [14,37.4,null,3.8,'R-50-FPN'],
        [10,41.5,null,5.1,'R-101-FPN'],
        [7,42.7,null,9.1,'ResneXt-101-32x4d-FPN'],
    ],
    'FCOS+DCN *':[
        [10,37.9,null,4.2,'R-50-FPN'],
        [8,42.6,null,5.6,'R-101-FPN'],
        [6,43.2,null,10,'ResneXt-101-32x4d-FPN'],
    ],
    'FCOS +GN *':[
        [13,38,null,3.9,'R-50-FPN'],
        [9.6,42.9,null,5.3,'R-101-FPN'],
        [6.7,43.7,null,9.3,'ResneXt-101-32x4d-FPN'],
    ],

};


// table
export let Detect_VOC_table_data = [
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

export let Detect_COCO_table_data = [
    {
        key: '1',
        Method:'Faster R-CNN',
        Backbone: 'R-50-FPN',
        train_mem: '3.8',
        inference_time:'13.6',
        box_AP:'36.4',
        mask_AP:null,
        note:null
    },{
        key: '2',
        Method:'Faster R-CNN',
        Backbone: 'R-101-FPN',
        train_mem: '5.7',
        inference_time:'11.9',
        box_AP:'38.5',
        mask_AP:null,
        note:null
    },{
        key: '3',
        Method:'Faster R-CNN',
        Backbone: 'ResneXt-101-32x4d-FPN',
        train_mem: '6.9',
        inference_time:'10.3',
        box_AP:'40.1',
        mask_AP:null,
        note:null
    },{
        key: '4',
        Method:'Mask R-CNN',
        Backbone: 'R-50-FPN',
        train_mem: '3.8',
        inference_time:'10.2',
        box_AP:'37.4',
        mask_AP:'34.2',
        note:null
    },{
        key: '5',
        Method:'Mask R-CNN',
        Backbone: 'R-101-FPN',
        train_mem: '5.8',
        inference_time:'9.5',
        box_AP:'39.4',
        mask_AP:'35.9',
        note:null
    },{
        key: '6',
        Method:'Mask R-CNN',
        Backbone: 'ResneXt-101-32x4d-FPN',
        train_mem: '7.1',
        inference_time:'8.3',
        box_AP:'41.1',
        mask_AP:'37.1',
        note:null
    },{
        key: '7',
        Method:'DCN',
        Backbone: 'R-50-FPN',
        train_mem: '4.5(3.9)',
        inference_time:'7.7(10.2)',
        box_AP:'41.1(40)',
        mask_AP:'37.2',
        note:null
    },{
        key: '8',
        Method:'DCN',
        Backbone: 'R-101-FPN',
        train_mem: '6.4',
        inference_time:'6.5(8.0)',
        box_AP:'43.2(42.1)',
        mask_AP:'38.7',
        note:null
    },{
        key: '9',
        Method:'DCN',
        Backbone: 'ResneXt-101-32x4d-FPN',
        train_mem: '7.1',
        inference_time:'6.6',
        box_AP:'43.4',
        mask_AP:null,
        note:'dconv(c3-c5)  / mask-rcnn'
    },{
        key: '10',
        Method:'GN',
        Backbone: 'R-50-FPN',
        train_mem: '7.2(5.5)',
        inference_time:'5.4(8.4)',
        box_AP:'39.6(38.2)',
        mask_AP:'36.1',
        note:null
    },{
        key: '11',
        Method:'GN',
        Backbone: 'R-101-FPN',
        train_mem: '8.9(6.1)',
        inference_time:'4.8(7.3)',
        box_AP:'41.5(39.7)',
        mask_AP:37,
        note:null
    },{
        key: '12',
        Method:'GN',
        Backbone: 'ResneXt-101-32x4d-FPN',
        train_mem: '9.7(7.1)',
        inference_time:'4.1(6.8)',
        box_AP:'41.9(40.1)',
        mask_AP:'37.3',
        note:null
    },{
        key: '13',
        Method:'RetinaNet',
        Backbone: 'R-50-FPN',
        train_mem: '6.8',
        inference_time:'8',
        box_AP:'35.7',
        mask_AP:null,
        note:null
    },{
        key: '14',
        Method:'RetinaNet',
        Backbone: 'R-101-FPN',
        train_mem: '5.3',
        inference_time:'10.4',
        box_AP:'37.8',
        mask_AP:null,
        note:null
    },{
        key: '15',
        Method:'RetinaNet',
        Backbone: 'ResneXt-101-32x4d-FPN',
        train_mem: '6.7',
        inference_time:'9.3',
        box_AP:'39',
        mask_AP:null,
        note:null
    },{
        key: '16',
        Method:'RetinaNet + DCN*',
        Backbone: 'R-50-FPN',
        train_mem: '7.5',
        inference_time:'7',
        box_AP:'37.7',
        mask_AP:null,
        note:null
    },{
        key: '17',
        Method:'RetinaNet + DCN*',
        Backbone: 'R-101-FPN',
        train_mem: '5.8',
        inference_time:'9.4',
        box_AP:'39.8',
        mask_AP:null,
        note:null
    },{
        key: '18',
        Method:'RetinaNet + DCN*',
        Backbone: 'ResneXt-101-32x4d-FPN',
        train_mem: '7.3',
        inference_time:'8.3',
        box_AP:'40.8',
        mask_AP:null,
        note:null
    },
    // {
    //     key: '19',
    //     Method:'RetinaNet + GN *',
    //     Backbone: 'R-50-FPN',
    //     train_mem: '7.6',
    //     inference_time:'8.2',
    //     box_AP:'37.8',
    //     mask_AP:null,
    //     note:null
    // },{
    //     key: '20',
    //     Method:'RetinaNet + GN *',
    //     Backbone: 'R-101-FPN',
    //     train_mem: '5.9',
    //     inference_time:'10.6',
    //     box_AP:'39.8',
    //     mask_AP:null,
    //     note:null
    // },{
    //     key: '21',
    //     Method:'RetinaNet + GN *',
    //     Backbone: 'ResneXt-101-32x4d-FPN',
    //     train_mem: '7.4',
    //     inference_time:'9.5',
    //     box_AP:'41.8',
    //     mask_AP:null,
    //     note:null
    // },{
    //     key: '22',
    //     Method:'Cascade',
    //     Backbone: 'R-50-FPN',
    //     train_mem: '5.1(4)',
    //     inference_time:'7.6(9.7)',
    //     box_AP:'40.9(39.7)',
    //     mask_AP:'35.5',
    //     note:null
    // },{
    //     key: '23',
    //     Method:'Cascade',
    //     Backbone: 'R-101-FPN',
    //     train_mem: '7.2(6.0)',
    //     inference_time:'6.8(10.3)',
    //     box_AP:'42.6(42.0)',
    //     mask_AP:'37',
    //     note:null
    // },{
    //     key: '24',
    //     Method:'Cascade',
    //     Backbone: 'ResneXt-101-32x4d-FPN',
    //     train_mem: '8.4(7.2)',
    //     inference_time:'6.6(8.9)',
    //     box_AP:'44.4(43.6)',
    //     mask_AP:'38.2',
    //     note:null
    // },{
    //     key: '25',
    //     Method:'Cascade R-CNN +DCN',
    //     Backbone: 'R-50-FPN',
    //     train_mem: '5.6(4.4)',
    //     inference_time:'6.6(8.7)',
    //     box_AP:'40.9(39.7)',
    //     mask_AP:'38.5',
    //     note:null
    // },{
    //     key: '26',
    //     Method:'Cascade + DCN',
    //     Backbone: 'R-101-FPN',
    //     train_mem: '7.9(6.6)',
    //     inference_time:'5.8(9.3)',
    //     box_AP:'42.6(42.0)',
    //     mask_AP:'39.7',
    //     note:null
    // },{
    //     key: '27',
    //     Method:'Cascade + DCN',
    //     Backbone: 'ResneXt-101-32x4d-FPN',
    //     train_mem: '9.2(7.9)',
    //     inference_time:'5.6(7.9)',
    //     box_AP:'44.4(43.6)',
    //     mask_AP:'41.3',
    //     note:null
    // },{
    //     key: '28',
    //     Method:'Cascade + DCN + GN',
    //     Backbone: 'R-50-FPN',
    //     train_mem: '5.8(4.7)',
    //     inference_time:'6.3(8.4)',
    //     box_AP:'41.9(40.3)',
    //     mask_AP:'38',
    //     note:null
    // },{
    //     key: '29',
    //     Method:'Cascade + DCN + GN',
    //     Backbone: 'R-101-FPN',
    //     train_mem: '8.2(6.8)',
    //     inference_time:'5.5(9.0)',
    //     box_AP:'43.6(43.0)',
    //     mask_AP:'39.3',
    //     note:null
    // },{
    //     key: '30',
    //     Method:'Cascade + DCN + GN',
    //     Backbone: 'ResneXt-101-32x4d-FPN',
    //     train_mem: '9.8(8.7)',
    //     inference_time:'5.3(7.5)',
    //     box_AP:'45.4(44.6)',
    //     mask_AP:'40.8',
    //     note:null
    // },{
    //     key: '31',
    //     Method:'FCOS',
    //     Backbone: 'R-50-FPN',
    //     train_mem: '3.8',
    //     inference_time:'14',
    //     box_AP:'37.4',
    //     mask_AP:null,
    //     note:null
    // },{
    //     key: '32',
    //     Method:'FCOS',
    //     Backbone: 'R-101-FPN',
    //     train_mem: '5.1',
    //     inference_time:'10',
    //     box_AP:'41.5',
    //     mask_AP:null,
    //     note:null
    // },{
    //     key: '33',
    //     Method:'FCOS',
    //     Backbone: 'ResneXt-101-32x4d-FPN',
    //     train_mem: '9.1',
    //     inference_time:'7',
    //     box_AP:'42.7',
    //     mask_AP:null,
    //     note:null
    // },{
    //     key: '34',
    //     Method:'FCOS + DCN *',
    //     Backbone: 'R-50-FPN',
    //     train_mem: '4.2',
    //     inference_time:'10',
    //     box_AP:'37.9',
    //     mask_AP:null,
    //     note:null
    // },{
    //     key: '35',
    //     Method:'FCOS + DCN *',
    //     Backbone: 'R-101-FPN',
    //     train_mem: '5.6',
    //     inference_time:'8',
    //     box_AP:'42.6',
    //     mask_AP:null,
    //     note:null
    // },{
    //     key: '36',
    //     Method:'FCOS + DCN *',
    //     Backbone: 'ResneXt-101-32x4d-FPN',
    //     train_mem: '10',
    //     inference_time:'6',
    //     box_AP:'43.2',
    //     mask_AP:null,
    //     note:null
    // },{
    //     key: '37',
    //     Method:'FCOS + GN *',
    //     Backbone: 'R-50-FPN',
    //     train_mem: '3.9',
    //     inference_time:'13',
    //     box_AP:'38',
    //     mask_AP:null,
    //     note:null
    // },{
    //     key: '38',
    //     Method:'FCOS + GN *',
    //     Backbone: 'R-101-FPN',
    //     train_mem: '5.3',
    //     inference_time:'9.6',
    //     box_AP:'42.9',
    //     mask_AP:null,
    //     note:null
    // },{
    //     key: '39',
    //     Method:'FCOS + GN *',
    //     Backbone: 'ResneXt-101-32x4d-FPN',
    //     train_mem: '9.3',
    //     inference_time:'6.7',
    //     box_AP:'43.7',
    //     mask_AP:null,
    //     note:null
    // },
];

export let Detect_VOC_detail_table_data = [];

export let Detect_COCO_detail_table_data = [
    ['faster_rcnn_R-50-C4_1x','34.6/55.3/37.0/17.5/39.2/47.8'],
    ['faster_rcnn_R-50-C5-2FC_1x','34.3/55.6/36.3/17.0/37.9/46.8'],
    ['faster_rcnn_R-50-FPN_1x','36.8/58.4/39.9/21.0/39.7/48.1'],
    ['faster_rcnn_R-50-FPN_2x','37.7/59.1/40.8/21.4/40.7/49.4'],
    ['faster_rcnn_R-101-C4_1x','38.0/59.2/41.0/19.6/42.6/52.8'],
    ['faster_rcnn_R-101-FPN_1x', '39.0/61.0/42.3/22.8/42.7/50.7'],
    ['faster_rcnn_R-101-FPN_2x', '39.7/61.2/43.1/22.3/43.1/52.5'],
    ['faster_rcnn_R-152-FPN_1x','39.9/61.6/43.7/22.7/43.6/52.1'],

    ['faster_rcnn_A-R-50-FPN_1x', '39.7/61.8/43.1/23.3/43.1/51.4'],
    ['faster_rcnn_A-R-101-FPN_1x','42.0/64.0/45.8/25.1/45.9/54.6'],
    ['faster_rcnn_A-RX-50-32x4d-FPN_1x','40.5/63.1/43.4/24.1/44.0/52.4'],
    ['faster_rcnn_SE-A-R-50-FPN_1x', '40.4/62.6/44.2/23.8/44.0/51.9'],
    ['faster_rcnn_A-R-50-DCN@C345-FPN_1x', '42.4/64.5/46.4/25.9/45.8/55.2'],
    ['faster_rcnn_R-50-DCN@C5-FPN_1x', '38.8/60.8/42.1/22.9/41.8/51.2'],
    ['faster_rcnn_R-50-DCN@C345-FPN_1x', '40.2/62.4/43.8/24.2/43.2/53.7'],
    ['faster_rcnn_R-50-DCN@C345-FPN_2x', '40.5/62.3/43.9/24.7/43.2/53.6'],
    ['faster_rcnn_R-50-DCN@C345-FPN_2x_ms', '41.7/63.9/45.6/26.3/44.9/54.4'],
    ['faster_rcnn_R-50-DCN@C345-FPN-4CONV1FC-GN_1x', '40.1/61.9/43.5/22.9/43.0/53.0'],
    ['faster_rcnn_R-50-MDCN@C5-FPN_1x', '39.3/61.5/42.8/23.2/42.1/51.9'],
    ['faster_rcnn_R-50-MDCN@C345-FPN_1x', '40.3/62.4/44.1/24.4/43.4/53.3'],
    ['faster_rcnn_R-50-MDCN@C345-FPN_2x', '40.3/62.0/44.0/23.6/43.2/53.6'],
    ['faster_rcnn_R-101-DCN@C345-FPN_1x', '41.8/63.9/45.7/24.6/45.8/54.7'],

    ['faster_rcnn_R-50-FPN-2FC-GN_1x', '37.3/59.7/40.2/22.5/40.3/47.9'],
    ['faster_rcnn_R-50-FPN-2FC-GN_2x', '37.0/59.5/39.6/22.1/40.0/47.3'],
    ['faster_rcnn_R-50-FPN-4CONV1FC-GN_1x', '38.0/59.5/41.1/22.2/40.7/49.0'],
    ['faster_rcnn_R-50-FPN-4CONV1FC-GN_2x', '39.0/59.9/42.5/22.7/41.9/50.7'],
    ['faster_rcnn_R-50-GN-FPN_1x', '37.0/59.5/39.6/22.1/40.0/47.3'],

    ['cascade_rcnn_A-R-50-FPN_1x', '42.6/61.5/46.6/24.9/46.2/56.0'],
    ['cascade_rcnn_R-50-DCN@C345-FPN_1x', '42.7/61.9/46.4/25.1/46.8/56.4'],
    ['cascade_rcnn_R-50-FPN_1x', '40.3/58.5/44.0/22.7/43.1/54.0'],
    ['cascade_rcnn_R-50-FPN_2x', '40.2/58.5/43.8/22.4/43.2/53.7']
];
