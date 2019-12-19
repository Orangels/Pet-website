import {screen_scale_width} from "../../common/parameter/parameters";
import React from "react";


let columnKey = 0
let rowNum = 0
let url = window.location.host;

export let table_1 = [{
    title: '  ',
    dataIndex: 'space',
    key: 'space',
    width:1720*screen_scale_width/8,
    render: (text, record, index) => {
        let row_color = '#414447'
        let row_style = {display:'flex',justifyContent:'center',fontSize:22*screen_scale_width,fontWeight:500,textAlign:'center'}
        return (<span style={{...row_style,color:row_color}}>{text}</span>)
    },
}, {
    title: ({ sortOrder, filters }) => (
        <a href={'https://www.petcv.net:3000/'} target="_blank" style={{display:'flex',justifyContent:'center',fontSize:14,fontWeight:500,textAlign:'center', }}>
            Pet
        </a>
    ),
    dataIndex: 'Pet',
    key: 'Pet',
    width:1720*screen_scale_width/8,
    render: (text, record, index) => {
        return (<span target="_blank" style={{display:'flex',justifyContent:'center',fontSize:27*screen_scale_width,fontWeight:500,textAlign:'center', }}>
                    {text}
                </span>)
        },
    },
    {
        title: ({ sortOrder, filters }) => (
            <a href={'https://github.com/facebookresearch/Detectron'} target="_blank" style={{display:'flex',justifyContent:'center',fontSize:14,fontWeight:500,textAlign:'center', }}>
                Detectron
            </a>
        ),
        dataIndex: 'Detectron',
        key: 'Detectron',
        className:"About_Pet_table_1_column",
        width:1720*screen_scale_width/8,
        render: (text, record, index) => {
            return (<span style={{display:'flex',justifyContent:'center',fontSize:27*screen_scale_width,fontWeight:500,textAlign:'center',}}>
                {text}
            </span>)
        },
    },
    {
        title: ({ sortOrder, filters }) => (
            <a href={'https://github.com/open-mmlab/mmdetection'} target="_blank" style={{display:'flex',justifyContent:'center',fontSize:14,fontWeight:500,textAlign:'center', }}>
                mmdetection
            </a>
        ),
        dataIndex: 'mmdetection',
        key: 'mmdetection',
        className:"About_Pet_table_1_column",
        width:1720*screen_scale_width/8,
        render: (text, record, index) => {
            return (<span style={{display:'flex',justifyContent:'center',fontSize:27*screen_scale_width,fontWeight:500,textAlign:'center'}}>
                {text}
            </span>)
        },
    },
    {
        title: ({ sortOrder, filters }) => (
            <a href={'https://github.com/TuSimple/simpledet'} target="_blank" style={{display:'flex',justifyContent:'center',fontSize:14,fontWeight:500,textAlign:'center', }}>
                simpledet
            </a>
        ),
        dataIndex: 'simpledet',
        key: 'simpledet',
        className:"About_Pet_table_1_column",
        width:1720*screen_scale_width/8,
        render: (text, record, index) => {
            return (<span style={{display:'flex',justifyContent:'center',fontSize:27*screen_scale_width,fontWeight:500,textAlign:'center'}}>
                {text}
            </span>)
        },
    },{
        title: ({ sortOrder, filters }) => (
            <a href={'https://github.com/facebookresearch/maskrcnn-benchmark'} target="_blank" style={{display:'flex',justifyContent:'center',fontSize:14,fontWeight:500,textAlign:'center', }}>
                maskrcnn
            </a>
        ),
        dataIndex: 'maskrcnn-benchmark',
        key: 'maskrcnn-benchmark',
        className:"About_Pet_table_1_column",
        width:1720*screen_scale_width/8,
        render: (text, record, index) => {
            return (<span style={{display:'flex',justifyContent:'center',fontSize:27*screen_scale_width,fontWeight:500,textAlign:'center'}}>
                {text}
            </span>)
        },
    },{
        title: ({ sortOrder, filters }) => (
            <a href={'https://github.com/donnyyou/torchcv'} target="_blank" style={{display:'flex',justifyContent:'center',fontSize:14,fontWeight:500,textAlign:'center', }}>
                torch.cv
            </a>
        ),
        dataIndex: 'torch.cv',
        key: 'torch.cv',
        className:"About_Pet_table_1_column",
        width:1720*screen_scale_width/8,
        render: (text, record, index) => {
            return (<span style={{display:'flex',justifyContent:'center',fontSize:27*screen_scale_width,fontWeight:500,textAlign:'center'}}>
                {text}
            </span>)
        },
    },{
        title: ({ sortOrder, filters }) => (
            <a href={'https://github.com/dmlc/gluon-cv'} target="_blank" style={{display:'flex',justifyContent:'center',fontSize:14,fontWeight:500,textAlign:'center', }}>
                GluonCV
            </a>
        ),
        dataIndex: 'GluonCV',
        key: 'GluonCV',
        className:"About_Pet_table_1_column",
        width:1720*screen_scale_width/8,
        render: (text, record, index) => {
            return (<span style={{display:'flex',justifyContent:'center',fontSize:27*screen_scale_width,fontWeight:500,textAlign:'center'}}>
                {text}
            </span>)
        },
    },
];


export let table_2 = [{
    title: '  ',
    dataIndex: 'space',
    key: 'space',
    width:1720*screen_scale_width/8,
    render: (text, record, index) => {
        let row_color = '#414447'
        let row_style = {display:'flex',justifyContent:'center',fontSize:20*screen_scale_width,textAlign:'center'}
        return (<span style={{...row_style,color:row_color}}>{text}</span>)
    },
}, {
    title: ({ sortOrder, filters }) => (
        <a href={'https://www.petcv.net:3000/'} target="_blank" style={{display:'flex',justifyContent:'center',fontSize:14,fontWeight:500,textAlign:'center', }}>
            Pet
        </a>
    ),
    dataIndex: 'Pet',
    key: 'Pet',
    width:1720*screen_scale_width/8,
    render: (text, record, index) => {
        return (<span style={{display:'flex',justifyContent:'center',fontSize:22*screen_scale_width,fontWeight:500,textAlign:'center'}}>
                    {text}
                </span>)
    },
},
    {
        title: ({ sortOrder, filters }) => (
            <a href={'https://github.com/facebookresearch/Detectron'} target="_blank" style={{display:'flex',justifyContent:'center',fontSize:14,fontWeight:500,textAlign:'center', }}>
                Detectron
            </a>
        ),
        dataIndex: 'Detectron',
        key: 'Detectron',
        className:"About_Pet_table_1_column",
        width:1720*screen_scale_width/8,
        render: (text, record, index) => {
            return (<span style={{display:'flex',justifyContent:'center',fontSize:22*screen_scale_width,fontWeight:500,textAlign:'center'}}>
                {text}
            </span>)
        },
    },
    {
        title: ({ sortOrder, filters }) => (
            <a href={'https://github.com/open-mmlab/mmdetection'} target="_blank" style={{display:'flex',justifyContent:'center',fontSize:14,fontWeight:500,textAlign:'center', }}>
                mmdetection
            </a>
        ),
        dataIndex: 'mmdetection',
        key: 'mmdetection',
        className:"About_Pet_table_1_column",
        width:1720*screen_scale_width/8,
        render: (text, record, index) => {
            return (<span style={{display:'flex',justifyContent:'center',fontSize:22*screen_scale_width,fontWeight:500,textAlign:'center'}}>
                {text}
            </span>)
        },
    },
    {
        title: ({ sortOrder, filters }) => (
            <a href={'https://github.com/TuSimple/simpledet'} target="_blank" style={{display:'flex',justifyContent:'center',fontSize:14,fontWeight:500,textAlign:'center', }}>
                simpledet
            </a>
        ),
        dataIndex: 'simpledet',
        key: 'simpledet',
        className:"About_Pet_table_1_column",
        width:1720*screen_scale_width/8,
        render: (text, record, index) => {
            return (<span style={{display:'flex',justifyContent:'center',fontSize:22*screen_scale_width,fontWeight:500,textAlign:'center'}}>
                {text}
            </span>)
        },
    },{
        title: ({ sortOrder, filters }) => (
            <a href={'https://github.com/facebookresearch/maskrcnn-benchmark'} target="_blank" style={{display:'flex',justifyContent:'center',fontSize:14,fontWeight:500,textAlign:'center', }}>
                maskrcnn
            </a>
        ),
        dataIndex: 'maskrcnn-benchmark',
        key: 'maskrcnn-benchmark',
        className:"About_Pet_table_1_column",
        width:1720*screen_scale_width/8,
        render: (text, record, index) => {
            return (<span style={{display:'flex',justifyContent:'center',fontSize:22*screen_scale_width,fontWeight:500,textAlign:'center'}}>
                {text}
            </span>)
        },
    },{
        title: ({ sortOrder, filters }) => (
            <a href={'https://github.com/donnyyou/torchcv'} target="_blank" style={{display:'flex',justifyContent:'center',fontSize:14,fontWeight:500,textAlign:'center', }}>
                torch.cv
            </a>
        ),
        dataIndex: 'torch.cv',
        key: 'torch.cv',
        className:"About_Pet_table_1_column",
        width:1720*screen_scale_width/8,
        render: (text, record, index) => {
            return (<span style={{display:'flex',justifyContent:'center',fontSize:22*screen_scale_width,fontWeight:500,textAlign:'center'}}>
                {text}
            </span>)
        },
    },{
        title: ({ sortOrder, filters }) => (
            <a href={'https://github.com/dmlc/gluon-cv'} target="_blank" style={{display:'flex',justifyContent:'center',fontSize:14,fontWeight:500,textAlign:'center', }}>
                GluonCV
            </a>
        ),
        dataIndex: 'GluonCV',
        key: 'GluonCV',
        className:"About_Pet_table_1_column",
        width:1720*screen_scale_width/8,
        render: (text, record, index) => {
            return (<span style={{display:'flex',justifyContent:'center',fontSize:22*screen_scale_width,fontWeight:500,textAlign:'center'}}>
                {text}
            </span>)
        },
    },
];

function generate_table_data(data,type) {
    // key: '1',
    //     space: 'cls',
    //     Pet:'',
    //     Detection:'',
    //     mmdetection:'',
    //     simpledet:'',
    //     "maskrcnn-benchmark":'',
    //     "torch.cv":'',
    //     GluonCV:''
    let data_column = [
            ['key','space','Pet','Detectron','mmdetection','simpledet',"maskrcnn-benchmark","torch.cv",'GluonCV'],
            ['key','space','Pet','Detectron','mmdetection','simpledet',"maskrcnn-benchmark","torch.cv",'GluonCV']
    ]
    return data.map((item,i)=>{
        let data_obj = {}
        for ( let j = 0, length = data_column[type].length; j < length; j++){
            if (j===0){
                data_obj[data_column[type][j]] = `table_data_${rowNum}`
            }else {
                data_obj[data_column[type][j]] = item[j-1]
            }
        }
        rowNum++
        return data_obj
    })
}

let table_1_data_source = [
    ['classification','✓','','','','','✓','✓',],
    ['segmentation','','','','','','✓','✓',],
    ['detection','✓','✓','✓','✓','✓','✓','✓',],
    ['keypoint','✓','✓','','','✓','✓','✓',],
    ['densepose','✓','','','','','','',],
    ['parsing','✓','','','','','','',],
    ['reid','','','','','','','',],
    ['gan','','','','','','✓','',],
]
let table_2_data_source = [
    ['classification','','','','','','','',],
    ['segmentation','','','','','','','',],
    ['detection','','','','','','','',],
    ['gan','','','','','','','',],
    ['densepose','','','','','','','',],
    ['keypoint','','','','','','','',],
    ['parsing','','','','','','','',],
    ['reid','','','','','','','',],
]
let table_3_data_source = [
    ['classification','','','','','','','',],
    ['segmentation','','','','','','','',],
    ['detection','','','','','','','',],
    ['gan','','','','','','','',],
    ['densepose','','','','','','','',],
    ['keypoint','','','','','','','',],
    ['parsing','','','','','','','',],
    ['reid','','','','','','','',],
]
let table_4_data_source = [
    ['classification','','','','','','','',],
    ['segmentation','','','','','','','',],
    ['detection','','','','','','','',],
    ['gan','','','','','','','',],
    ['densepose','','','','','','','',],
    ['keypoint','','','','','','','',],
    ['parsing','','','','','','','',],
    ['reid','','','','','','','',],
]

let table_5_data_source = [
    ['单标签图像分类','✓','','','','','✓','✓',],
    ['语义分割','','','','','','✓','✓',],
    ['目标检测、实例分割','✓','✓','✓','✓','✓','✓','✓',],
    ['姿态估计','✓','✓','','','✓','✓','✓',],
    ['密集姿态估计','✓','','','','','','',],
    ['人体部位分析、人体部位检测','✓','','','','','','',],
    ['目标重实别','','','','','','','',],
    ['对抗生成网络','','','','','','✓','',],
]

let table_6_data_source = [
    ['单标签图像分类','✓','','','','','✓','✓',],
    ['语义分割','✓','','','','','✓','✓',],
    ['目标检测、实例分割','✓','✓','✓','✓','✓','✓','✓',],
    ['姿态估计','✓','✓','','','✓','✓','✓',],
    ['密集姿态估计','✓','','','','','','',],
    ['人体部位分析、人体部位检测','✓','','','','','','',],
    ['目标重实别','✓','','','','','','',],
    ['对抗生成网络','','','','','','✓','',],
]

let table_7_data_source = [
    ['单标签图像分类','✓','','','','','✓','✓',],
    ['语义分割','✓','','','','','✓','✓',],
    ['目标检测、实例分割','✓','✓','✓','✓','✓','✓','✓',],
    ['姿态估计','✓','✓','','','✓','✓','✓',],
    ['密集姿态估计','✓','','','','','','',],
    ['人体部位分析、人体部位检测','✓','','','','','','',],
    ['目标重实别','✓','','','','','','',],
    ['对抗生成网络','','','','','','✓','',],
]

let table_8_data_source = [
    ['单标签图像分类','✓','','','','','✓','✓',],
    ['语义分割','✓','','','','','✓','✓',],
    ['目标检测、实例分割','✓','✓','✓','✓','✓','✓','✓',],
    ['姿态估计','✓','✓','','','✓','✓','✓',],
    ['密集姿态估计','✓','','','','','','',],
    ['人体部位分析、人体部位检测','✓','','','','','','',],
    ['目标重实别','✓','','','','','','',],
    ['对抗生成网络','','','','','','✓','',],
]

export let table_1_data = generate_table_data(table_1_data_source,0)
export let table_2_data = generate_table_data(table_2_data_source,0)
export let table_3_data = generate_table_data(table_3_data_source,0)
export let table_4_data = generate_table_data(table_4_data_source,0)

export let table_5_data = generate_table_data(table_5_data_source,1)
export let table_6_data = generate_table_data(table_6_data_source,1)
export let table_7_data = generate_table_data(table_7_data_source,1)
export let table_8_data = generate_table_data(table_8_data_source,1)


export let table_1_data_ori = [
    {
        key: '1',
        space: 'cls',
        Pet:'',
        Detectron:'',
        mmdetection:'',
        simpledet:'',
        "maskrcnn-benchmark":'',
        "torch.cv":'',
        GluonCV:''

    },
    {
        key: '2',
        space: 'seg',
        Pet:'',
        Detectron:'',
        mmdetection:'',
        simpledet:'',
        "maskrcnn-benchmark":'',
        "torch.cv":'',
        GluonCV:''

    },
    {
        key: '3',
        space: 'det',
        Pet:'',
        Detectron:'',
        mmdetection:'',
        simpledet:'',
        "maskrcnn-benchmark":'',
        "torch.cv":'',
        GluonCV:''

    },
    {
        key: '4',
        space: 'gan',
        Pet:'',
        Detectron:'',
        mmdetection:'',
        simpledet:'',
        "maskrcnn-benchmark":'',
        "torch.cv":'',
        GluonCV:''

    },
    {
        key: '5',
        space: 'densepose',
        Pet:'',
        Detectron:'',
        mmdetection:'',
        simpledet:'',
        "maskrcnn-benchmark":'',
        "torch.cv":'',
        GluonCV:''

    },
    {
        key: '6',
        space: 'keypoint',
        Pet:'',
        Detectron:'',
        mmdetection:'',
        simpledet:'',
        "maskrcnn-benchmark":'',
        "torch.cv":'',
        GluonCV:''

    },
    {
        key: '7',
        space: 'parsing',
        Pet:'',
        Detectron:'',
        mmdetection:'',
        simpledet:'',
        "maskrcnn-benchmark":'',
        "torch.cv":'',
        GluonCV:''

    },
    {
        key: '8',
        space: 'reid',
        Pet:'',
        Detectron:'',
        mmdetection:'',
        simpledet:'',
        "maskrcnn-benchmark":'',
        "torch.cv":'',
        GluonCV:''

    },
];