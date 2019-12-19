import React from "react";
import { Divider } from 'antd/lib/index';
import background from "../../asset/HOME_icon/Model_Zoo/bj.jpg";
import {screen_scale_width} from "../../common/parameter/parameters";

const style = {
    bj:{
        background:`url(${background}) no-repeat `,
        width: '100%',
        height:210*screen_scale_width,
        backgroundSize: '100% 100%',
        display:'flex',
        flexDirection:'column',
        alignItems:'center',
        justifyContent:'center',
        marginTop:50*screen_scale_width,
        marginBottom:50*screen_scale_width
    },
    text:[
        {
            // fontSize:27*screen_scale_width,
            fontSize:16,
            color: '#FFFFFF',
            letterspacing:'0.12px'
        },
        {
            fontSize:22*screen_scale_width,
            color: '#FFFFFF',
            letterspacing:'0.12px'
        }
    ],
    table_text:{
        fontSize:15*screen_scale_width,
        fontWeight:500,
        // textAlign:'center',
    }
}

const colums_detection = [{
    title: 'Method',
    dataIndex: 'Method',
    key: 'Method',
    align:'center',
    render: (text, record, index) => {
        const obj = {
            children: text,
            props: {},
        };
        if (index%3===0){
            obj.props.rowSpan = 3;
        }else {
            obj.props.rowSpan = 0;
        }
        obj.props.style = style.table_text;
        return obj;
    },
}, {
    title: 'Backbone',
    dataIndex: 'Backbone',
    key: 'Backbone',
    align:'center',
    render: (text, record, index) => {
        let href_str = record['href'] || 'https://pan.baidu.com/s/1SYNAa78wgPS-5P2FRLl1qQ'
        return (<a herf={href_str} style={{display:'flex',justifyContent:'center' ,...style.table_text}}>{text}</a>)
    },
},{
    title: 'train mem (GB)',
    dataIndex: 'train_mem',
    key: 'train_mem',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
},{
    title: 'inference time',
    dataIndex: 'inference_time',
    key: 'inference_time',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
},{
    title: 'box AP',
    dataIndex: 'box_AP',
    key: 'box_AP',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
},{
    title: 'mask_AP',
    dataIndex: 'mask_AP',
    key: 'mask_AP',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
},

];
let colums_detail_detection = [{
    title: 'Checkpoint name',
    dataIndex: 'Checkpoint_name',
    key: 'Checkpoint_name',
    align:'center',
    render: (text, record, index) => {
        let href_str = record['href'] || 'https://pan.baidu.com/s/1SYNAa78wgPS-5P2FRLl1qQ'
        return (<a href={href_str} style={{display:'flex',justifyContent:'center' ,...style.table_text}}>{text}</a>)
    },
}, {
    title: 'train mem (GB)',
    dataIndex: 'train_mem',
    key: 'train_mem',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
},{
    title: 'inference time',
    dataIndex: 'inference_time',
    key: 'inference_time',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
},{
    title: ({sortOrder, filters})=>{
        return (
            <div style={{display:"flex", flexDirection:"column", alignItems:'center', justifyContent:'center'}}>
                <span>box AP</span>
                <span>AP /AP5/AP75/ APS/ APM/ APL</span>
            </div>
        )
    },
    align:'center',
    dataIndex: 'box_AP',
    key: 'box_AP',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
},{
    title: 'mask_AP',
    dataIndex: 'mask_AP',
    key: 'mask_AP',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
},
];

export { colums_detection ,colums_detail_detection}