import React from "react";
import { Divider } from 'antd/lib/index';
import background from "../../asset/HOME_icon/Model_Zoo/bj.jpg";
import {screen_scale_width} from "../../common/parameter/parameters";
import baidu_icon from '../../asset/HOME_icon/Model_Zoo/baidu.png'
import google_icon from '../../asset/HOME_icon/Model_Zoo/google.png'

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

const columns_classification = [{
    title: 'Network',
    dataIndex: 'Network',
    key: 'Network',
    align:'center',
    render: (text, record, index) => {
        index = Number(index)
        let href_str = record['href'] || 'https://github.com/BUPT-PRIV/Pet-dev/blob/master/cfgs/cls/imagenet/hrnet/aligned_hrnet_w48.yaml'

        return (<a href={href_str} style={style.table_text} target="_blank">{text}</a>)
    },
}, {
    title: 'Top1/Top5',
    dataIndex: 'Top1/Top5',
    key: 'Top1/Top5',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
},{
    title: 'speed',
    dataIndex: 'speed',
    key: 'speed',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
}, {
    title: 'Params',
    dataIndex: 'Params',
    key: 'Params',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
},{
    title: 'Flops',
    dataIndex: 'Flops',
    key: 'Flops',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
}, {
    title: 'download',
    dataIndex: 'download',
    align:'center',
    key: 'download',
    render: (text, record, index) => {
        return (
            <div>
                <a href={record['download']['baidu_download']} target="_blank" style={style.table_text}>
                    <img src={baidu_icon} />
                </a>
                <span style={{...style.table_text, marginLeft:10*screen_scale_width, fontWeight:500, color:'#31B7F8'}}>
                    {record['download']['baidu_code']}
                    </span>
                <Divider type="vertical" />
                <a href={'https://pan.baidu.com/s/1SYNAa78wgPS-5P2FRLl1qQ'} target="_blank" style={style.table_text}>
                    <img src={google_icon} />
                </a>
                {/*<span style={{...style.table_text, marginLeft:10*screen_scale_width, fontWeight:500, color:'#31B7F8'}}>*/}
                {/*        1234*/}
                {/*</span>*/}
            </div>
        )
    },
},
];



const columns_3rd_classification = [{
    title: 'Network',
    dataIndex: 'Network',
    key: 'Network',
    align:'center',
    render: (text, record, index) => {
        index = Number(index)
        let href_str = record['href'] || 'https://github.com/BUPT-PRIV/Pet-dev/blob/master/cfgs/cls/imagenet/hrnet/aligned_hrnet_w48.yaml'

        return (<a href={href_str} target="_blank" style={{...style.table_text}}>{text}</a>)
    },
}, {
    title: 'Top1/Top5',
    dataIndex: 'Top1/Top5',
    key: 'Top1/Top5',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
},{
    title: 'speed(s)',
    dataIndex: 'speed',
    key: 'speed',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
}, {
    title: 'Params',
    dataIndex: 'Params',
    key: 'Params',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
},{
    title: 'Flops',
    dataIndex: 'Flops',
    key: 'Flops',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
},{
    title: 'source',
    dataIndex: 'source',
    key: 'source',
    align:'center',
    render: (text, record, index) => {
        let return_str = text ? 'pre-trained model' : '--'
        return (
                <a href={record['source'][0]} target="_blank" style={{...style.table_text, padding:0}}>{record['source'][1]}</a>
        )
    },
}, {
    title: 'download',
    dataIndex: 'download',
    align:'center',
    key: 'download',
    render: (text, record, index) => {
        return (
            <div>
                <a href={record['download']['baidu_download']} target="_blank" style={style.table_text}>
                    <img src={baidu_icon} />
                </a>
                <span style={{...style.table_text, marginLeft:10*screen_scale_width, fontWeight:500, color:'#31B7F8'}}>
                    {record['download']['baidu_code']}
                    </span>
                <Divider type="vertical" />
                <a href={'https://pan.baidu.com/s/1SYNAa78wgPS-5P2FRLl1qQ'} target="_blank" style={style.table_text}>
                    <img src={google_icon} />
                </a>
                {/*<span style={{...style.table_text, marginLeft:10*screen_scale_width, fontWeight:500, color:'#31B7F8'}}>*/}
                {/*        1234*/}
                {/*</span>*/}
            </div>
        )
    },
},
];


const columns_detail_classification = [{
    title: 'Network',
    dataIndex: 'Network',
    key: 'Network',
    align:'center',
    render: (text, record, index) => {
        index = Number(index)
        let href_str = record['href'] || 'https://github.com/BUPT-PRIV/Pet-dev/blob/master/cfgs/cls/imagenet/hrnet/aligned_hrnet_w48.yaml'

        return (<a href={href_str} target="_blank" style={style.table_text}>{text}</a>)
    },
}, {
    title: 'Top1/Top5',
    dataIndex: 'Top1/Top5',
    key: 'Top1/Top5',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
}, {
    title: 'Params',
    dataIndex: 'Params',
    key: 'Params',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
},{
    title: 'Flops',
    dataIndex: 'Flops',
    key: 'Flops',
    align:'center',
    render: (text, record, index) => {
        return (<p style={style.table_text}>{text}</p>)
    },
}, {
    title: 'download',
    dataIndex: 'download',
    align:'center',
    key: 'download',
    render: (text, record, index) => {
        return (
            <div>
                <a href={record['download']['baidu_download']} target="_blank" style={style.table_text}>
                    <img src={baidu_icon} />
                </a>
                <span style={{...style.table_text, marginLeft:10*screen_scale_width, fontWeight:500, color:'#31B7F8'}}>
                    {record['download']['baidu_code']}
                    </span>
                <Divider type="vertical" />
                <a href={'https://pan.baidu.com/s/1SYNAa78wgPS-5P2FRLl1qQ'} target="_blank" style={style.table_text}>
                    <img src={google_icon} />
                </a>
            </div>
        )
    },
},
];

export { columns_classification, columns_detail_classification, columns_3rd_classification }