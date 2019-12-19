import { Table, Button, Divider } from 'antd/lib/index';
import React, { Component } from 'react';
import {
    Detection_coco_content
} from './table_content'
import '../../App.css';
import './table.css'
import background from "../../asset/HOME_icon/Model_Zoo/bj.jpg";
import {screen_scale_width} from "../../common/parameter/parameters";
import {
    columns_classification, columns_detail_classification,columns_3rd_classification,
    colums_detection ,colums_detail_detection,
    detail_title_arr
} from './columns_template'

import {introduce_3rd_data} from './introduce_3rd_data'

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
    },
    title_1_style:[
        {
            fontSize:34*screen_scale_width,
            letterSpacing:0.15*screen_scale_width,
            color:'#484B4D',
            marginTop:50*screen_scale_width,
            marginBottom:40*screen_scale_width,
        },
        {
            fontSize:28*screen_scale_width,
            letterSpacing:0.13*screen_scale_width,
            color:'#484B4D',
            marginTop:50*screen_scale_width,
            marginBottom:40*screen_scale_width,
        }
    ],
}


class TableTest extends React.Component {
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            en:0,
            filteredInfo: null,
            sortedInfo: null,
            show_detail_data:false,
            data:null,
            detail_data:null,
            type: 0,
            table_content:null,
            model_btn_text:['more', '更多'],
            btn_type : {
                btn_1_type:'',
                btn_2_type:'',
                btn_3_type:'',
            },
        };

        this._btnonMouseEnter = this._btnonMouseEnter.bind(this);
        this._btnonMouseLeave = this._btnonMouseLeave.bind(this);
        this._model_more = this._model_more.bind(this);
        this.setRowClassName = this.setRowClassName.bind(this)
      }

    setRowClassName(record, index){
          // console.log(record)
        //   let num = index%6
        if (record['type']===1){
            return 'model_zoo_table_1_row'
        }
    }

    _model_more(e){
        let btn_str_arr = [
            ['pick up', '收起'],
            ['more', '更多']
        ]
        let show_btn_str = this.state.show_detail_data ? btn_str_arr[1] : btn_str_arr[0]
        console.log(show_btn_str)
        this.setState({
            show_detail_data: !this.state.show_detail_data,
            model_btn_text: show_btn_str
        },()=>{

        })

    }

    _btnonMouseEnter(event){
        // console.log(event.target.className);
        event.target.style.color = '#FFFFFF';
        if (event.target.className.indexOf('home_learn_button_1') !== -1){
            this.setState({
                btn_type:{
                    btn_1_type:'primary',
                }
            })
        }
        if (event.target.className.indexOf('home_learn_button_2') !== -1){
            this.setState({
                btn_type:{
                    btn_2_type:'primary',
                }
            })
        }
        if (event.target.className.indexOf('home_learn_button_3') !== -1){
            this.setState({
                btn_type:{
                    btn_3_type:'primary',
                }
            })
        }

    }

    _btnonMouseLeave(event){
        // console.log(`btn leave_${event.target.props.type}`);
        event.target.style.color = '#EC6730';
        if (event.target.className.indexOf('home_learn_button_1') !== -1){
            this.setState({
                btn_type:{
                    btn_1_type:'',
                }
            })
        }
        if (event.target.className.indexOf('home_learn_button_2') !== -1){
            this.setState({
                btn_type:{
                    btn_2_type:'',
                }
            })
        }
        if (event.target.className.indexOf('home_learn_button_3') !== -1){
            this.setState({
                btn_type:{
                    btn_3_type:'',
                }
            })
        }
    }


    componentDidMount() {
        let { style, ...props } = this.props;
        let detail_table_data = props.detail_table_data
        let titles = detail_title_arr[props.type]
        let dataSource = detail_table_data.map((item,i)=>{
            let data_obj = {}
            for ( let i = 0; i <titles.length-1; i++){
                data_obj[titles[i]] = item[i]
            }
            data_obj['key'] = `detail_table_key_${i}`
            data_obj['href'] = item[titles.length] || 'https://www.baidu.com/'
            return data_obj
        });
        this.setState({
            data:props.table_data,
            detail_data:dataSource,
            data_3rd:props.data_3rd,
            type:props.type,
            table_content:props.table_component,
            en:props.en
        })
    }

    componentWillReceiveProps(nextProps, nextContext) {

        let detail_table_data = nextProps.detail_table_data
        let titles = detail_title_arr[nextProps.type]

        let dataSource = detail_table_data.map((item,i)=>{
            let data_obj = {}
            for ( let i = 0; i <titles.length - 1; i++){
                data_obj[titles[i]] = item[i]
            }
            data_obj['key'] = `detail_table_key_${i}`
            data_obj['href'] = item[titles.length] || 'https://pan.baidu.com/s/1SYNAa78wgPS-5P2FRLl1qQ'
            return data_obj
        });

        console.log(dataSource)
        this.setState({
            data:nextProps.table_data,
            data_3rd:nextProps.data_3rd,
            detail_data:dataSource,
            show_detail_data: false,
            type:nextProps.type,
            table_content:nextProps.table_component,
            en:nextProps.en
        })

        // 初始化 detail btn
        this.setState({
            show_detail_data: false,
            model_btn_text: ['more', '更多'],
        },()=>{

        })
    }

    componentWillUpdate(nextProps, nextState, nextContext) {

    }

    componentDidUpdate(prevProps, prevState, snapshot) {

    }


    render() {

        let {...props } = this.props;
        let en = props.en
        let { sortedInfo, filteredInfo } = this.state;


        let columns_arr = [columns_classification, colums_detection, columns_classification, colums_detection];
        let columns_detail_arr = [columns_detail_classification, colums_detail_detection, columns_classification, colums_detail_detection];
        let columns_3rd_arr = [columns_3rd_classification]
        // let table_data = this.state.show_detail_data ? this.state.detail_data : this.state.data;
        let Table_component = this.state.table_content ? this.state.table_content: Detection_coco_content;

        return (
            <div style={{display:'flex',flexDirection:"column"}}>
                <div style={{display:'flex',flexDirection:"column"}}>
                    {/*<Table_component type={0}/>*/}
                    <Table className={'model_zoo_table_1'} columns={columns_arr[this.state.type]} dataSource={this.state.data}  pagination={false}  bordered
                           rowClassName={this.setRowClassName}/>

                    {this.state.type === 0 ?
                        <div style={{display:'flex',flexDirection:"column"}}>
                            <span style={style.title_1_style[en]}>
                                {introduce_3rd_data[en]}
                            </span>
                            <Table className={'model_zoo_table_2'} columns={columns_3rd_arr[this.state.type]} dataSource={this.state.data_3rd}  pagination={false}  style={{marginTop:20*screen_scale_width}}  bordered
                                   rowClassName={this.setRowClassName}/>
                        </div>
                         : null
                    }

                    {en === 0 ?
                        <div style={style.bj}>
                            <span style={style.text[en]}>More models and precision are given here</span>
                            <span style={style.text[en]}>Each model provides a download connection and gives a corresponding yaml,</span>
                            <span style={style.text[en]}>see the corresponding yaml file for configuration details</span>
                        </div> :
                        <div style={style.bj}>
                            <span style={style.text[en]}>这里给出了更多的模型和精度</span>
                            <span style={style.text[en]}>每个模型都提供了一个下载连接并提供了相应的yaml，有关配置细节，请参阅相应的yaml文件</span>
                        </div>
                    }
                </div>
                <div>
                    {this.state.show_detail_data ?
                        <div>
                            <Table_component type={this.state.type} en={en}/>
                            <Table columns={columns_detail_arr[this.state.type]} dataSource={this.state.detail_data}  pagination={false} bordered/>
                        </div> : null}
                </div>
                {/*<div classNetwork="table-operations" style={{alignSelf:'center', marginTop: 20*screen_scale_width}}>*/}
                {/*    <Button onClick={this._moreCell} style={{fontSize:15,fontWeight:500}}>More</Button>*/}
                {/*    <Button onClick={this._unmore} style={{fontSize:15,fontWeight:500}}>pack up</Button>*/}
                {/*</div>*/}
                <div style={{alignSelf:'center',marginBottom:80*screen_scale_width, marginTop:50*screen_scale_width}}>
                    <Button className='home_learn_button_3' type={this.state.btn_type.btn_3_type} style={{width:200*screen_scale_width, height:68*screen_scale_width, transition: '.25s all',fontSize:20*screen_scale_width,color:'#FC8732',}} size='small'
                            shape="round"
                            onClick={this._model_more}
                            type={this.state.btn_type.btn_3_type} onMouseEnter={this._btnonMouseEnter} onMouseLeave={this._btnonMouseLeave}
                    >
                        {this.state.model_btn_text[en]}
                    </Button>
                </div>
            </div>
        );
    }
}

export default TableTest;

