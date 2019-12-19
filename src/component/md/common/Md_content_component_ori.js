import React, { Component } from 'react';
import Help from "../Doc/Help";
import { Collapse, Icon } from 'antd';
import { Table, Button } from 'antd/lib/index';
import {system_param, screen_width,model_width, model_height,screen_scale_width,screen_scale_height} from '../../../common/parameter/parameters'

const Panel = Collapse.Panel;


let num = 0;

const style = {
    wrap:{
        width:1000,
        marginLeft:50,
        display:'flex',
        flexDirection:'column',
    },
    code_part:{
        borderWidth:2,
        borderLeftStyle:"solid",
        borderColor:"#F78E3D",
        marginBottom:10,
        paddingLeft: 10
    },
    segmentation_part:{
        borderWidth:2,
        borderBottomStyle:"solid",
        borderColor:"#E4E4E4"
    },
    part:{
        marginTop:30,
        marginBottom:20,
        backgroundColor:'#FCE8D5',
        // height:60,
        height:'auto',
        display:'flex',
        flexDirection:'column',
        justifyContent:'center',
    },
    row_center_part:{
        marginTop:5,
        marginBottom:20,
        height:5,
        display:'flex',
        flexDirection:'row',
        justifyContent:'center',
        alignItems:'center'
    },
    img:{
        width:600
    },
    singal_img:{
        width:500
    },
    customPanelStyle:{
        background: '#E4E4E4',
        borderRadius: 4,
        // marginBottom: 24,
        border: 0,
        overflow: 'hidden',
    },
    Collapse:{
        width:800,
        alignSelf:'center',
        borderTopStyle:"solid",
        borderTopColor:"#FFA500",
        borderTopWidth:2,
        marginBottom:20
    },
    h4_block:{
        width:920,
        alignSelf:'center',
        display: "flex",
        flexDirection:"column",
        // background: '#E4E4E4',
        // borderRadius: 4,
        // padding: 20
    },
    note:{
        borderWidth:2,
        borderStyle:"dashed",
        borderColor:"#F78E3D",
        display: "flex",
        flexDirection:"column",
        padding:20,
        marginBottom:20
}
};


export default class Md_content_component extends Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {};
      }

    objToArr(obj){
        let arr = new Array();
        for (let key in obj){
            arr.push({[key]:obj[key]})
        }
        return arr
    }

    objToRender(item,type,com_key){
        let RenderComponent;
        let className = `${com_key}_`;
        if (item.className !== undefined){
            className = `${com_key}_${item['className']}`;
            // delete item['className'];
        }
        for (let key in item){
            switch (key) {
                case 'title':
                    RenderComponent = (<h1 id={item[key]} key={`${com_key}_${num}`} className={className} style={{height:45, marginBottom:20}}>{item[key]}</h1>);

                    break;
                case 'part_title':
                    RenderComponent = (
                        <div id={item[key]} style={style.part} className={className}>
                            <h2 key={`${com_key}_${num}`} style={{paddingLeft:20, color:'#F78E3D'}}>{item[key]}</h2>
                        </div>
                    );
                    break;
                case 'h3_title':
                    RenderComponent = (<h3 key={`${com_key}_${num}`} className={className} style={{height:45, marginTop:20 ,
                        paddingTop:10,
                        borderTopStyle:"solid",
                        borderTopColor:"#F6F7F8",
                        borderTopWidth:2,}}><a id={item[key]} href={`#${item[key]}`}>{item[key]}</a></h3>);
                    break;
                case 'h4_title':
                    RenderComponent = (<h4 key={`${com_key}_${num}`} className={className} style={{height:20, marginTop:20,marginBottom:5}}><a id={item[key]} href={`#${item[key]}`}>{item[key]}</a></h4>);
                    break;
                case 'h5_title':
                    RenderComponent = (<h5 key={`${com_key}_${num}`} className={className} style={{height:20, marginTop:20,
                        marginBottom:5}}><a id={item[key]} href={`#${item[key]}`}>{item[key]}</a></h5>);
                    break;
                case 'img':
                    let img_site = require(item[key]);
                    RenderComponent = (
                        type === 'block' ?
                        <div style={{marginTop:5,
                            marginBottom:20,
                            display:'flex',
                            flexDirection:'row',
                            justifyContent:'center',
                            alignItems:'center',
                            background: '#E4E4E4'}}
                             className={className}>
                            <img key={`${com_key}_${num}`} src={img_site} style={style.singal_img}/>
                        </div> : <img key={`${com_key}_${num}`} src={img_site} style={style.singal_img}/>
                    );
                    break;
                case 'text':
                    RenderComponent = (type === 'block') ? <Help className={className} key={`${com_key}_${num}`} border={true} style={{background: '#E4E4E4',marginBottom:5}} markdown_text={item[key]}/> : <div key={`${com_key}_${num}`} className={className}>
                        <Help
                            style={{marginBottom:5}} markdown_text={item[key]} />
                    </div>
                    break;
                case 'ul':
                    RenderComponent = (
                        <ul style={{marginLeft:50, marginBottom:5}}>
                            <li type="disc"><Help markdown_text={item[key]}/></li>
                        </ul>
                    );
                    break;
                case 'shell':
                    RenderComponent = <Help  className={className} key={`${com_key}_${num}`} type='shell' style={style.code_part} markdown_text={item[key]}/>;
                    break;
                case 'yaml':
                    RenderComponent = <Help className={className} key={`${com_key}_${num}`} type='yaml' style={style.code_part} markdown_text={item[key]}/>;
                    break;
                case 'out':
                    RenderComponent = (
                        <Help style={{marginBottom:10}} className={className} key={`${com_key}_num`} markdown_text={item[key]} text='Out' />
                    );
                    break;
                case "h4_block":
                    // let h4_block_arr = this.objToArr(item[key]);
                    let h4_block_arr = item[key];
                    let h4_block_compent = this.getChildrenToRender(h4_block_arr,'div',`${className}h4_block`)
                    RenderComponent = (
                        <div
                            key={`${com_key}_${num}`}
                            style={style.h4_block}
                        >
                            {h4_block_compent}
                        </div>
                    );
                    break;
                case "note":

                    let note_arr = this.objToArr(item[key]);
                    let note_compent = this.getChildrenToRender(note_arr,'div',`${className}note`)
                    let node_header = (
                        <div style={{color:"#F78E3D",}}>
                            Note
                        </div>
                    )
                    note_compent.splice(0,0,node_header)
                    RenderComponent = (
                        <div
                            key={`${com_key}_${num}`}
                            style={style.note}
                        >
                            {note_compent}
                        </div>
                    );
                    break;
                case 'table':
                    let table_type = 'center'
                    let table_width = 400
                    if (item['type'] !== undefined){
                        table_type = item['type'];
                    }
                    if (item['table_width'] !== undefined){
                        table_width = item['table_width'];
                    }

                    let data = item[key]["data"];
                    let titles = item[key]["titles"];
                    let columns = titles.map((item,i)=>{
                        return {
                            title: item,
                            dataIndex: item,
                            key: item,
                            render: (text, record, index) => {
                                return table_type === 'center' ? (<p style={{fontSize:15,fontWeight:500,textAlign:table_type}}>{text}</p>) : (<p style={{fontSize:15,fontWeight:500,textAlign:table_type,width:table_width}}>{text}</p>)
                            },
                        }
                    });
                    let dataSource = data.map((item,i)=>{
                        let data_obj = {}
                        for ( let i = 0; i <titles.length; i++){
                            data_obj[titles[i]] = item[i]
                        }
                        return data_obj
                    });
                    // console.log(`data--${data}`)
                    // console.log(`dataSource--${dataSource}`)
                    // console.log(`columns--${columns}`)

                    RenderComponent = (
                        <Table className={className} key={`${com_key}_num`} style={{marginBottom:20, marginTop:20}} columns={columns} dataSource={dataSource} rowClassName={'table_1'} pagination={false} />
                    );
                    break;
                case 'block':
                    let title = item[key]["title"]
                    delete item[key]['title']
                    // let block_arr = this.objToArr(item[key]);
                    let block_arr = item[key];
                    RenderComponent = (
                        <Collapse
                            key={`${com_key}_${num}`}
                            style={style.Collapse}
                            bordered={false}
                            defaultActiveKey={[`${com_key}_Panel_${num}`]}
                            expandIcon={({ isActive }) => (<div style={{paddingTop:10}}>
                                <Icon style={{color:'orange', float:'left'}} type="right-circle" rotate={isActive ? 90 : 0} />
                                <p style={{color:'orange', marginLeft:5, float:'left' ,fontSize:12}}>{isActive ? '收起' : '展开'}</p>
                            </div>)}
                        >
                            <Panel key={`${com_key}_Panel_${num}`} style={style.customPanelStyle} header={(<div style={style.row_center_part}>
                                <h3 style={{paddingTop:20,color:'#2F2376',}}>{title}</h3>
                            </div>)}>
                                {this.getChildrenToRender(block_arr, 'block',`${className}block`)}
                            </Panel>
                        </Collapse>
                    );


                    // let img_site_arr = item[key]['imgs'] ? item[key]['imgs']:[];
                    // img_site_arr.map((img,i)=>{
                    //     let img_site = require(img);
                    //     return <img src={img_site} style={style.singal_img}/>
                    // });
                    //
                    //
                    //
                    // RenderComponent = (
                    //     <Collapse
                    //         style={style.Collapse}
                    //         bordered={false}
                    //         defaultActiveKey={['1']}
                    //         expandIcon={({ isActive }) => (<div style={{paddingTop:10}}>
                    //             <Icon style={{color:'orange', float:'left'}} type="right-circle" rotate={isActive ? 90 : 0} />
                    //             <p style={{color:'orange', marginLeft:5, float:'left' ,fontSize:12}}>{isActive ? '收起' : '展开'}</p>
                    //         </div>)}
                    //     >
                    //         <Panel key="1" style={style.customPanelStyle} header={(<div style={style.row_center_part}>
                    //             <h3 style={{paddingTop:20,color:'#2F2376',}}>{item[key]["title"]}</h3>
                    //         </div>)}>
                    //
                    //             <Help border={true} style={{background: '#E4E4E4',}} markdown_text={item[key]["text"][0]}/>
                    //             <div style={{marginTop:5,
                    //                 marginBottom:20,
                    //                 display:'flex',
                    //                 flexDirection:'row',
                    //                 justifyContent:'center',
                    //                 alignItems:'center',
                    //                 background: '#E4E4E4'}}>
                    //                 {img_site_arr}
                    //             </div>
                    //             <Help border={true} style={{background: '#E4E4E4',}} markdown_text={item[key]["text"][1]}/>
                    //         </Panel>
                    //     </Collapse>
                    // );
                    break;

            }
            num ++;
            return RenderComponent
        }
    }


    getChildrenToRender(data,type,key) {

        return data.map((item,i) => {
            let RenderComponent = this.objToRender(item,type,key);
            return RenderComponent;
        });
    }


      render() {
          const { ...props } = this.props;
          const { data } = props;
          const childrenToRender = this.getChildrenToRender(
              data.dataSource,'div',data.key
          );
          console.log(childrenToRender);
          return (
              <div style={style.wrap}>
                  {childrenToRender}
              </div>
          )
      }
}