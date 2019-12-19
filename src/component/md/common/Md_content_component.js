import React, { Component } from 'react';
import Help from "../Doc/Help";
import { Collapse, Icon } from 'antd';
import { Table, Button, Anchor } from 'antd/lib/index';
import {system_param, screen_width,model_width, model_height,screen_scale_width,screen_scale_height} from '../../../common/parameter/parameters'

import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';


import note_icon from '../../../asset/HOME_icon/2x/note_icon.png'
import footer_icon1 from "../../../asset/HOME_icon/2x/email@2x.png";
import test_img from '../Pet-DOC-master/cifar10_pic.png'

import imgs_obj from '../../../common/data/img_path_obj'

const Panel = Collapse.Panel;
const { Link } = Anchor

let num = 0;

const style = {
    wrap:{
        display:'flex',
        flexDirection:'row',
        justifyContent:'space-between'
    },
    content_wrap:[
        {
            width:1280*screen_scale_width,
            marginLeft:40*screen_scale_width,
            marginBottom:80*screen_scale_width,
            display:'flex',
            flexDirection:'column',
        },
        {
            width:1500*screen_scale_width,
            marginLeft:40*screen_scale_width,
            marginBottom:80*screen_scale_width,
            display:'flex',
            flexDirection:'column',
        }
    ],
    code_part:{
        borderWidth:2,
        borderLeftStyle:"solid",
        borderColor:"#FF8722",
        marginBottom:50*screen_scale_width,
        // paddingLeft: 10*screen_scale_width
    },
    segmentation_part:{
        borderWidth:2,
        borderBottomStyle:"solid",
        borderColor:"#E4E4E4"
    },
    part:{
        // marginTop:30*screen_scale_width,
        marginBottom:20*screen_scale_width,
        // backgroundColor:'#FCE8D5',
        // height:60,
        paddingBottom:5*screen_scale_width,
        // borderBottom:'1px solid #FF8722',
        borderBottom:'1px dotted #FF8722',
        height:'auto',
        display:'flex',
        flexDirection:'column',
        justifyContent:'center',
    },
    part_3:{
        marginTop:30*screen_scale_width,
        marginBottom:20*screen_scale_width,
        // backgroundColor:'#FCAC8D',
        // height:60,
        height:'auto',
        display:'flex',
        flexDirection:'column',
        justifyContent:'center',
    },
    row_center_part:{
        marginTop:5*screen_scale_width,
        marginBottom:20*screen_scale_width,
        height:5*screen_scale_width,
        display:'flex',
        flexDirection:'row',
        justifyContent:'center',
        alignItems:'center'
    },
    img:{
        width:600*screen_scale_width
    },
    singal_img:{
        // width:800*screen_scale_width
        marginBottom:20*screen_scale_width,
        maxWidth:1350*screen_scale_width,
    },
    customPanelStyle:{
        // background: '#E4E4E4',
        background:'#F7F7F7',
        borderRadius: 4,
        // marginBottom: 24,
        border: 0,
        overflow: 'hidden',
    },
    Collapse:{
        // width:800*screen_scale_width,
        width:1500*screen_scale_width,
        alignSelf:'center',
        borderTopStyle:"solid",
        borderTopColor:"#3765A0",
        borderTopWidth:2,
        marginBottom:30*screen_scale_width,
        marginTop:10*screen_scale_width
    },
    h4_block:{
        width:(1220-110)*screen_scale_width,
        // alignSelf:'center',
        marginLeft:95*screen_scale_width,
        display: "flex",
        flexDirection:"column",
        whiteSpace:'pre'
        // background: '#E4E4E4',
        // borderRadius: 4,
        // padding: 20
    },
    note:{
        borderWidth:1,
        borderStyle:"dashed",
        borderColor:"#FF8722",
        display: "flex",
        flexDirection:"column",
        padding:`${10*screen_scale_width}px ${20*screen_scale_width}px ${20*screen_scale_width}px ${20*screen_scale_width}px`,
        marginBottom:20*screen_scale_width,
        marginTop:50*screen_scale_width
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
                case 'katex':
                    RenderComponent = (
                    <BlockMath >{item[key]}</BlockMath>
                    )
                    break;
                case 'title':
                    RenderComponent = (<h1
                        // id={item[key].replace(/ /g,'_')}
                        key={`${com_key}_${num}`} className={`${className} Doc_title`} style={{ marginBottom:40*screen_scale_width, fontSize:40*screen_scale_width, lineHeight:'30px'}}>{item[key]}</h1>);

                    break;
                case 'part_title':
                    RenderComponent = (
                        <div id={item[key].replace(/ /g,'_')} key={`${com_key}_${num}`} style={style.part} className={className}>
                            <h2 key={`${com_key}_${num}`} style={{color:'#FF8722'}}>{item[key]}</h2>
                        </div>
                    );
                    break;
                case 'h3_title':
                    RenderComponent = (
                        <div key={`${com_key}_${num}`} id={item[key].replace(/ /g,'_')} style={style.part_3} className={className}>
                            {/*<h3 key={`${com_key}_${num}`} style={{color:'#FF8722'}}>{item[key]}</h3>*/}
                            <h3 key={`${com_key}_${num}`} style={{color:'rgba(0,0,0,0.85)'}}>{item[key]}</h3>
                        </div>
                    );
                    // RenderComponent = (<h3 key={`${com_key}_${num}`} className={className} style={{height:45*screen_scale_width, marginTop:20*screen_scale_width ,
                    //     paddingTop:10*screen_scale_width,
                    //     borderTopStyle:"solid",
                    //     borderTopColor:"#F6F7F8",
                    //     borderTopWidth:2,}}><a id={item[key]} href={`#${item[key]}`}>{item[key]}</a></h3>);
                    break;
                case 'h4_title':
                    RenderComponent = (<h4 key={`${com_key}_${num}`} className={className} style={{height:20*screen_scale_width, marginTop:20*screen_scale_width,marginBottom:20*screen_scale_width}}><span id={item[key].replace(/ /g,'_')} href={`#${item[key]}`} style={{color:'#BF6135'}}>{item[key]}</span></h4>);
                    break;
                case 'h5_title':
                    RenderComponent = (<h5 key={`${com_key}_${num}`} className={className} style={{height:20*screen_scale_width, marginTop:20*screen_scale_width,
                        marginBottom:10*screen_scale_width}}><span id={item[key].replace(/ /g,'_')}  style={{color:'#3765A0'}}>{item[key]}</span></h5>);
                    break;
                case 'table_header':
                    RenderComponent = (<span key={`${com_key}_${num}`} className={className} style={{ marginTop:20*screen_scale_width,marginBottom:20*screen_scale_width,fontSize:14,fontWeight:500}}>{item[key]}</span>);
                    break;
                case 'img':
                    let img_path = item[key]

                    // console.log(`path -- ${img_path}`)
                    // console.log(`type ${typeof img_path}`)
                    let img_site = imgs_obj[img_path]
                    RenderComponent = (
                        type === 'block' ?
                        <div style={{marginTop:5*screen_scale_width,
                            marginBottom:20*screen_scale_width,
                            display:'flex',
                            flexDirection:'row',
                            justifyContent:'center',
                            alignItems:'center',
                            background: '#F7F7F7'}}
                             className={className}>
                            <img key={`${com_key}_${num}`} src={img_site} style={style.singal_img}/>
                        </div> :
                            <img key={`${com_key}_${num}`} src={img_site} style={{...style.singal_img, alignSelf:"center"}}/>
                    );
                    break;
                case 'text':
                    let text_type = item[`type`]
                    RenderComponent = (type === 'block') ? <Help className={className} key={`${com_key}_${num}`} border={true} style={{background: '#F7F7F7',marginBottom:20*screen_scale_width, alignSelf:text_type}} markdown_text={item[key]}/> : <div key={`${com_key}_${num}`} className={className} style={{alignSelf:text_type, display:'flex', flexDirection:'column'}}>
                        <Help
                            style={{marginBottom:20*screen_scale_width, alignSelf:text_type}} markdown_text={item[key]} />
                    </div>
                    break;
                case 'ul':

                    let ul_arr = item[key]
                    if (typeof ul_arr === 'string'){

                        RenderComponent = (
                            <ul key={`${com_key}_${num}`} style={{marginLeft:50*screen_scale_width, marginBottom:25*screen_scale_width}}>
                                <li type="disc"><Help markdown_text={item[key]} /></li>
                            </ul>
                        );
                    }else {
                        let li_component = ul_arr.map((li_item,i)=>{
                            return (
                                <li type="disc" ><Help markdown_text={li_item} /></li>
                            )
                        })
                        RenderComponent = (
                            <ul key={`${com_key}_${num}`} style={{marginLeft:50*screen_scale_width, marginBottom:25*screen_scale_width}}>
                                {li_component}
                            </ul>
                        );
                    }
                    break;
                case 'shell':
                    RenderComponent = <Help  className={className} key={`${com_key}_${num}`} type='shell' style={style.code_part} markdown_text={item[key]}/>;
                    break;
                case 'yaml':
                    RenderComponent = <Help className={className} key={`${com_key}_${num}`} type='yaml' style={style.code_part} markdown_text={item[key]}/>;
                    break;
                case 'out':
                    RenderComponent = (
                        <Help style={{marginBottom:10*screen_scale_width}} className={className} key={`${com_key}_num`} markdown_text={item[key]} text='Out' type='code'/>
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

                    let note_arr = item[key];
                    let note_compent = this.getChildrenToRender(note_arr,'div',`${className}note`)
                    let node_header = (
                        <div key={`${com_key}_${num}`} style={{ marginBottom:10*screen_scale_width, display:'flex'}}>
                            <img key={'footer_icon1'} src={note_icon} style={{width:25*screen_scale_width,height:25*screen_scale_width}}/>
                            <span style={{color:"#FF8722", marginLeft:10*screen_scale_width, alignSelf:'center'}}>Note</span>
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
                    // 1 用 p, 2 用 markdown
                    let table_text_type = item[key]["text_type"] || 1
                    let table_width = 400*screen_scale_width
                    if (item['type'] !== undefined){
                        table_type = item['type'];
                    }
                    if (item['table_width'] !== undefined){
                        table_width = item['table_width'];
                        if (typeof(table_width) !== 'string'){
                            table_width = table_width*screen_scale_width
                        }
                    }

                    let data = item[key]["data"];
                    let titles = item[key]["titles"];
                    let columns = titles.map((item,i)=>{
                        return {
                            title: item,
                            dataIndex: item,
                            key: item,
                            align:'center',
                            render: (text, record, index) => {
                                if (table_text_type===1){
                                    let text_color = i === 0 ? '#A05937' : '#484B4D'
                                    if (typeof(text) == 'string'){
                                        if (text.indexOf('\n') !== -1){
                                            let text_arr = text.split('\n')
                                            text_arr = text_arr.map((text_item,text_index)=>{
                                                return table_type === 'center' ? (<p style={{fontSize:(14/0.75)*screen_scale_width,fontWeight:500,textAlign:table_type,color:text_color}}>{text_item}</p>) : (<p style={{fontSize:(14/0.75)*screen_scale_width,fontWeight:500,textAlign:table_type,color:text_color}}>{text_item}</p>)
                                            })
                                            return <div>{text_arr}</div>
                                        }else {
                                            return table_type === 'center' ? (<p style={{fontSize:(14/0.75)*screen_scale_width,fontWeight:500,textAlign:table_type,color:text_color}}>{text}</p>) : (<p style={{fontSize:(14/0.75)*screen_scale_width,fontWeight:500,textAlign:table_type,color:text_color}}>{text}</p>)
                                        }
                                    }

                                    return table_type === 'center' ? (<p style={{fontSize:(14/0.75)*screen_scale_width,fontWeight:500,textAlign:table_type}}>{text}</p>) : (<p style={{fontSize:(14/0.75)*screen_scale_width,fontWeight:500,textAlign:table_type,}}>{text}</p>)
                                }else {

                                    if (typeof(text) == 'string'){
                                        if (text.indexOf('\n') !== -1){
                                            let text_arr = text.split('\n')
                                            text_arr = text_arr.map((text_item,text_index)=>{
                                                return table_type === 'center' ? (<Help markdown_text={text_item} />) : (<Help markdown_text={text_item} />)
                                            })
                                            return <div>{text_arr}</div>
                                        }else {
                                            return table_type === 'center' ? (<Help markdown_text={text} />) : (<Help markdown_text={text} />)
                                        }
                                    }

                                    return table_type === 'center' ? (<Help markdown_text={`${text}`} />) : (<Help markdown_text={`${text}`} />)
                                }
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
                        <Table className={className} key={`${com_key}_${num}`} style={{marginBottom:25*screen_scale_width, marginTop:20*screen_scale_width}} columns={columns} dataSource={dataSource} rowClassName={`${com_key}_row_${num}`} pagination={false} bordered/>
                    );
                    break;
                case 'block':
                    let title = item[key]["title"]
                    let block_arr = item[key]['children']
                    console.log(`title-${title}`)
                    console.log(`item-${item[key]}`)
                    RenderComponent = (
                        <Collapse
                            key={`${com_key}_${num}`}
                            style={style.Collapse}
                            bordered={false}
                            defaultActiveKey={[`${com_key}_Panel_${num}`]}
                            expandIcon={({ isActive }) => (<div style={{paddingTop:10*screen_scale_width}}>
                                <Icon style={{color:'orange', float:'left'}} type="right-circle" rotate={isActive ? 90 : 0} />
                                <p style={{color:'orange', marginLeft:5*screen_scale_width, float:'left' ,fontSize:12*screen_scale_width}}>{isActive ? '收起' : '展开'}</p>
                            </div>)}
                        >
                            <Panel key={`${com_key}_Panel_${num}`} style={style.customPanelStyle} header={(<div style={style.row_center_part}>
                                <h3 style={{paddingTop:20*screen_scale_width,color:'#3765A0',}}>{title}</h3>
                            </div>)}>
                                {this.getChildrenToRender(block_arr, 'block',`${className}block`)}
                            </Panel>
                        </Collapse>
                    );
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


    getNavToRender(data) {
          return data.map((item, i)=>{
              if (typeof(item) == 'string'){
                  let new_item = item.replace(/ /g,'_')
                  return (
                      <Link href={`#${new_item}`}  title={item}
                      />
                  )
              }else if(typeof(item) == 'object'){
                  for (let item_key in item){
                      let new_item_key = item_key.replace(/ /g,'_')
                      return (
                          <Link href={`#${new_item_key}`}  title={item_key}
                          >
                              {this.getNavToRender(item[item_key])}
                          </Link>
                      )
                  }
              }
          })
    }

      render() {
          const { ...props } = this.props;
          const { data } = props;
          const childrenToRender = this.getChildrenToRender(
              data.dataSource,'div',data.key
          );
          let pageNav = null
          if (data.dataNav !== undefined){
              pageNav = this.getNavToRender(data.dataNav)
          }

          let offsetTop = props.offsetTop || 88.154*screen_scale_width;
          // 0: DOC 1:Tutorial
          let type = props.type || 0

          return (
              <div style={style.wrap} className={'content_component_wrap'}>
                  <div style={style.content_wrap[type]} className={'Md_content_component'}>
                      {childrenToRender}
                  </div>
                  {pageNav ? (
                      <Anchor className={'dataNav'}
                              offsetTop={offsetTop+10}
                              style={{marginLeft:10*screen_scale_width,
                                  width:280*screen_scale_width,zIndex:0,
                              }} affix={true}>
                          {pageNav}
                      </Anchor>
                  ):null}
              </div>
          )
      }
}