import React, { Component } from 'react';
import { Layout, Menu, Icon, Tooltip } from 'antd';
import Theme_show from '../Theme_show'
import Help from '../../component/md/Doc/Help'
import MSCOCO2017 from '../../component/md/Doc/MSCOCO2017'
import Optimization from '../../component/md/Doc/Optimization_iteration'
import Md_content_component from '../../component/md/common/Md_content_component'
import {screen_width,model_width, model_height,screen_scale_width,screen_scale_height} from '../../common/parameter/parameters'
import Pet_img from "../../asset/HOME_icon/1x/Pet.png";
import header_img from '../../asset/img/Group 7.png'
import tutorials_sider_data from './tutorials_sider_data'
import Sider_Menu from '../../component/md/common/Sider_Menu'

const {
    Header, Content, Footer, Sider,
} = Layout;

const { SubMenu } = Menu;


const style = {
    slider:{

    },
    slider_ori:{
        overflow: 'auto',
        height: '100vh',
        position: 'fixed',
        left: 0,
    },
    subMenu_item:{
        fontWeight:500,
        fontSize:14,
        // textColor:"#8E8E8E"
        color: '#262626',
        backgroundColor:'#FFFFFF'
    },
    left_item:{
        fontWeight:500,
        fontSize:12,
        width:'auto'
    },
    left_item_1:{
        fontWeight:500,
        fontSize:12,
        position:'relative',
        width:600
    },
    right_item:{
        fontWeight:500,
        fontSize:15,
    },
    header_logo:{
        width:115*screen_scale_width,
        height:40*screen_scale_width,
        background:`url(${Pet_img}) center no-repeat`,
        // backgroundSize:'cover%'
        backgroundSize:'100% 100%',
        marginTop:27*screen_scale_width,
        marginLeft:119*screen_scale_width,
        cursor:'pointer',
    },
    Content:{
        display:'flex',
        flexDirection:'row',
        justifyContent:'space-between',
    },
    Content_main:{
        display:'flex',
        flexDirection:'column',
        justifyContent:'space-between',
        padding: 24, background: '#fff', textAlign: 'center'
    },
    Content_slider:{
        display:'flex',
        flexDirection:'column',
        justifyContent:'space-between',
        height:200,
        backgroundColor:"orange"

    },
    header_menu:{
        lineHeight: '64px',
        display:'flex',
        width:'auto',
        justifyContent:'space-between',
        backgroundColor:'rgba(0,0,0,0)',
        marginLeft:418*screen_scale_width
    },
    header_menu_item:{
        // width:100,
        // paddingLeft:10
    },
    footer_logo:{
        height:53*screen_scale_height,
        width:152*screen_scale_width,
        background:`url(${Pet_img}) center no-repeat`,
        backgroundSize:'cover%',
        marginLeft:150*screen_scale_width,
        marginTop:74*screen_scale_height
    },
};


class Tutorials extends Component {
    // 构造
    constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            content_component : MSCOCO2017,
            Menu:0
        };
        this._click = this._click.bind(this)
        this._click_doc = this._click_doc.bind(this)
        this._onMouseEnter = this._onMouseEnter.bind(this);
        this._onMouseLeave = this._onMouseLeave.bind(this);
    }

    componentDidMount() {
        if (this.props.history.location.state !== undefined){
            this.setState(
                {
                    Menu:this.props.history.location.state.some
                }
            )
        }else {
            console.log('has no key some')
        }

    } 

    _onMouseEnter(eventKey,domEvent) {
        console.log('enter item')
        console.log(`${domEvent.value} hover`)
    }

    _onMouseLeave(eventKey,domEvent){

        console.log('leave item')

    }

    _router(link){
        document.documentElement.scrollTop = document.body.scrollTop =0;
        this.props.history.replace(link, { some: 'state' });
    }

    _jump(address){
        switch (address) {
            case 'github':
                window.location.href = 'https://github.com/soeaver/PytorchEveryThing'
                break;
            case 'pytorch':
                window.location.href = 'https://pytorch.org'
                break;
            case 'bupt':
                break;
            default:


        }
    }

    _click_doc(item, key, keyPath) {
        var {key, keyPath} = item;
        switch (key) {
            case '18':
                this.setState({
                    content_component:Optimization
                });
                break;
            case '32':
                this.setState({
                    content_component:MSCOCO2017
                });
                break;
            default:
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                break;
        }
    }

    _click(item, key, keyPath) {
        var {key, keyPath} = item;

        // var temp = "";
        // console.log(item)
        // for(var i in item){//用javascript的for/in循环遍历对象的属性
        //     temp += i+":"+item[i]+"\n";
        // }
        // alert(temp);
        switch (key) {
            case '1':
                this.props.history.replace('/', { some: 'state' });
                break;
            case '2':
                console.log(`set state ${key}`);
                console.log(`set state ${keyPath}`);
                break;
            case '3':
                console.log(`set state ${key}`);
                console.log(`set state ${keyPath}`);
                break;
            case '4':
                this.props.history.replace('/Doc', { some: 'state' });
                break;
            case '5':
                console.log(`set state ${key}`);
                console.log(`set state ${keyPath}`);
                break;
            case '6':
                this.props.history.replace('/Model_Zoo', { some: 'state' });
                break;
            case '7':
                console.log(`set state ${key}`);
                console.log(`set state ${keyPath}`);
                console.log(this.Menu)
                break;
            default:
                // console.log(`set state ${key}`);
                // console.log(`set state ${keyPath}`);

                break;
        }
    }

    render() {
        let slider_left_width = 250;
        let slider_right_width = 150;
        let header_img_h = 100;
        let Content_component = this.state.content_component


        return (
            <Layout style={{backgroundColor:'#FFFFFF'}}>
                <Layout style={{display:'flex',
                    flexDirection:'column', heighit:'auto',}}>
                    <div style={{display:'flex',
                        flexDirection:'row', justifyContent:'start', heighit:'64',width:'100%',backgroundColor:'none'}}>
                        <div className="logo" style={style.header_logo}
                             onClick={this._router.bind(this,"/")}/>
                        <Menu
                            theme="light"
                            mode="horizontal"
                            defaultSelectedKeys={['5']}
                            style={style.header_menu}
                            onClick={this._click}
                        >
                            <Menu.Item key="1"
                                       style={{backgroundColor:'rgba(0,0,0,0)',fontSize:24*screen_scale_width,fontWeight:1000,}}>Home</Menu.Item>
                            <Menu.Item key="2" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}>About Pet</Menu.Item>
                            <Menu.Item key="3" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}>Install</Menu.Item>
                            <Menu.Item key="4" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}>Doc</Menu.Item>
                            <Menu.Item key="5" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}>Tutorials</Menu.Item>
                            <SubMenu key="6" style={{fontSize:24*screen_scale_width,fontWeight:1000,}} title="Model Zoo">
                                <Menu.Item key="11" style={{fontSize:24*screen_scale_width}}>Model Zoo</Menu.Item>
                                <Menu.Item key="12" style={{fontSize:24*screen_scale_width}}>Classification</Menu.Item>
                                <Menu.Item key="13" style={{fontSize:24*screen_scale_width}}>Detection</Menu.Item>
                                <Menu.Item key="14" style={{fontSize:24*screen_scale_width}}>Segmentation</Menu.Item>
                                <Menu.Item key="15" style={{fontSize:24*screen_scale_width}}>Posture</Menu.Item>
                            </SubMenu>
                            <Menu.Item key="7" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}>Github</Menu.Item>
                        </Menu>
                        <div style={{alignSelf:'center', width:'auto', display:"flex", marginLeft:78*screen_scale_width,}}>
                            <div style={{textAlign:'center',fontSize:20*screen_scale_width,
                                marginRight:24*screen_scale_width,cursor:'pointer',width:44*screen_scale_width,height:28*screen_scale_width,}} >中</div>
                            <div style={{textAlign:'center', fontSize:20*screen_scale_width,color:"#FFFFFF",backgroundColor:'#FD8023',borderRadius:4,width:44*screen_scale_width,height:28*screen_scale_width, cursor:'pointer',
                            }}>EN</div>
                        </div>
                    </div>
                    <img src={require('../../asset/img/Group 7.png')} style={{height:header_img_h}} />
                </Layout>
                <Layout className="Content_layout" style={{ marginTop:67, background:'#FFFFFF',}}>
                    <Sider style={{
                        overflow: 'auto', height: '100vh', position: 'sticky', left: 5, top:0
                    }} width={slider_left_width} theme="light"
                    >
                        <Menu theme="light" mode="inline"
                              onClick={this._click_doc}
                              onSelect={this._select} style={{paddingBottom:20}}

                        >
                            <SubMenu key="0" style={style.subMenu_item}
                                     title={
                                         <span>
                                             <Icon type="mail" />
                                             <span style={style.subMenu_item}>快速开始</span>
                                         </span>
                                     }/>
                            <SubMenu key="sub1" style={{backgroundColor: '#FFFFFF'}}
                                     title={
                                         <span>
                                             <Icon type="mail" />
                                             <span style={style.subMenu_item}>初级教程</span>
                                         </span>
                                     }
                            >
                                <Menu.Item key="10" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                                <Menu.Item key="11" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                                <Menu.Item key="12" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                                <Menu.Item key="13" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                                <Menu.Item key="14" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                            </SubMenu>
                            <SubMenu key="sub2" style={{backgroundColor: '#FFFFFF'}}
                                     title={
                                         <span>
                                             <Icon type="mail" />
                                             <span style={style.subMenu_item}>中级教程</span>
                                         </span>
                                     }
                            >
                                <Menu.Item key="20" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                                <Menu.Item key="21" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                                <Menu.Item key="22" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                                <Menu.Item key="23" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                                <Menu.Item key="24" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                            </SubMenu>
                            <SubMenu key="sub3" style={{backgroundColor: '#FFFFFF'}}
                                     title={
                                         <span>
                                             <Icon type="mail" />
                                             <span style={style.subMenu_item}>高级教程</span>
                                         </span>
                                     }
                            >
                                <Menu.Item key="30" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                                <Menu.Item key="31" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                                <Menu.Item key="32" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                                <Menu.Item key="33" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                                <Menu.Item key="34" style={style.left_item}>
                                    <Tooltip placement="right" title='在MSCOCO2017数据集上训练Faster-RCNN模型'>
                                        <div style={{width:'auto%',overflow: "hidden",textOverflow:"ellipsis"}}>在MSCOCO2017数据集上训练Faster-RCNN模型</div>
                                    </Tooltip>
                                </Menu.Item>
                            </SubMenu>

                        </Menu>
                    </Sider>
                    <Layout style={{backgroundColor:'#FFFFFF'}}>
                        <Content className="Content_layout_Content"
                                 style={{margin: '0 16px 0', display: 'flex', flexDirection: 'column', backgroundColor:'#FFFFFF'}}>
                            <Content_component style={{height: 'auto',backgroundColor:'#F8F8F8'}}/>
                            {/*<Md_content_component />*/}
                        </Content>
                        <Footer className="Content_layout_Footer" style={{backgroundColor:'#3D3E3F',color:'#FFFFFF',textAlign: 'center', height: 310*screen_scale_height, display:"flex",justifyContent:"space-between",}}>
                            <div className="logo" style={style.footer_logo}/>
                            <div style={{display:"flex",justifyContent:"space-between",marginRight:200*screen_scale_width}}>
                                <div style={{display:"flex",justifyContent:"start",flexDirection:"column", textAlign:'left',marginTop:60*screen_scale_height,marginBottom:72*screen_scale_height}}>
                                    <span style={{fontSize:18, color:'#D3D6DD', marginBottom:17*screen_scale_height}}>About Pet</span>
                                    <span style={{fontSize:14, color:'#828282',marginBottom:1*screen_scale_height}}>Function</span>
                                    <span style={{fontSize:14, color:'#828282',marginBottom:1*screen_scale_height}}>Feature</span>
                                    <span style={{fontSize:14, color:'#828282',marginBottom:1*screen_scale_height}}>Contrast</span>
                                    <span style={{fontSize:14, color:'#828282',marginBottom:1*screen_scale_height}}>Expand</span>
                                </div>
                                <div style={{display:"flex",justifyContent:"start",flexDirection:"column", textAlign:'left',marginLeft:135*screen_scale_width,marginTop:60*screen_scale_height,marginBottom:72*screen_scale_height}}>
                                    <span style={{fontSize:18, color:'#D3D6DD',marginBottom:17*screen_scale_height}}>Resources</span>
                                    <span style={{fontSize:14, color:'#828282',marginBottom:1*screen_scale_height}}>Install</span>
                                    <span style={{fontSize:14, color:'#828282',marginBottom:1*screen_scale_height}}>Doc</span>
                                    <span style={{fontSize:14, color:'#828282',marginBottom:1*screen_scale_height}}>Tutorials</span>
                                    <span style={{fontSize:14, color:'#828282',marginBottom:1*screen_scale_height}}>Model Zoo</span>
                                </div>
                                <div style={{display:"flex",justifyContent:"start",flexDirection:"column", textAlign:'left',marginLeft:135*screen_scale_width,marginTop:60*screen_scale_height,marginBottom:72*screen_scale_height}}>
                                    <span style={{fontSize:18, color:'#D3D6DD',marginBottom:17*screen_scale_height}}>Contact us</span>
                                </div>
                            </div>
                        </Footer>
                    </Layout>
                </Layout>
            </Layout>
        )
    }
}

export default Tutorials;


