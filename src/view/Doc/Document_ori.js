import React, { Component } from 'react';
import { Layout, Menu, Icon } from 'antd/lib/index';
import Theme_show from '../Theme_show'
import Help from '../../component/md/Doc/Help'
import MSCOCO2017 from '../../component/md/Doc/MSCOCO2017'
import Optimization from '../../component/md/Doc/Optimization_iteration'
import Visualization from '../../component/md/Doc/Visualization'
import Prepare_data from '../../component/md/Doc/Prepare_data'
import Log_system from '../../component/md/Doc/Log_system'
import Model_construction from '../../component/md/Doc/Model_construction'
import Config_system from '../../component/md/Doc/Config_system'
import Load_model from '../../component/md/Doc/Load_model'

import './Document.less'

import Pet_img from "../../asset/HOME_icon/1x/Pet.png";
import header_img from '../../asset/img/Group 7.png'
import Home from "../Home/Home";
import Model_Zoo from "../Model/Model_Zoo";
import Tutorials from "../Tutorials/Tutorials";
import Test_md from "../../component/md/Doc/test_md";
import Demo from "../../component/test/test_banner_anim";
import HomeDemo from "../../Home";

import {
    BrowserRouter as Router,
    Route,
    Link,
    Redirect,
    Switch,
} from 'react-router-dom'


const {
    Header, Content, Footer, Sider,
} = Layout;

const { SubMenu } = Menu;

//  1397
let screen_width = document.documentElement.clientWidth;
// 798
let screen_height = document.documentElement.clientHeight;
let screen_scale_width = screen_width/1920;
let screen_scale_height = screen_height/1080;

const routes = [
    { path: '/Doc/',
        component: Home
    },
    { path: '/Doc/',
        component: Model_Zoo,
    },
    { path: '/Doc/',
        component: Document
    },
    {
        path:'/Doc/',
        component:Tutorials
    },
    { path: '/Doc/',
        component: Test_md,
    },
    { path: '/Doc/',
        component: Demo,
    },
    { path: '/Doc/',
        component: HomeDemo,
    },
];

const RouteWithSubRoutes = (route) => (
    <Route path={route.path} exact render={props => (
        // 把自路由向下传递来达到嵌套。
        <route.component {...props} style={{ width: document.body.clientWidth, height: document.body.clientHeight}} routes={route.routes}/>
    )}/>
);

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
    },
    right_item:{
        fontWeight:500,
        fontSize:15,
    },
    header_logo:{
        height:'auto',
        width:240,
        background:`url(${Pet_img}) center no-repeat`,
        backgroundSize:'50%'
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


class Document extends Component {
    // 构造
    constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            content_component : Optimization
        };
        this._click = this._click.bind(this)
        this._click_doc = this._click_doc.bind(this)
    }


    _click_doc(item, key, keyPath) {
        var {key, keyPath} = item;
        switch (key) {
            case 'Doc-10':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.setState({
                    content_component:Optimization
                });
                break;
            case 'Doc-11':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.setState({
                    content_component:Visualization
                });
                break;
            case 'Doc-12':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.setState({
                    content_component:Prepare_data
                });
                break;
            case 'Doc-13':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.setState({
                    content_component:Prepare_data
                });
                break;
            case 'Doc-14':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.setState({
                    content_component:Log_system
                });
                break;
            case 'Doc-15':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.setState({
                    content_component:Load_model
                });
                break;
            case 'Doc-16':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.setState({
                    content_component:Model_construction
                });
                break;
            case 'Doc-17':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.setState({
                    content_component:Config_system
                });
                break;
            case '32':
                document.documentElement.scrollTop = document.body.scrollTop =0;
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
                this.props.history.push('/', { some: 'state' });
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
                this.props.history.push('/Doc', { some: 'state' });
                break;
            case '5':
                this.props.history.push('/Tutorials', { some: 'state' });
                break;
            case '6':
                this.props.history.push('/Model_Zoo', { some: 'state' });
                break;
            case '7':
                console.log(`set state ${key}`);
                console.log(`set state ${keyPath}`);
                break;
            default:
                this.props.history.push('/Model_Zoo', { some: 'state' });
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
                        flexDirection:'row', justifyContent:'space-between', heighit:'64'}}>
                        <div className="logo" style={style.header_logo}/>
                        <Header className="Content_layout_Header" style={{background: '#FFFFFF', height: 'auto',flex:1, textAlign:'center'}}>
                            <Menu
                                theme="light"
                                mode="horizontal"
                                defaultSelectedKeys={['4']}
                                style={style.header_menu}
                                onClick={this._click}
                            >
                                <Menu.Item key="1" style={style.header_menu_item}>Home</Menu.Item>
                                <Menu.Item key="2" style={style.header_menu_item}>About Pet</Menu.Item>
                                <Menu.Item key="3" style={style.header_menu_item}>Install</Menu.Item>
                                <Menu.Item key="4" style={style.header_menu_item}>Doc</Menu.Item>
                                <Menu.Item key="5" style={style.header_menu_item}>Tutorials</Menu.Item>
                                <SubMenu key="6" title="Model Zoo">
                                    <Menu.Item key="11" >Model Zoo</Menu.Item>
                                    <Menu.Item key="12" >Classification</Menu.Item>
                                    <Menu.Item key="13" >Detection</Menu.Item>
                                    <Menu.Item key="14" >Segmentation</Menu.Item>
                                    <Menu.Item key="15" >Posture</Menu.Item>
                                </SubMenu>
                                <Menu.Item key="7" style={style.header_menu_item}>contact us</Menu.Item>
                            </Menu>
                        </Header>
                    </div>
                    <img src={require('../../asset/img/Group 7.png')} style={{height:header_img_h}} />
                </Layout>
                <Layout className="Content_layout" style={{ marginTop:67, background:'#FFFFFF',}}>
                    <Sider style={{
                        overflow: 'auto', height: '100vh', position: 'sticky', left: 5, top:0
                    }} width={slider_left_width} theme="light"
                    >
                        <Menu theme="light" mode="inline" defaultOpenKeys={['sub1']} defaultSelectedKeys={['Doc-10']} onClick={this._click_doc}
                              onSelect={this._select} style={{paddingBottom:20}}>
                            <SubMenu key="sub1" style={{backgroundColor: '#FFFFFF'}}
                                     title={<span style={style.subMenu_item}>组件</span>}
                            >
                                <Menu.Item key="Doc-10" style={style.left_item}>优化迭代</Menu.Item>
                                <Menu.Item key="Doc-11" style={style.left_item}>可视化</Menu.Item>
                                <Menu.Item key="Doc-12" style={style.left_item}>数据制备</Menu.Item>
                                <Menu.Item key="Doc-13" style={style.left_item}>数据加载</Menu.Item>
                                <Menu.Item key="Doc-14" style={style.left_item}>日志系统</Menu.Item>
                                <Menu.Item key="Doc-15" style={style.left_item}>模型加载和保存</Menu.Item>
                                <Menu.Item key="Doc-16" style={style.left_item}>模型构建</Menu.Item>
                                <Menu.Item key="Doc-17" style={style.left_item}>配置系统</Menu.Item>
                            </SubMenu>
                            <SubMenu key="sub2" style={{backgroundColor: '#FFFFFF'}}
                                     title={<span style={style.subMenu_item}>高效计算</span>}
                            >
                                <Menu.Item key="Doc-20" style={style.left_item}>在MSCOCO2017数据集上训练Faster-RCNN模型</Menu.Item>
                                <Menu.Item key="Doc-21" style={style.left_item}>在MSCOCO2017数据集上训练Faster-RCNN模型</Menu.Item>
                                <Menu.Item key="Doc-22" style={style.left_item}>在MSCOCO2017数据集上训练Faster-RCNN模型</Menu.Item>
                                <Menu.Item key="Doc-23" style={style.left_item}>在MSCOCO2017数据集上训练Faster-RCNN模型</Menu.Item>
                                <Menu.Item key="Doc-24" style={style.left_item}>在MSCOCO2017数据集上训练Faster-RCNN模型</Menu.Item>
                            </SubMenu>
                            <SubMenu key="sub3" style={{backgroundColor: '#FFFFFF'}}
                                     title={<span style={style.subMenu_item}>编程风格</span>}
                            >
                                <Menu.Item key="Doc-30" style={style.left_item}>在MSCOCO2017数据集上训练Faster-RCNN模型</Menu.Item>
                                <Menu.Item key="Doc-31" style={style.left_item}>在MSCOCO2017数据集上训练Faster-RCNN模型</Menu.Item>
                                <Menu.Item key="Doc-32" style={style.left_item}>在MSCOCO2017数据集上训练Faster-RCNN模型</Menu.Item>
                                <Menu.Item key="Doc-33" style={style.left_item}>在MSCOCO2017数据集上训练Faster-RCNN模型</Menu.Item>
                                <Menu.Item key="Doc-34" style={style.left_item}>在MSCOCO2017数据集上训练Faster-RCNN模型</Menu.Item>
                            </SubMenu>
                        </Menu>
                    </Sider>
                    <Layout style={{backgroundColor:'#FFFFFF'}}>
                        <Content className="Content_layout_Content"
                                 style={{margin: '0 16px 0', display: 'flex', flexDirection: 'column', backgroundColor:'#FFFFFF'}}>
                            <Content_component style={{height: 'auto',backgroundColor:'#F8F8F8'}}/>
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

export default Document;


