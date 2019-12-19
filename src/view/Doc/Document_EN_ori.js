import React, { Component } from 'react';
import { Layout, Menu, Icon, Dropdown, Button } from 'antd';

import MSCOCO2017 from '../../component/md/Doc/MSCOCO2017'
import Optimization from '../../component/md/Doc/Optimization_iteration'
import Visualization from '../../component/md/Doc/Visualization'
import Prepare_data from '../../component/md/Doc/Prepare_data'
import Load_data from '../../component/md/Doc/Load_data'
import Log_system from '../../component/md/Doc/Log_system'
import Model_construction from '../../component/md/Doc/Model_construction'
import Config_system from '../../component/md/Doc/Config_system'
import Load_model from '../../component/md/Doc/Load_model'
import test_md_data from '../../component/md/Doc/test_md'
import {screen_width,model_width, model_height,screen_scale_width,screen_scale_height} from '../../common/parameter/parameters'

import './Document.less'

import Pet_img from "../../asset/HOME_icon/1x/Pet.png";
import header_logo from "../../asset/HOME_icon/logo/pet-logo2.png";
import footer_logo from "../../asset/HOME_icon/logo/pet-footer3.png";

import {
    BrowserRouter as Router,
    Route,
    Link,
    Redirect,
    Switch,
} from 'react-router-dom'
import footer_icon1 from "../../asset/HOME_icon/2x/email@2x.png";
import footer_icon2 from "../../asset/HOME_icon/2x/g@2x.png";
import footer_back from "../../asset/HOME_icon/footer.jpg";


const {
    Header, Content, Footer, Sider,
} = Layout;

const { SubMenu } = Menu;


const routes = [
    { path: '/Doc',
        component: Optimization
    },
    { path: '/Doc/Optimization',
        component: Optimization
    },
    { path: '/Doc/Visualization',
        component: Visualization,
        routes:[
            {
                path:'/Doc/Visualization#支持的可视化内容',
                component:Visualization,
            }
        ]
    },
    { path: '/Doc/Prepare_data',
        component: Prepare_data
    },
    {
        path:'/Doc/Log_system',
        component:Log_system
    },
    { path: '/Doc/Load_model',
        component: Load_model,
    },
    { path: '/Doc/Model_construction',
        component: Model_construction,
    },
    { path: '/Doc/Config_system',
        component: Config_system,
    },
];

const RouteWithSubRoutes = (route) => (
    <Route path={route.path} exact render={props => (
        // 把自路由向下传递来达到嵌套。
        <route.component {...props} style={{ width: document.body.clientWidth, height: document.body.clientHeight}} routes={route.routes}/>
    )}/>
);

const style = {
    wrap:{
        display:'flex',
        flexDirection:'column',
        justifyContent:'space-around',
        backgroundColor:'white',
        width:screen_width,
        // overflowX: 'hidden',
    },
    part_wrap:{
        display:"flex",
        justifyContent:"center",
        alignItems:"center",
        backgroundColor:'#FFFFFF',
        width:'100%'
    },
    header_logo:{
        // height:'auto',
        // width:115*screen_scale_width,
        // height:40*screen_scale_width,
        width:60*screen_scale_width,
        height:60*screen_scale_width,
        marginTop:15*screen_scale_width,
        background:`url(${header_logo}) center no-repeat`,
        // backgroundSize:'cover%'
        backgroundSize:'100% 100%',
        // marginTop:27*screen_scale_width,
        marginLeft:119*screen_scale_width,
        cursor:'pointer',
    },
    header_menu:{
        lineHeight: `${88.154*screen_scale_width}px`,
        display:'flex',
        width:(958+170)*screen_scale_width,
        backgroundColor:'rgba(0,0,0,0)',
        marginLeft:(418-170)*screen_scale_width,
    },
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
        fontSize:24*screen_scale_width,
        // textColor:"#8E8E8E"
        color: '#262626',
        backgroundColor:'#FFFFFF'
    },
    left_item:{
        fontWeight:1000,
        fontSize:22*screen_scale_width,
    },
    right_item:{
        fontWeight:500,
        fontSize:15,
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
    header_menu_item:{
        // width:100,
        // paddingLeft:10
    },
    footer_logo:{
        // height:53*screen_scale_width,
        // width:152*screen_scale_width,
        height:60*screen_scale_width,
        width:154*screen_scale_width,
        background:`url(${footer_logo}) center no-repeat`,
        backgroundSize:'100% 100%',
        marginLeft:349*screen_scale_width,
        marginTop:40*screen_scale_width,
        cursor:'pointer',
    },
};


class Document_EN extends Component {
    // 构造
    constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            content_component : Prepare_data,
            version:0.1
        };
        this._click = this._click.bind(this)
        this._click_doc = this._click_doc.bind(this)
        this._verson_selected = this._verson_selected.bind(this)
        this._handleScroll = this._handleScroll.bind(this)
        this._header_item_onMouseEnter = this._header_item_onMouseEnter.bind(this)
        this._header_item_onMouseLeave = this._header_item_onMouseLeave.bind(this)

        this.Versions_menu = (
            <Menu onClick={this._verson_selected}>
                <Menu.Item key={0}>
                    <span target="_blank" rel="noopener noreferrer" >
                        0.1
                    </span>
                </Menu.Item>
            </Menu>
        );
    }


    _header_item_onMouseEnter(eventKey){
        let dom = eventKey.domEvent.target
        dom.style.color = 'rgba(251,139,35,1)'
    }

    _header_item_onMouseLeave(eventKey){
        console.log(`mouse leave`)
        let scrollTop = document.body.scrollTop || document.documentElement.scrollTop
        let font_color = scrollTop/(820*screen_scale_width)*255
        font_color = font_color > 255 ? 255 : font_color

        let dom = eventKey.domEvent.target
        dom.style.color = `rgba(${font_color},${font_color},${font_color},1)`
    }

    _handleScroll(e){
        let header_wrap = this.header_wrap
        let header = this.header
        let other_btn = this.other_btn
        let scrollTop = document.body.scrollTop || document.documentElement.scrollTop

        let opaque = scrollTop/(820*screen_scale_width)
        let font_color = scrollTop/(820*screen_scale_width)*255
        font_color = font_color > 255 ? 255 : font_color


        if (scrollTop < 820*screen_scale_width){
            header_wrap.style.background = `rgba(61,62,73,${opaque})`
            //这里写死了, 根据不通页面, i 的起始值不同
            other_btn.style.color = `rgba(${font_color},${font_color},${font_color},1)`
            for (let i = 1; i<17 ;i = i+2) {
                if ((i+1)/2 ===4){
                    continue;
                }
                if (header.children[1].children[i].className.indexOf('active')!==-1){
                    continue
                }
                header.children[1].children[i].style.color = `rgba(${font_color},${font_color},${font_color},1)`
            }
        }

    }


    componentDidMount(){
        console.log('home scroll listener add')
        window.addEventListener('scroll',this._handleScroll)

        let header_wrap = this.header_wrap
        let header = this.header
        let other_btn = this.other_btn
        let scrollTop = document.body.scrollTop || document.documentElement.scrollTop

        let opaque = scrollTop/(820*screen_scale_width)
        let font_color = scrollTop/(820*screen_scale_width)*255
        font_color = font_color > 255 ? 255 : font_color

        header_wrap.style.background = `rgba(61,62,73,${opaque})`
        other_btn.style.color = `rgba(${font_color},${font_color},${font_color},1)`
        for (let i = 1; i<17 ;i = i+2) {
            if ((i+1)/2 ===4){
                continue;
            }
            header.children[1].children[i].style.color = `rgba(${font_color},${font_color},${font_color},1)`
        }
    }

    componentWillUnmount() {
        window.removeEventListener('scroll',this._handleScroll)
        console.log('home scroll listener remove')
        if (this.props.history.location.state !== undefined){
            console.log(`doc ${this.props.history.location.state.some}`);
            let location_state = this.props.history.location.state.some
        }
    }


    componentWillReceiveProps(nextProps, nextContext) {

    }


    _router(link){
        document.documentElement.scrollTop = document.body.scrollTop =0;
        // this.props.history.push(link, { some: 'state' });
        this.props.history.push(link, { some: 'state' });
    }

    _jump(address){
        switch (address) {
            case 'github':
                window.location.href = 'https://github.com/BUPT-PRIV/Pet-dev'
                break;
            case 'pytorch':
                window.location.href = 'https://pytorch.org'
                break;
            case 'bupt':
                break;
            default:


        }
    }

    _verson_selected(item, key, keyPath) {
        var {key, keyPath} = item;
        let version = item.item.props.children.props.children
        this.setState({
            version:version
        })
    }


    _click_doc(item, key, keyPath) {
        var {key, keyPath} = item;
        switch (key) {
            case 'Doc-09':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                // this.setState({
                //     content_component:test_md_data
                // });
                break;
            case 'Doc-10':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                // this.setState({
                //     content_component:Optimization
                // });
                break;
            case 'Doc-11':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                // this.setState({
                //     content_component:Visualization
                // });
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
                // this.setState({
                //     content_component:Load_data
                // });
                break;
            case 'Doc-14':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                // this.setState({
                //     content_component:Log_system
                // });
                break;
            case 'Doc-15':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                // this.setState({
                //     content_component:Load_model
                // });
                break;
            case 'Doc-16':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                // this.setState({
                //     content_component:Model_construction
                // });
                break;
            case 'Doc-17':
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                document.documentElement.scrollTop = document.body.scrollTop =0;
                // this.setState({
                //     content_component:Config_system
                // });
                break;
            case '32':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                // this.setState({
                //     content_component:MSCOCO2017
                // });
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
                document.documentElement.scrollTop = document.body.scrollTop =0;
                // this.props.history.push('/', { some: 'state' });
                this.props.history.push('/', { some: 'state' });
                break;
            case '2':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.props.history.push('/About_Pet', { some: 'state' });
                break;
            case '3':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.props.history.push('/Install', { some: 'state' });
                break;
            case '4':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.props.history.push('/Doc', { some: 'state' });
                break;
            case '5':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.props.history.push('/Tutorials', { some: 'state' });
                break;
            case '6':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.props.history.push('/Model_Zoo', { some: 'state' });
                break;
            case '7':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.props.history.push('/Contact_us', { some: 'state' });
                break;
            case '8':
                this._jump("github")
                console.log(`set state ${key}`);
                console.log(`set state ${keyPath}`);
                break;
            default:
                this.props.history.push('/Model_Zoo', { some: 'state' });
                break;
        }
    }

    render() {
        let change_lang = this.props.onClick

        let slider_left_width = 250*screen_scale_width;
        let slider_right_width = 150;
        let header_img_h = 120*screen_scale_width;
        let Content_component = this.state.content_component

        let head_height = 88.154*screen_scale_width;
        console.log(`render version ${this.state.version}`)

        return (
            <Layout style={style.wrap} className='Doc Doc_EN'>
                <div style={{display:'flex',
                    flexDirection:'row', justifyContent:'center', height:head_height,width:'100%',backgroundColor:"rgba(0,0,0,1)",position: 'fixed', top:0,zIndex:1,}}
                     ref={(input) => { this.header_wrap = input; }}>
                    <div style={{display:'flex',
                        flexDirection:'row', justifyContent:'start', height:head_height,width:model_width,backgroundColor:"none",position:"relative"}}
                         ref={(input) => { this.header = input; }}>
                        <div className="logo" style={style.header_logo}
                             onClick={this._router.bind(this,"/",0)}/>
                        <Menu
                            theme="light"
                            mode="horizontal"
                            defaultSelectedKeys={['4']}
                            style={style.header_menu}
                            onClick={this._click}
                            className={'Doc-header'}
                        >
                            <Menu.Item key="1" style={{backgroundColor:'rgba(0,0,0,0)',fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>Home</Menu.Item>
                            <Menu.Item key="2" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>About Pet</Menu.Item>
                            <Menu.Item key="3" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>Install</Menu.Item>
                            <Menu.Item key="4" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}>Document</Menu.Item>
                            <Menu.Item key="5" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>Tutorials</Menu.Item>
                            <Menu.Item key="6" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>Model&nbsp;&nbsp;&nbsp;&nbsp;Zoo</Menu.Item>
                            <Menu.Item key="7" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>Contact Us</Menu.Item>
                            <Menu.Item key="8" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>Github</Menu.Item>
                        </Menu>
                        <div style={{alignSelf:'center', width:'auto', display:"flex",position:"absolute",right:50*screen_scale_width}}>
                            <div style={{textAlign:'center',fontSize:20*screen_scale_width,
                                marginRight:24*screen_scale_width,cursor:'pointer',width:44*screen_scale_width,height:28*screen_scale_width,opacity:1,
                            }} onClick={change_lang} ref={(input) => { this.other_btn = input; }}>中</div>
                            <div style={{textAlign:'center', fontSize:20*screen_scale_width,color:"#FFFFFF",backgroundColor:'#FD8023',borderRadius:20,width:44*screen_scale_width,height:28*screen_scale_width, cursor:'pointer',
                            }}
                            >EN</div>
                        </div>
                    </div>
                </div>
                <img src={require('../../asset/HOME_icon/banner/doc.jpg')} style={{height:header_img_h,marginTop:head_height,
                    }} />
                <Layout className="Content_layout" style={{ marginTop:40*screen_scale_width, background:'#FFFFFF',}}>
                    <Sider style={{
                        overflow: "visible", height: '100vh', position: 'sticky', left: 5, top:head_height+5*screen_scale_width
                    }} width={slider_left_width} theme="light"
                           className={'sider_nav'}
                    >
                        <Dropdown overlay={this.Versions_menu} placement="bottomCenter">
                            <Button style={{width:'100%',marginBottom:20*screen_scale_width,display:'flex',justifyContent:'space-between'}}>{this.state.version}<Icon type="down" />
                            </Button>
                        </Dropdown>
                        <Menu theme="light" mode="inline" defaultOpenKeys={['sub1']} defaultSelectedKeys={['Doc-12']} onClick={this._click_doc}
                              onSelect={this._select} style={{paddingBottom:20}}>
                            <Menu.Item key="Doc-09" style={style.left_item}>Overview</Menu.Item>
                            <Menu.Item key="Doc-10" style={style.left_item}>Iterative optimization</Menu.Item>
                            <Menu.Item key="Doc-11" style={style.left_item}>Visualization</Menu.Item>
                            <Menu.Item key="Doc-12" style={style.left_item}>Prepare data</Menu.Item>
                            <Menu.Item key="Doc-13" style={style.left_item}>Data loading</Menu.Item>
                            <Menu.Item key="Doc-14" style={style.left_item}>Log system</Menu.Item>
                            <Menu.Item key="Doc-15" style={style.left_item}>Model loading and saving</Menu.Item>
                            <Menu.Item key="Doc-16" style={style.left_item}>Model building</Menu.Item>
                            <Menu.Item key="Doc-17" style={style.left_item}>Config system</Menu.Item>
                        </Menu>
                    </Sider>
                    <Layout style={{backgroundColor:'#FFFFFF'}}>
                        <Content className="Content_layout_Content"
                                 style={{display: 'flex', flexDirection: 'column', backgroundColor:'#FFFFFF'}}>
                            <Content_component style={{height: 'auto',backgroundColor:'#F8F8F8'}} en={0} offsetTop={head_height}/>
                            {/*<div style={{height: 'auto',backgroundColor:'#F8F8F8'}}>*/}
                            {/*    {routes.map((route, i) => (*/}
                            {/*        <RouteWithSubRoutes key={i} {...route} value='asdasd'/>*/}
                            {/*    ))}*/}
                            {/*</div>*/}
                        </Content>
                    </Layout>
                </Layout>
                <div style={{...style.part_wrap, background:`url(${footer_back}) no-repeat `,backgroundSize: '100% 100%',zIndex:1}}>
                    <div className="Content_layout_Footer" style={{color:'#FFFFFF',textAlign: 'center', height: 350*screen_scale_width, display:"flex",flexDirection:"column",justifyContent:"start",width:model_width}}>
                        <div style={{textAlign: 'center', height: 270*screen_scale_width, display:"flex",justifyContent:"start",
                            borderBottomStyle:"solid",
                            borderBottomWidth:1,
                            borderBottomColor:'#6A6F73'}}>
                            <div className="logo" style={style.footer_logo} onClick={this._router.bind(this,"/",0)}/>
                            <div style={{display:"flex"}}>
                                <div style={{display:"flex",justifyContent:"start",flexDirection:"column", textAlign:'left',marginTop:40*screen_scale_width,width:100*screen_scale_width,
                                    marginLeft:235*screen_scale_width
                                }}>
                                    <span style={{fontSize:22*screen_scale_width, color:'#D3D6DD', marginBottom:20*screen_scale_width}}>Pet</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/",0)}>Get Started</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/",0)}>About Pet</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/",0)}>Resource</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._jump.bind(this,"github")}>Github</span>
                                </div>
                                <div style={{display:"flex",justifyContent:"start",flexDirection:"column", textAlign:'left',width:112*screen_scale_width, marginLeft:230*screen_scale_width,marginTop:40*screen_scale_width}}>
                                    <span style={{fontSize:22*screen_scale_width, color:'#D3D6DD',marginBottom:20*screen_scale_width}}>Resources</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/Doc",0)}>Doc</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/Tutorials",0)}>Tutorials</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/Model_Zoo",0)}>ModelZoo</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._jump.bind(this,"github",0)}>Github issues</span>
                                </div>
                                <div style={{display:"flex",justifyContent:"start",flexDirection:"column", textAlign:'left',marginLeft:280*screen_scale_width,marginTop:40*screen_scale_width,}}>
                                    <span style={{cursor:'pointer',fontSize:22*screen_scale_width, color:'#D3D6DD',marginBottom:22*screen_scale_width}}>Contact us</span>
                                    <div>
                                        <img key={'footer_icon1'} src={footer_icon1} style={{cursor:'pointer',width:44*screen_scale_width,height:44*screen_scale_width}}
                                             onClick={this._router.bind(this,"/",0)}/>
                                        <img key={'footer_icon2'} src={footer_icon2} style={{cursor:'pointer',width:44*screen_scale_width,height:44*screen_scale_width,marginLeft:31*screen_scale_width}}
                                             onClick={this._jump.bind(this,"github")}/>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <span style={{fontSize: 16*screen_scale_width,
                            color: "#B0B0B0",
                            letterSpacing: 0,
                            marginTop:22*screen_scale_width}}>Copyright © Pet | 京ICP备17031240号-1 | Song技术支持</span>
                    </div>
                </div>
            </Layout>

        )
    }
}

export default Document_EN;


