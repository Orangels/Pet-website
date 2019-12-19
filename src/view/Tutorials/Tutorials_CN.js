import React, { Component } from 'react';
import { Layout, Menu, Icon, Tooltip } from 'antd';

import Quick_start from '../../component/md/Tutorial/Quick_start'
import ImageNet_train from '../../component/md/Tutorial/ImageNet_train'
import MSCOCO_Simple_Baselines from '../../component/md/Tutorial/MSCOCO_Simple_Baselines'
import MSCOCO_SSD from '../../component/md/Tutorial/MSCOCO_SSD'
import MSCOCO_Mask_RCNN from '../../component/md/Tutorial/MSCOCO_Mask_RCNN'
import CIHP_Parsing_R_CNN from '../../component/md/Tutorial/CIHP_Parsing_R_CNN'
import Make_DataSet from '../../component/md/Tutorial/Make_DataSet'
import MSCOCO_RetinaNet from '../../component/md/Tutorial/MSCOCO_RetinaNet'

import {screen_width,model_width, model_height,screen_scale_width,screen_scale_height} from '../../common/parameter/parameters'
import Pet_img from "../../asset/HOME_icon/1x/Pet.png";
import header_logo from "../../asset/HOME_icon/logo/pet-logo2.png";
import footer_logo from "../../asset/HOME_icon/logo/pet-footer3.png";
import header_img from '../../asset/img/Group 7.png'
import { tutorials_sider_data_CN as tutorials_sider_data } from './tutorials_sider_data'
import Sider_Menu from '../../component/md/common/Sider_Menu'
import footer_icon1 from "../../asset/HOME_icon/2x/email@2x.png";
import footer_icon2 from "../../asset/HOME_icon/2x/g@2x.png";

import './Tutorials.less'
import footer_back from "../../asset/HOME_icon/footer.jpg";
import Prepare_data from "../../component/md/Doc/Prepare_data";
import workplace from "../../asset/HOME_icon/workplace/workplace.jpg";

const {
    Header, Content, Footer, Sider,
} = Layout;

const { SubMenu } = Menu;


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
        background:`url(${header_logo}) center no-repeat`,
        // backgroundSize:'cover%'
        backgroundSize:'100% 100%',
        marginTop:15*screen_scale_width,
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


const Placeholder_Component = ()=>(
    <img src={workplace} style={{width:800*screen_scale_width, height:600*screen_scale_width, marginTop:50*screen_scale_width,
    alignSelf:'center'}}/>
)


class Tutorials_CN extends Component {
    // 构造
    constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            content_component : Quick_start,
            selectedKeys:['tutorials_sider_0'],
            openKeys:['tutorials_sider_1'],

        };
        this._click = this._click.bind(this)
        this._click_doc = this._click_doc.bind(this)
        this._onMouseEnter = this._onMouseEnter.bind(this);
        this._onMouseLeave = this._onMouseLeave.bind(this);
        this._openChange = this._openChange.bind(this)
        this._handleScroll = this._handleScroll.bind(this)
        this._header_item_onMouseEnter = this._header_item_onMouseEnter.bind(this)
        this._header_item_onMouseLeave = this._header_item_onMouseLeave.bind(this)
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
                if ((i+1)/2 ===5){
                    continue;
                }
                if (header.children[1].children[i].className.indexOf('active')!==-1){
                    continue
                }

                header.children[1].children[i].style.color = `rgba(${font_color},${font_color},${font_color},1)`
            }
        }

    }


    renderContentComponent(key){

        switch (key) {
            case 'tutorials_sider_0':
                this.setState({
                    content_component:Quick_start
                });
                break;
            // 初级
            case 'tutorials_sider_20_0':
                this.setState({
                    content_component:Make_DataSet
                });
                break;
            case 'tutorials_sider_20_1':
                this.setState({
                    content_component:ImageNet_train
                });
                break;
            case 'tutorials_sider_20_2':
                this.setState({
                    content_component:MSCOCO_Simple_Baselines
                });
                break;
            case 'tutorials_sider_20_3':
                this.setState({
                    content_component:MSCOCO_SSD
                });
                break;
            case 'tutorials_sider_20_4':
                this.setState({
                    content_component:MSCOCO_Mask_RCNN
                });
                break;
            case 'tutorials_sider_20_5':
                this.setState({
                    content_component:CIHP_Parsing_R_CNN
                });
                break;
            case 'tutorials_sider_20_6':
                this.setState({
                    content_component:MSCOCO_RetinaNet
                    // content_component:Placeholder_Component
                });
                break;
            // 中级
            case 'tutorials_sider_30_0':
                this.setState({
                    content_component:Placeholder_Component
                });
                break;
            // 高级
            case 'tutorials_sider_40_0':
                this.setState({
                    content_component:Placeholder_Component
                });
                break;
            default:
                this.setState({
                    content_component:Placeholder_Component
                });
                break;
        }
    }

    componentDidMount(){

        if (this.props.history.location.state !== undefined){
            if (this.props.history.location.state.some.openKeys !== undefined){
                if (this.props.history.location.state.some.openKeys!==0){
                    this.setState(
                        {
                            openKeys:[`${tutorials_sider_data.key}_${this.props.history.location.state.some.openKeys}`],
                            selectedKeys:[`${tutorials_sider_data.key}_${this.props.history.location.state.some.selectedKeys}_0`]
                        },()=>{
                            let key = this.state.selectedKeys[0]
                            this.renderContentComponent(key)
                        }
                    )
                }else {
                    this.setState(
                        {
                            selectedKeys:[`${tutorials_sider_data.key}_${this.props.history.location.state.some.openKeys}`],
                        },()=>{
                            let key = this.state.selectedKeys[0]
                            this.renderContentComponent(key)
                        }
                    )
                }

            }else {
                this.setState(
                    {
                        selectedKeys:[`${tutorials_sider_data.key}_${this.props.history.location.state.some.selectedKeys}`],
                    },()=>{
                        let key = this.state.selectedKeys[0]
                        this.renderContentComponent(key)
                    }
                )
            }

        }else {
            console.log('has no key some')
        }

        console.log(`select key init ${this.state.selectedKeys}`)

        console.log('home scroll listener add')
        window.addEventListener('scroll',this._handleScroll)

        let header_wrap = this.header_wrap
        let header = this.header
        let other_btn = this.other_btn
        let scrollTop = document.body.scrollTop || document.documentElement.scrollTop

        let opaque = scrollTop/(820*screen_scale_width)
        let font_color = scrollTop/(820*screen_scale_width)*255
        font_color = font_color > 255 ? 255 : font_color

        // header_wrap.style.background = `rgba(0,0,0,${opaque})`
        header_wrap.style.background = `rgba(61,62,73,${opaque})`
        other_btn.style.color = `rgba(${font_color},${font_color},${font_color},1)`
        for (let i = 1; i<17 ;i = i+2) {
            if ((i+1)/2 ===5){
                continue;
            }
            // console.log(header.children[1].children[i].className)
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


    _onMouseEnter(eventKey,domEvent) {
        console.log('enter item')
        console.log(`${domEvent.value} hover`)
    }

    _onMouseLeave(eventKey,domEvent){

        console.log('leave item')

    }

    _router(link){
        document.documentElement.scrollTop = document.body.scrollTop =0;
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
                window.location.href = 'https://www.bupt.edu.cn/'
                break;
            default:


        }
    }

    _openChange(openKeys){
        console.log(openKeys)
        this.setState(
            {
                openKeys:openKeys
            }
        )
    }

    _click_doc(item, key, keyPath) {
        var {key, keyPath} = item;
        console.log(`tuturials key -- ${key}`)
        this.renderContentComponent(key)
        this.setState(
            {
                selectedKeys:[`${key}`]
            }
        )
        document.documentElement.scrollTop = document.body.scrollTop =0;
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
                break;
            case '4':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.props.history.push('/Doc', { some: 'state' });
                break;
            case '5':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                console.log(`set state ${key}`);
                console.log(`set state ${keyPath}`);
                break;
            case '6':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.props.history.push('/Model_Zoo', { some: {
                        selectedKeys:(0+1)*10
                    } });
                break;
            case '7':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.props.history.push('/Contact_us', { some: 'state' });
                break;
            case '8':
                this._jump("github")
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
        let slider_left_width = 310*screen_scale_width;
        let slider_right_width = 150;
        let header_img_h = 120*screen_scale_width;
        let Content_component = this.state.content_component

        
        console.log(`open keys ${this.state.openKeys}`)
        console.log(`select keys ${this.state.selectedKeys}`)
        let change_lang = this.props.onClick
        let Sider_Menu_component = <Sider_Menu data={tutorials_sider_data} onClick={this._click_doc}
                                               openKeys={this.state.openKeys}
                                               selectedKeys={this.state.selectedKeys}
                                               onOpenChange={this._openChange}
                                               en={1}
        />

        let head_height = 88.154*screen_scale_width;

        return (
            <Layout style={{...style.wrap, backgroundColor:'#F7F7F7'}} className='Tutorials Tutorials_CN'>
                <div style={{display:'flex',
                    flexDirection:'row', justifyContent:'center', height:head_height,width:'100%',position: 'fixed', top:0,zIndex:1,
                    backgroundColor:"rgba(0,0,0,1)",
                }}
                     ref={(input) => { this.header_wrap = input; }}>
                    <div style={{display:'flex',
                        flexDirection:'row', justifyContent:'start', height:head_height,width:model_width,backgroundColor:"none",position:"relative"}}
                         ref={(input) => { this.header = input; }}>
                        <div className="logo" style={style.header_logo}
                             onClick={this._router.bind(this,"/",0)}/>
                        <Menu
                            theme="light"
                            mode="horizontal"
                            defaultSelectedKeys={['5']}
                            style={style.header_menu}
                            onClick={this._click}
                            className={'Tutorials-header'}
                        >
                            <Menu.Item key="1" style={{backgroundColor:'rgba(0,0,0,0)',fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>首页</Menu.Item>
                            <Menu.Item key="2" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>关于 Pet</Menu.Item>
                            <Menu.Item key="3" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>安装</Menu.Item>
                            <Menu.Item key="4" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>文档</Menu.Item>
                            <Menu.Item key="5" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}>教程</Menu.Item>
                            <Menu.Item key="6" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>模型库</Menu.Item>
                            <Menu.Item key="7" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>联系我们</Menu.Item>
                            <Menu.Item key="8" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>Github</Menu.Item>
                        </Menu>
                        <div style={{alignSelf:'center', width:'auto', display:"flex",position:"absolute",right:50*screen_scale_width}}>
                            <div style={{textAlign:'center', fontSize:20*screen_scale_width,color:"#FFFFFF",backgroundColor:'#FD8023',borderRadius:20,width:44*screen_scale_width,height:28*screen_scale_width, cursor:'pointer',marginRight:24*screen_scale_width
                            }}>中</div>
                            <div style={{textAlign:'center',fontSize:20*screen_scale_width,width:44*screen_scale_width,height:28*screen_scale_width,
                                cursor:'pointer',}} onClick={change_lang} ref={(input) => { this.other_btn = input; }}>EN</div>

                        </div>
                    </div>
                </div>
                <img src={require('../../asset/HOME_icon/banner/tutorials.jpg')} style={{height:header_img_h,marginTop:head_height,
                    backgroundPositionX:0, backgroundPositionY:120*screen_scale_width}} />
                <Layout className="Content_layout" style={{ backgroundColor:'#F7F7F7'}}>
                    <Sider className={'sider sider_nav left-sider'} style={{
                        overflow:"auto", height: '80vh', position: "sticky", left: 5, top:head_height-10,
                        marginTop:40*screen_scale_width,
                    }} width={slider_left_width} theme="light">
                        {Sider_Menu_component}
                    </Sider>
                    <Layout style={{backgroundColor:'#FFFFFF', marginLeft:5*screen_scale_width, paddingTop:40*screen_scale_width}}>
                        <Content className="Content_layout_Content"
                                 style={{display: 'flex', flexDirection: 'column', backgroundColor:'#FFFFFF'}}>
                            <Content_component style={{height: 'auto',backgroundColor:'#F8F8F8'}} en={1} type={1}/>
                            {/*<Optimization style={{height: 'auto',backgroundColor:'#F8F8F8'}} en={1}/>*/}
                        </Content>
                    </Layout>
                </Layout>
                <div style={{...style.part_wrap, background:`url(${footer_back}) no-repeat `,backgroundSize: '100% 100%',}}>
                    <div className="Content_layout_Footer" style={{color:'#FFFFFF',textAlign: 'center', height: 340*screen_scale_width, display:"flex",flexDirection:"column",justifyContent:"start",width:model_width}}>
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
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/",0)}>快速开始</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/",0)}>关于 Pet</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/",0)}>资源</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._jump.bind(this,"github")}>Github</span>
                                </div>
                                <div style={{display:"flex",justifyContent:"start",flexDirection:"column", textAlign:'left',width:112*screen_scale_width, marginLeft:230*screen_scale_width,marginTop:40*screen_scale_width}}>
                                    <span style={{fontSize:22*screen_scale_width, color:'#D3D6DD',marginBottom:20*screen_scale_width}}>资源</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/Doc",0)}>文档</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/Tutorials",0)}>教程</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/Model_Zoo",0)}>模型库</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._jump.bind(this,"github",0)}>Github 问题</span>
                                </div>
                                <div style={{display:"flex",justifyContent:"start",flexDirection:"column", textAlign:'left',marginLeft:280*screen_scale_width,marginTop:40*screen_scale_width,}}>
                                    <span style={{fontSize:22*screen_scale_width, color:'#D3D6DD',marginBottom:22*screen_scale_width}}>联系我们</span>
                                    <div>
                                        <img key={'footer_icon1'} src={footer_icon1} style={{cursor:'pointer',width:44*screen_scale_width,height:44*screen_scale_width}}
                                             onClick={this._router.bind(this,"/Contact_us",0)}/>
                                        <img key={'footer_icon2'} src={footer_icon2} style={{cursor:'pointer',width:44*screen_scale_width,height:44*screen_scale_width,marginLeft:31*screen_scale_width}}
                                             onClick={this._jump.bind(this,"github")}/>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <span style={{fontSize: 16*screen_scale_width,
                            color: "#B0B0B0",
                            letterSpacing: 0,
                            marginTop:22*screen_scale_width}}>Copyright © pet | 京ICP备19030700号-1 | Song技术支持</span>
                    </div>
                </div>
            </Layout>
        )
    }
}

export default Tutorials_CN;


