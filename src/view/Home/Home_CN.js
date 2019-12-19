import React, { Component } from 'react';
import { Layout, Menu, Icon } from 'antd';
import { Button , Col, Row} from 'antd';
import Content_2_compent from './Home_content_2_component'
import Content_Tutorials from './Content_Tutorials'
import Home_About_pet_component from './Home_About_pet_component'
import Home_Community_component from './Home_Community_component'
import Home_model_zoo_component from './Home_model_zoo_component'

import { Home_model_zoo_component_data } from './Home_model_zoo_component_data'

import './Home.less'
import { OverPack } from 'rc-scroll-anim';
import QueueAnim from 'rc-queue-anim';
import BannerAnim, { Element,Thumb, Arrow} from 'rc-banner-anim';
import { TweenOneGroup } from 'rc-tween-one';
import TweenOne from 'rc-tween-one';

import Pet_img from "../../asset/HOME_icon/1x/Pet.png";
import header_logo from "../../asset/HOME_icon/logo/pet-logo1.png";
import footer_logo from "../../asset/HOME_icon/logo/pet-footer3.png";
import Header_banner from "../../asset/HOME_icon/banner.jpg"

import Functions_icon from "../../asset/HOME_icon/2x/icon1@2x.png"
import Features_icon from "../../asset/HOME_icon/2x/icon2@2x.png"
import Contrast_icon from "../../asset/HOME_icon/2x/icon3@2x.png"
import Expand_icon from "../../asset/HOME_icon/2x/icon4@2x.png"

import quic_pic_icon from '../../asset/HOME_icon/2x/Quick start Copy 2@2x.png'
import quic_pic_icon_hover from '../../asset/HOME_icon/2x/icon5 copy@2x.png'

import pri_pic_icon from '../../asset/HOME_icon/2x/Primary@2x.png'
import pri_pic_icon_hover from '../../asset/HOME_icon/2x/icon6 copy.png'

import int_pic_icon from '../../asset/HOME_icon/2x/Intermediate Copy 2@2x.png'
import int_pic_icon_hover from '../../asset/HOME_icon/2x/icon7 copy@2x.png'

import adv_pic_icon from '../../asset/HOME_icon/2x/Advanced Copy 2@2x.png'
import adv_pic_icon_hover from '../../asset/HOME_icon/2x/icon8 copy@2x.png'

import classification_icon from '../../asset/HOME_icon/2x/分类@2x.png'
import detection_icon from '../../asset/HOME_icon/2x/检测@2x.png'
import segmentation_icon from '../../asset/HOME_icon/2x/分割@2x.png'
import posture_icon from '../../asset/HOME_icon/2x/关键点@2x.png'
import face_icon from '../../asset/HOME_icon/2x/人脸@2x.png'
import parsing_icon from '../../asset/HOME_icon/2x/人体@2x.png'
import Dense_pose_icon from '../../asset/HOME_icon/2x/密集@2x.png'

import bjbg from '../../asset/HOME_icon/2x/bj@2x.jpg'
import github_icon from '../../asset/HOME_icon/2x/github.png'
import pytorch_icon from '../../asset/HOME_icon/2x/pytorch@2x.png'
import BUPT_icon from '../../asset/HOME_icon/2x/bupt.png'

import footer_icon2 from '../../asset/HOME_icon/2x/g@2x.png'
import footer_icon1 from '../../asset/HOME_icon/2x/email@2x.png'


import 'rc-banner-anim/assets/index.css';

import {screen_width,model_width, model_height,screen_scale_width,screen_scale_height} from '../../common/parameter/parameters'
import footer_back from "../../asset/HOME_icon/footer.jpg";
import {home_use_pet_data} from "./Home_use_pet_data";
import Home_use_pet from "./Home_use_pet";

const BgElement = Element.BgElement;
const { animType, setAnimCompToTagComp } = BannerAnim;
const {
    Header, Content, Footer, Sider,
} = Layout;

const { SubMenu } = Menu;

//13 687 1440

//  1397
// let screen_width = document.documentElement.clientWidth;
// // 798
// let screen_height = document.documentElement.clientHeight;
//
// let model_width = screen_width > 1920 ? 1920 : screen_width;
// let model_height = screen_height > 1080 ? 1080 : screen_height;
//
// let screen_scale_width = model_width/1920;
// let screen_scale_width = model_height/1080;


const style = {
    wrap:{
        display:'flex',
        flexDirection:'column',
        justifyContent:'space-around',
        backgroundColor:'white',
        width:screen_width,
        overflowX: 'hidden',
    },
    header_banner:{
        background:`url(${Header_banner}) no-repeat `,
        width: model_width+1,
        // width: "100%",
        // height:500,
        height:720*screen_scale_width,
        backgroundSize: '100% 100%',
        display:'flex',
        flexDirection:'column',
        justifyContent:'space-between',
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
    header_menu:{
        lineHeight: `${88.154*screen_scale_width}px`,
        display:'flex',
        // width:'auto',
        width:(958+170)*screen_scale_width,
        backgroundColor:'rgba(0,0,0,0)',
        marginLeft:(418-170)*screen_scale_width,
    },
    content:{

    },
    content_part:{

    },
    part_wrap:{
        display:"flex",
        justifyContent:"center",
        alignItems:"center",
        backgroundColor:'#FFFFFF',
        width:'100%'},
    model_zoo_normal:{
        display:"flex",
        flexWrap:"wrap",
        height:420*screen_scale_width,
        marginLeft:150*screen_scale_width,
        marginRight:150*screen_scale_width,
        marginTop:60*screen_scale_width,
        justifyContent:"space-between",
    },
    model_zoo_more:{
        display:"flex",
        flexWrap:"wrap",
        height:420*2*screen_scale_width,
        marginLeft:150*screen_scale_width,
        marginRight:150*screen_scale_width,
        marginTop:60*screen_scale_width,
        justifyContent:"space-between",
    }

};


export default class Home extends Component{
    // 构造
    constructor(props) {
        super(props);
        // 初始状态

        this._click = this._click.bind(this);
        this._click_tmp = this._click_tmp.bind(this);
        this._model_more = this._model_more.bind(this);
        this._btnonMouseEnter = this._btnonMouseEnter.bind(this);
        this._btnonMouseLeave = this._btnonMouseLeave.bind(this);
        this._handleScroll = this._handleScroll.bind(this)
        this._header_item_onMouseEnter = this._header_item_onMouseEnter.bind(this)
        this._header_item_onMouseLeave = this._header_item_onMouseLeave.bind(this)

        this.state = {
            Model_com_arr:[
                <Home_model_zoo_component num={1} onClick={this._router.bind(this,'/Model_Zoo',0)} img={classification_icon} title={Home_model_zoo_component_data[1].Classification.title}
                                          text={Home_model_zoo_component_data[1].Classification.text}
                                          header_font_size={26}
                                          text_font_size={20}/>,
                <Home_model_zoo_component num={2} onClick={this._router.bind(this,'/Model_Zoo',1)} img={detection_icon} title={Home_model_zoo_component_data[1].Detection.title}
                                          text={Home_model_zoo_component_data[1].Detection.text}
                                          header_font_size={26}
                                          text_font_size={20}/>,
                <Home_model_zoo_component num={3} onClick={this._router.bind(this,'/Model_Zoo',2)} img={segmentation_icon} title={Home_model_zoo_component_data[1].Segmentation.title}
                                          text={Home_model_zoo_component_data[1].Segmentation.text}
                                          header_font_size={26}
                                          text_font_size={20}/>,
                <Home_model_zoo_component num={4} onClick={this._router.bind(this,'/Model_Zoo',3)} img={posture_icon} title={Home_model_zoo_component_data[1].Human_Pose.title}
                                          text={Home_model_zoo_component_data[1].Human_Pose.text}
                                          header_font_size={26}
                                          text_font_size={20}/>,
                <Home_model_zoo_component num={5} onClick={this._router.bind(this,'/Model_Zoo',4)} img={face_icon} title={Home_model_zoo_component_data[1].Face.title}
                                          text={Home_model_zoo_component_data[1].Face.text}
                                          header_font_size={26}
                                          text_font_size={20}/>,
                <Home_model_zoo_component num={6} onClick={this._router.bind(this,'/Model_Zoo',5)} img={parsing_icon} title={Home_model_zoo_component_data[1].Parsing.title}
                                          text={Home_model_zoo_component_data[1].Parsing.text}
                                          header_font_size={26}
                                          text_font_size={20}/>,
                <Home_model_zoo_component num={7} onClick={this._router.bind(this,'/Model_Zoo',6)} img={Dense_pose_icon} title={Home_model_zoo_component_data[1].Dense_Pose.title}
                                          text={Home_model_zoo_component_data[1].Dense_Pose.text}
                                          header_font_size={26}
                                          text_font_size={20}/>,
                <Home_model_zoo_component num={8} text={[]}/>,,
            ],
            model_btn_text:'收起',
            model_btn_icon:"up",
            btn_type : {
                btn_1_type:'',
                btn_2_type:'',
                btn_3_type:'',
            },
            progress:0
        };

    }


    _header_item_onMouseEnter(eventKey){
        let dom = eventKey.domEvent.target
        // dom.style.color = 'rgba(251,139,35,1)'
    }

    _header_item_onMouseLeave(eventKey){
        console.log(`mouse leave`)
        let scrollTop = document.body.scrollTop || document.documentElement.scrollTop
        let font_color = scrollTop/(820*screen_scale_width)*255
        // font_color = font_color > 255 ? 255 : font_color
        font_color = 255

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
        // font_color = font_color > 255 ? 255 : font_color
        font_color = 255

        // console.log(scrollTop)

        let progress_start = 625/0.75*screen_scale_width
        let progress_end = 925/0.75*screen_scale_width

        if (scrollTop < progress_end+50 && scrollTop > progress_start){
            this.setState({
                progress:(scrollTop-progress_start)/(progress_end-progress_start)*100
            })
        }

        if (scrollTop < 820*screen_scale_width){
            header_wrap.style.background = `rgba(61,62,73,${opaque})`
            //这里写死了, 根据不通页面, i 的起始值不同
            other_btn.style.color = `rgba(${font_color},${font_color},${font_color},1)`
            for (let i = 3; i<17 ;i = i+2) {
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
        // font_color = font_color > 255 ? 255 : font_color
        font_color = 255

        let progress_start = 625/0.75*screen_scale_width
        let progress_end = 925/0.75*screen_scale_width

        if (scrollTop < progress_end && scrollTop > progress_start){
            this.setState({
                progress:(scrollTop-progress_start)/(progress_end-progress_start)*100
            })
        }

        header_wrap.style.background = `rgba(61,62,73,${opaque})`
        other_btn.style.color = `rgba(${font_color},${font_color},${font_color},1)`
        for (let i = 3; i<17 ;i = i+2) {
            // console.log(header.children[1].children[i].className)
            header.children[1].children[i].style.color = `rgba(${font_color},${font_color},${font_color},1)`
        }
    }

    componentWillUnmount() {
        window.removeEventListener('scroll',this._handleScroll)
        console.log('home scroll listener remove')
    }

    _router(link,state,e){
        e.stopPropagation()
        document.documentElement.scrollTop = document.body.scrollTop =0;
        // this.props.history.push(link, { some: 'state' });
        console.log(link)
        console.log(state)
        this.props.history.push(link, { some: {
                openKeys:state,
                selectedKeys:(state+1)*10
            } });
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

    _btnonMouseEnter(event){
        console.log(event.target.className);
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



    _model_more(event){
        if (this.state.model_btn_text === '显示全部'){
            this.setState({
                Model_com_arr:[
                    <Home_model_zoo_component num={1} onClick={this._router.bind(this,'/Model_Zoo',0)} img={classification_icon} title={Home_model_zoo_component_data[1].Classification.title}
                                              text={Home_model_zoo_component_data[1].Classification.text}
                                              header_font_size={26}
                                              text_font_size={20}/>,
                    <Home_model_zoo_component num={2} onClick={this._router.bind(this,'/Model_Zoo',1)} img={detection_icon} title={Home_model_zoo_component_data[1].Detection.title}
                                              text={Home_model_zoo_component_data[1].Detection.text}
                                              header_font_size={26}
                                              text_font_size={20}/>,
                    <Home_model_zoo_component num={3} onClick={this._router.bind(this,'/Model_Zoo',2)} img={segmentation_icon} title={Home_model_zoo_component_data[1].Segmentation.title}
                                              text={Home_model_zoo_component_data[1].Segmentation.text}
                                              header_font_size={26}
                                              text_font_size={20}/>,
                    <Home_model_zoo_component num={4} onClick={this._router.bind(this,'/Model_Zoo',3)} img={posture_icon} title={Home_model_zoo_component_data[1].Human_Pose.title}
                                              text={Home_model_zoo_component_data[1].Human_Pose.text}
                                              header_font_size={26}
                                              text_font_size={20}/>,
                    <Home_model_zoo_component num={5} onClick={this._router.bind(this,'/Model_Zoo',4)} img={face_icon} title={Home_model_zoo_component_data[1].Face.title}
                                              text={Home_model_zoo_component_data[1].Face.text}
                                              header_font_size={26}
                                              text_font_size={20}/>,
                    <Home_model_zoo_component num={6} onClick={this._router.bind(this,'/Model_Zoo',5)} img={parsing_icon} title={Home_model_zoo_component_data[1].Parsing.title}
                                              text={Home_model_zoo_component_data[1].Parsing.text}
                                              header_font_size={26}
                                              text_font_size={20}/>,
                    <Home_model_zoo_component num={7} onClick={this._router.bind(this,'/Model_Zoo',6)} img={Dense_pose_icon} title={Home_model_zoo_component_data[1].Dense_Pose.title}
                                              text={Home_model_zoo_component_data[1].Dense_Pose.text}
                                              header_font_size={26}
                                              text_font_size={20}/>,
                    <Home_model_zoo_component num={8} text={[]}/>,,
                ],
                model_btn_text:"收起",
                model_btn_icon:"up",
            })
        }else {
            this.setState({
                Model_com_arr:[
                    <Home_model_zoo_component num={1} onClick={this._router.bind(this,'/Model_Zoo',0)} img={classification_icon} title={Home_model_zoo_component_data[1].Classification.title}
                                              text={Home_model_zoo_component_data[1].Classification.text}
                                              header_font_size={26}
                                              text_font_size={20}/>,
                    <Home_model_zoo_component num={2} onClick={this._router.bind(this,'/Model_Zoo',1)} img={detection_icon} title={Home_model_zoo_component_data[1].Detection.title}
                                              text={Home_model_zoo_component_data[1].Detection.text}
                                              header_font_size={26}
                                              text_font_size={20}/>,
                    <Home_model_zoo_component num={3} onClick={this._router.bind(this,'/Model_Zoo',2)} img={segmentation_icon} title={Home_model_zoo_component_data[1].Segmentation.title}
                                              text={Home_model_zoo_component_data[1].Segmentation.text}
                                              header_font_size={26}
                                              text_font_size={20}/>,
                    <Home_model_zoo_component num={4} onClick={this._router.bind(this,'/Model_Zoo',3)} img={posture_icon} title={Home_model_zoo_component_data[1].Human_Pose.title}
                                              text={Home_model_zoo_component_data[1].Human_Pose.text}
                                              header_font_size={26}
                                              text_font_size={20}/>,
                ],
                model_btn_text:"显示全部",
                model_btn_icon:"down",
            })
        }

    }

    _click_tmp(item, key, keyPath){
        document.documentElement.scrollTop = document.body.scrollTop =0;
        this.props.history.push(item, { some: 'state' });
    }

    _click(item, key, keyPath) {
        item.domEvent.stopPropagation()
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
            case '4':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                console.log(`menu Doc`)
                this.props.history.push('/Doc', { some: 'state' });
                break;
            case '5':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.props.history.push('/Tutorials', { some: {
                        selectedKeys:0
                    } });
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
                break;
            default:
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.props.history.push('/Model_Zoo', { some: 'state' });
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                break;
        }
    }



    render() {
        // console.log(document.documentElement.clientHeight)
        // console.log(document.documentElement.clientWidth)
        let Model_com_arr = this.state.Model_com_arr
        let change_lang = this.props.onClick
        let heart_height = 88.154*screen_scale_width;


        return (
            <Layout style={style.wrap} className='home'>
                <div style={{display:'flex',
                    flexDirection:'row', justifyContent:'center', height:heart_height,width:'100%',backgroundColor:"rgba(0,0,0,0)",position: 'fixed', top:0,zIndex:1,}}
                     ref={(input) => { this.header_wrap = input; }}>
                    <div style={{display:'flex',
                        flexDirection:'row', justifyContent:'start', height:heart_height,width:model_width,backgroundColor:"none",position:"relative"}}
                         ref={(input) => { this.header = input; }}>
                        <div className="logo" style={style.header_logo}
                             onClick={this._router.bind(this,"/",0)}/>
                        <Menu
                            theme="light"
                            mode="horizontal"
                            defaultSelectedKeys={['1']}
                            style={style.header_menu}
                            onClick={this._click}
                        >
                            <Menu.Item key="1" style={{backgroundColor:'rgba(0,0,0,0)',fontSize:24*screen_scale_width,fontWeight:1000,}}>首页</Menu.Item>
                            <Menu.Item key="2" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>关于 Pet</Menu.Item>
                            <Menu.Item key="3" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>安装</Menu.Item>
                            <Menu.Item key="4" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>文档</Menu.Item>
                            <Menu.Item key="5" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>教程</Menu.Item>
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
                <div style={{...style.part_wrap,backgroundColor:"#FE7C18"}}>
                    <div style={style.header_banner}>
                        <div style={{marginLeft:151*screen_scale_width, display:'flex',flexDirection:'column',}}>
                            <BannerAnim prefixCls="header" autoPlay>
                                <Element
                                    prefixCls="banner-user-elem"
                                    key="0"
                                    componentProps={{style:{height:465*screen_scale_width, marginTop:((257-77+50))*screen_scale_width}}}
                                >
                                    <BgElement
                                        key="bg"
                                        className="bg"
                                    />
                                    <TweenOne className="banner-user-title" animation={{ y: 30, opacity: 0, type: 'from' }}>
                                        <span style={{fontSize:38*screen_scale_width,color:"#FFFFFF"}}>基于Pytorch的</span>
                                    </TweenOne>
                                    <TweenOne className="banner-user-text"
                                              animation={{ y: 30, opacity: 0, type: 'from', delay: 100 }}
                                              style={{marginBottom:10*screen_scale_width}}
                                    >
                                    <span style={{fontSize:62*screen_scale_width,color:"#FFFFFF"}}>高效计算机视觉工具
</span>
                                    </TweenOne>
                                    <TweenOne className="banner-user-text"
                                              animation={{ y: 30, opacity: 0, type: 'from', delay: 100 }}
                                    >
                                        <div style={{display:"flex", flexDirection:'row'}}>
                                            <div>
                                                <Button shape="round" style={{width:200*screen_scale_width, height:68*screen_scale_width,
                                                    fontSize:20*screen_scale_width
                                                }} size='small' onClick={this._click_tmp.bind(this,'/Install')}
                                                        className='quick_start_button_1'>
                                                    快速开始
                                                </Button>
                                            </div>
                                            <div style={{marginLeft:30*screen_scale_width}}>
                                                <Button className='home_learn_button_1' shape="round" style={{width:200*screen_scale_width, height:68*screen_scale_width,
                                                    fontSize:20*screen_scale_width,
                                                }} size='small' onClick={this._click_tmp.bind(this,'/About_Pet')}
                                                        // type={this.state.btn_type.btn_1_type} onMouseEnter={this._btnonMouseEnter} onMouseLeave={this._btnonMouseLeave}
                                                >
                                                    了解更多
                                                </Button>
                                            </div>
                                        </div>
                                    </TweenOne>
                                </Element>
                            </BannerAnim>
                        </div>
                        <div ></div>
                    </div>
                </div>
                <div style={style.content}>
                    <div style={style.part_wrap}>
                        <div style={{display:"flex", flexDirection:"column", height:730*screen_scale_width, width:model_width,}}>
                            <div style={{display:'flex',flexDirection:'column', marginTop:120*screen_scale_width,marginLeft:150*screen_scale_width,
                                marginBottom:42*screen_scale_width
                            }}>
                                <div style={{width:72*screen_scale_width, height:2, backgroundColor:'#F98B34'}}></div>
                                <span style={{fontSize:44*screen_scale_width, color:'black', marginTop:20*screen_scale_width}}>关于 Pet</span>
                            </div>
                            <OverPack playScale={0}>
                                <QueueAnim
                                    type="bottom"
                                    key="block"
                                    leaveReverse
                                    style={{marginLeft:150*screen_scale_width,display:"flex",flexWrap:'wrap'}}
                                >
                                    <Home_About_pet_component  propkey={'about_1'} img={Functions_icon} header={"功能"}
                                                               text1={"支持多类计算机视觉领域的任务;"}
                                                               text2={"提供各种计算机视觉领域最前沿算法的具体实现;"}
                                                               text3={"旨在帮助学习者和开发者了解计算机视觉领域并开始自己的研究。"}
                                                               header_font_size={26}
                                                               text_font_size={20}/>
                                    <Home_About_pet_component propkey={'about_2'} left={140*screen_scale_width} img={Features_icon} header={"特性"}
                                                              text1={"提供各种计算机视觉算法实现并不断更新拓展;"}
                                                              text2={"使用python语言，代码风格统一、简洁，适宜快速学习;"}
                                                              text3={"函数功能模块化，根据不同任务灵活配置实现不同的深度学习网络。"}
                                                              header_font_size={26}
                                                              text_font_size={20}/>
                                    <Home_About_pet_component propkey={'about_3'} top={87*screen_scale_width} img={Contrast_icon} header={"对比"}
                                                              text1={"功能全面，支持多种视觉领域的任务;"}
                                                              text2={"健全的模型库，提供大量高质量的预训练模型;"}
                                                              text3={"采用数据并行的训练与测试，Pet在速度和精度上有独特的优势。"}
                                                              header_font_size={26}
                                                              text_font_size={20}/>
                                    <Home_About_pet_component propkey={'about_4'}  top={88*screen_scale_width} left={150*screen_scale_width} img={Expand_icon} header={'拓展'}
                                                              text1={'代码规范、风格统一，扩展性强;'}
                                                              text2={'提供各类基础函数，快速实现的新算法，添加新的功能;'}
                                                              text3={'持续更新新功能、添加新任务，同时欢迎开发者将Pet应用于自己的研究领域。'}
                                                              header_font_size={26}
                                                              text_font_size={20}/>
                                </QueueAnim>
                            </OverPack>
                            {/*<div key={'3'} style={{alignSelf:'center', marginTop:50*screen_scale_width, marginBottom:50*screen_scale_width, marginRight:100*screen_scale_width}}>*/}
                            {/*    <Button className='home_learn_button_2' style={{width:200*screen_scale_width, height:68*screen_scale_width,*/}
                            {/*        fontSize:20*screen_scale_width,color:'#FC8732',*/}
                            {/*        marginLeft:120*screen_scale_width,transition: '.25s all'}} size='small' onClick={this._click_tmp.bind(this,'/About_Pet')}*/}
                            {/*            shape="round"*/}
                            {/*            type={this.state.btn_type.btn_2_type} onMouseEnter={this._btnonMouseEnter} onMouseLeave={this._btnonMouseLeave}>*/}
                            {/*        了解更多*/}
                            {/*    </Button>*/}
                            {/*</div>*/}
                        </div>
                    </div>
                    <div style={{...style.part_wrap}}>
                        <div style={{display:"flex", flexDirection:"column", justifyContent:"space-between", width:model_width,position:"relative"
                        }}>
                            <Home_use_pet en={1} data={home_use_pet_data[1]} progress={this.state.progress}/>
                        </div>
                    </div>
                    <div style={{...style.part_wrap,backgroundColor:"#F7F7F7"}}>
                        <div style={{display:"flex", flexDirection:"column", justifyContent:"space-between", width:model_width,position:"relative"
                        }}>
                            <img src={require('../../asset/HOME_icon/2x/img.png')} style={{width:400*screen_scale_width, height:350*screen_scale_width,position: 'absolute', right: 130*screen_scale_width, top:220*screen_scale_width}}/>
                            <div style={{display:'flex',flexDirection:'column', marginTop:120*screen_scale_width, marginLeft:150*screen_scale_width}}>
                                <div style={{width:72*screen_scale_width, height:2, backgroundColor:'#F98B34'}}></div>
                                <span style={{fontSize:44*screen_scale_width, color:'black', marginTop:30*screen_scale_width}}>教程</span>
                                <span style={{fontSize:20*screen_scale_width, marginTop:39*screen_scale_width,}}>以下是针对初学者和高级开发人员的渐进式教程</span>
                            </div>
                            <OverPack playScale={0.1}>
                                <TweenOneGroup
                                    key="ul"
                                    enter={{
                                        y: '+=30',
                                        opacity: 0,
                                        type: 'from',
                                        ease: 'easeInOutQuad',
                                    }}
                                    leave={{ y: '+=30', opacity: 0, ease: 'easeInOutQuad' }}
                                    style={{display:"flex", flexDirection:"column",
                                        height:400*screen_scale_width, marginLeft:150*screen_scale_width,marginTop:41*screen_scale_width}}
                                >
                                    <Content_Tutorials img={quic_pic_icon} img_hover={quic_pic_icon_hover} onClick={this._router.bind(this,'/Tutorials',0)}
                                                       header={'快速开始'} text={'以cifar10分类网络展示如何使用Pet。'} num={1}
                                                       header_font_size={26}
                                                       text_font_size={20}/>
                                    <Content_Tutorials img={pri_pic_icon} img_hover={pri_pic_icon_hover} onClick={this._router.bind(this,'/Tutorials',1)}
                                                       header={'初级教程'} text={'展示如何实现各类经典计算机视觉算法。'} num={2}
                                                       header_font_size={26}
                                                       text_font_size={20}/>
                                    <Content_Tutorials img={int_pic_icon} img_hover={int_pic_icon_hover} onClick={this._router.bind(this,'/Tutorials',2)}
                                                       header={'中级教程'} text={'灵活使用组件，搭建不同的深度学习网络，开始自己的深度学习的研究。'} num={3}
                                                       header_font_size={26}
                                                       text_font_size={20}/>
                                    <Content_Tutorials img={adv_pic_icon} img_hover={adv_pic_icon_hover} onClick={this._router.bind(this,'/Tutorials',3)}
                                                       header={'高级教程'} text={'拓展自定义组件，添加新的函数和算法，实现最前沿的计算机视觉算法。'} num={4}
                                                       header_font_size={26}
                                                       text_font_size={20}/>
                                </TweenOneGroup>
                            </OverPack>
                        </div>
                    </div>
                    <div style={{...style.part_wrap}}>
                        <div style={{display:"flex", flexDirection:"column", justifyContent:"space-between",width:model_width}}>
                            <div style={{display:'flex',flexDirection:'column', marginTop:120*screen_scale_width, marginLeft:150*screen_scale_width}}>
                                <div style={{width:72*screen_scale_width, height:2, backgroundColor:'#F98B34'}}></div>
                                <span style={{fontSize:44*screen_scale_width, color:'black', marginTop:20}}>模型库</span>
                                <span style={{fontSize:20*screen_scale_width, width: model_width, marginTop:20}}>提供大量高质量模型，开发者可以自由下载用于自己的研究</span>

                            </div>
                            <OverPack playScale={0.1}>
                                {/*<QueueAnim*/}
                                {/*    type="bottom"*/}
                                {/*    key="block"*/}
                                {/*    leaveReverse*/}
                                {/*    style={this.state.model_btn_text === 'show all' ? {display:"flex", flexWrap:"wrap", height:282}:{display:"flex", flexWrap:"wrap",height:595}}*/}
                                {/*>*/}
                                <TweenOneGroup
                                    key="ul"
                                    enter={{
                                        y: '+=30',
                                        opacity: 0,
                                        type: 'from',
                                        ease: 'easeInOutQuad',
                                    }}
                                    leave={{ y: '+=30', opacity: 0, ease: 'easeInOutQuad' }}
                                    style={this.state.model_btn_text === '显示全部' ? style.model_zoo_normal:style.model_zoo_more}
                                >
                                    {Model_com_arr}

                                </TweenOneGroup>
                            </OverPack>
                            <div style={{alignSelf:'center',marginBottom:80*screen_scale_width, marginTop:50*screen_scale_width}}>
                                <Button className='home_learn_button_3' type={this.state.btn_type.btn_3_type} style={{width:200*screen_scale_width, height:68*screen_scale_width, transition: '.25s all',fontSize:20*screen_scale_width,color:'#FC8732',}} size='small'
                                        shape="round"
                                        onClick={this._model_more}
                                        type={this.state.btn_type.btn_3_type} onMouseEnter={this._btnonMouseEnter} onMouseLeave={this._btnonMouseLeave}>
                                    {this.state.model_btn_text}
                                </Button>
                            </div>
                        </div>
                    </div>
                    <div style={{...style.part_wrap,background:`url(${bjbg}) no-repeat `,backgroundSize: '100% 100%',}}>
                        <div style={{display:"flex", flexDirection:"column", justifyContent:"space-between", width:model_width,position:"relative"
                        }}>
                            <div style={{display:'flex',flexDirection:'column', marginTop:120*screen_scale_width, marginLeft:150*screen_scale_width}}>
                                <div style={{width:72*screen_scale_width, height:2, backgroundColor:'#F98B34'}}></div>
                                <span style={{fontSize:44*screen_scale_width, color:'black', marginTop:30*screen_scale_width}}>用户社区</span>
                                <span style={{fontSize:20*screen_scale_width, marginTop:39*screen_scale_width}}>加入Pet开发人员社区学习,讨论，解决问题</span>
                            </div>
                            <OverPack playScale={0.1}
                                      style={{marginTop:80*screen_scale_width,marginBottom:90*screen_scale_width,height:260*screen_scale_width, marginLeft:150*screen_scale_width}}>
                                <TweenOneGroup
                                    key="ul"
                                    enter={{
                                        y: '+=30',
                                        opacity: 0,
                                        type: 'from',
                                        ease: 'easeInOutQuad',
                                    }}
                                    leave={{ y: '+=30', opacity: 0, ease: 'easeInOutQuad' }}
                                    style={{display:"flex",
                                        }}
                                >
                                    <Home_Community_component img={github_icon} onClick={this._jump.bind(this,"github")} part_style={{borderBottomWidth:4*screen_scale_width,borderBottomStyle:'solid',borderBottomColor:"#414447"}}
                                                              header={'Github'} text1={'有关Pet的使用和讨论'}
                                                              text2={'获取源码，报告错误、请求功能、讨'}
                                                              text3={'论问题等'}
                                                              header_font_size={30}
                                                              text_font_size={20}
                                                              num={1}/>
                                    <Home_Community_component img={pytorch_icon} onClick={this._jump.bind(this,"pytorch")}
                                                              part_style={{borderBottomWidth:4*screen_scale_width,borderBottomStyle:'solid',borderBottomColor:"#EE583B"}}
                                                              header_font_size={30}
                                                              text_font_size={20}
                                                              header={'Pytorch'} text1={'有关PyTorch 的使用和谈论'}
                                                              text2={'深度学习运算库PyTorch的源码获取、'}
                                                              text3={'使用与探讨'}
                                                              num={2}
                                                              style={{marginLeft:30*screen_scale_width}} imgstyle={{width:40*screen_scale_width,height:61*screen_scale_width}}/>
                                    <Home_Community_component img={BUPT_icon} onClick={this._jump.bind(this,'bupt')}
                                                              part_style={{borderBottomWidth:4*screen_scale_width,borderBottomStyle:'solid',borderBottomColor:"#073C86"}}
                                                              header_font_size={30}
                                                              text_font_size={20}
                                                              header={'BUPT'} text1={'Pet的发布与支持机构'}
                                                              text2={'官方信息发布、技术支持、联合研发、'}
                                                              text3={'合作咨询'}
                                                              num={3} style={{marginLeft:30*screen_scale_width}}/>
                                </TweenOneGroup>
                            </OverPack>
                        </div>
                    </div>
                </div>
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
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/Install",0)}>快速开始</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/About_Pet",0)}>关于 Pet</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/Doc",0)}>资源</span>
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
                            marginTop:22*screen_scale_width}}>Copyright © Pet | 京ICP备19030700号-1 | Song技术支持</span>
                    </div>
                </div>
            </Layout>
        )
    }
}