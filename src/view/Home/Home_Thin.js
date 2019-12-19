import React, { Component } from 'react';
import { Layout, Menu, Icon } from 'antd';
import { Button , Col, Row} from 'antd';
import Content_2_compent from './Home_content_2_component'
import Content_Tutorials from './Content_Tutorials'
import Home_About_pet_component from './Home_About_pet_component'
import Home_Community_component from './Home_Community_component'
import Home_model_zoo_component from './Home_model_zoo_component'
import Home_use_pet from './Home_use_pet'
import {home_use_pet_data} from './Home_use_pet_data'

import './Home.less'
import { OverPack } from 'rc-scroll-anim';
import QueueAnim from 'rc-queue-anim';
import BannerAnim, { Element,Thumb, Arrow} from 'rc-banner-anim';
import { TweenOneGroup } from 'rc-tween-one';
import TweenOne from 'rc-tween-one';

import Pet_img from "../../asset/HOME_icon/1x/Pet.png";
import header_logo from "../../asset/HOME_icon/logo/pet-logo1.png";
// import header_logo from "../../asset/HOME_icon/logo/pet-logo2.png";
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
import footer_back from '../../asset/HOME_icon/footer.jpg'


import 'rc-banner-anim/assets/index.css';

import {system_param, screen_width,model_width, model_height,screen_scale_width,screen_scale_height} from '../../common/parameter/parameters'
import {Home_model_zoo_component_data} from "./Home_model_zoo_component_data";


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
        // width:115*screen_scale_width,
        // height:40*screen_scale_width,
        width:60*screen_scale_width,
        height:60*screen_scale_width,
        marginTop:15*screen_scale_width,
        background:`url(${header_logo}) center no-repeat`,
        // backgroundSize:'cover%'
        backgroundSize:'100% 100%',
        // marginTop:27*screen_scale_width,
        // marginTop:45*screen_scale_width,
        marginLeft:119*screen_scale_width,
        cursor:'pointer',
    },
    footer_logo:{
        // height:53*screen_scale_width,
        // width:152*screen_scale_width,
        height:60*screen_scale_width,
        width:154*screen_scale_width,
        background:`url(${footer_logo}) center no-repeat`,
        backgroundSize: '100% 100%',
        marginLeft:349*screen_scale_width,
        marginTop:40*screen_scale_width,
        cursor:'pointer',
    },
    header_menu:{
        lineHeight: `${88.154*screen_scale_width}px`,
        // lineHeight: `${120*screen_scale_width}px`,
        display:'flex',
        width:(958+170)*screen_scale_width,
        backgroundColor:'rgba(0,0,0,0)',
        marginLeft:(418-170)*screen_scale_width
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
                <Home_model_zoo_component num={1} onClick={this._router.bind(this,'/Model_Zoo',0)} img={classification_icon} title={Home_model_zoo_component_data[0].Classification.title}
                                          text={Home_model_zoo_component_data[0].Classification.text}/>,
                <Home_model_zoo_component num={2} onClick={this._router.bind(this,'/Model_Zoo',1)} img={detection_icon} title={Home_model_zoo_component_data[0].Detection.title}
                                          text={Home_model_zoo_component_data[0].Detection.text}/>,
                <Home_model_zoo_component num={3} onClick={this._router.bind(this,'/Model_Zoo',2)} img={segmentation_icon} title={Home_model_zoo_component_data[0].Segmentation.title}
                                          text={Home_model_zoo_component_data[0].Segmentation.text}/>,
                <Home_model_zoo_component num={4} onClick={this._router.bind(this,'/Model_Zoo',3)} img={posture_icon} title={Home_model_zoo_component_data[0].Human_Pose.title}
                                          text={Home_model_zoo_component_data[0].Human_Pose.text}
                                          />,
                <Home_model_zoo_component num={5} onClick={this._router.bind(this,'/Model_Zoo',4)} img={face_icon} title={Home_model_zoo_component_data[0].Face.title}
                                          text={Home_model_zoo_component_data[0].Face.text}/>,
                <Home_model_zoo_component num={6} onClick={this._router.bind(this,'/Model_Zoo',5)} img={parsing_icon} title={Home_model_zoo_component_data[0].Parsing.title}
                                          text={Home_model_zoo_component_data[0].Parsing.text}
                                          />,
                <Home_model_zoo_component num={7} onClick={this._router.bind(this,'/Model_Zoo',6)} img={Dense_pose_icon} title={Home_model_zoo_component_data[0].Dense_Pose.title}
                                          text={Home_model_zoo_component_data[0].Dense_Pose.text}
                                          />,
                <Home_model_zoo_component num={8} text={[]}/>,
            ],
            model_btn_text:'Pack Up',
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

        let progress_start = 700/0.75*screen_scale_width
        let progress_end = 970/0.75*screen_scale_width

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

        let progress_start = 700/0.75*screen_scale_width
        let progress_end = 970/0.75*screen_scale_width

        if (scrollTop < progress_end+1 && scrollTop > progress_start){
            this.setState({
                progress:(scrollTop-progress_start)/(progress_end-progress_start)*100
            })
        }

        // header_wrap.style.background = `rgba(0,0,0,${opaque})`
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
        event.target.style.color = '#FC8732';
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
        if (this.state.model_btn_text === 'Show All'){
            this.setState({
                Model_com_arr:[
                    <Home_model_zoo_component num={1} onClick={this._router.bind(this,'/Model_Zoo',0)} img={classification_icon} title={Home_model_zoo_component_data[0].Classification.title}
                                              text={Home_model_zoo_component_data[0].Classification.text}/>,
                    <Home_model_zoo_component num={2} onClick={this._router.bind(this,'/Model_Zoo',1)} img={detection_icon} title={Home_model_zoo_component_data[0].Detection.title}
                                              text={Home_model_zoo_component_data[0].Detection.text}/>,
                    <Home_model_zoo_component num={3} onClick={this._router.bind(this,'/Model_Zoo',2)} img={segmentation_icon} title={Home_model_zoo_component_data[0].Segmentation.title}
                                              text={Home_model_zoo_component_data[0].Segmentation.text}/>,
                    <Home_model_zoo_component num={4} onClick={this._router.bind(this,'/Model_Zoo',3)} img={posture_icon} title={Home_model_zoo_component_data[0].Human_Pose.title}
                                              text={Home_model_zoo_component_data[0].Human_Pose.text}
                                              />,
                    <Home_model_zoo_component num={5} onClick={this._router.bind(this,'/Model_Zoo',4)} img={face_icon} title={Home_model_zoo_component_data[0].Face.title}
                                              text={Home_model_zoo_component_data[0].Face.text}/>,
                    <Home_model_zoo_component num={6} onClick={this._router.bind(this,'/Model_Zoo',5)} img={parsing_icon} title={Home_model_zoo_component_data[0].Parsing.title}
                                              text={Home_model_zoo_component_data[0].Parsing.text}
                                              />,
                    <Home_model_zoo_component num={7} onClick={this._router.bind(this,'/Model_Zoo',6)} img={Dense_pose_icon} title={Home_model_zoo_component_data[0].Dense_Pose.title}
                                              text={Home_model_zoo_component_data[0].Dense_Pose.text}
                                              />,
                    <Home_model_zoo_component num={8} text={[]}/>,
                    ],
                model_btn_text:"Pack Up",
                model_btn_icon:"up",
            })
        }else {
            this.setState({
                Model_com_arr:[
                    <Home_model_zoo_component num={1} onClick={this._router.bind(this,'/Model_Zoo',0)} img={classification_icon} title={Home_model_zoo_component_data[0].Classification.title}
                                              text={Home_model_zoo_component_data[0].Classification.text}/>,
                    <Home_model_zoo_component num={2} onClick={this._router.bind(this,'/Model_Zoo',1)} img={detection_icon} title={Home_model_zoo_component_data[0].Detection.title}
                                              text={Home_model_zoo_component_data[0].Detection.text}/>,
                    <Home_model_zoo_component num={3} onClick={this._router.bind(this,'/Model_Zoo',2)} img={segmentation_icon} title={Home_model_zoo_component_data[0].Segmentation.title}
                                              text={Home_model_zoo_component_data[0].Segmentation.text}/>,
                    <Home_model_zoo_component num={4} onClick={this._router.bind(this,'/Model_Zoo',3)} img={posture_icon} title={Home_model_zoo_component_data[0].Human_Pose.title}
                                              text={Home_model_zoo_component_data[0].Human_Pose.text}
                                              />,
                ],

                model_btn_text:"Show All",
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
                // this.props.history.push('/Doc', { some: 'state' });
                this.props.history.push('/Doc', { some: 'state' });
                break;
            case '5':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.props.history.push('/Tutorials', { some: {
                        selectedKeys:0
                    } });
                break;
            case '6':
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
            case '11':
                document.documentElement.scrollTop = document.body.scrollTop =0;
                this.props.history.push('/Model_Zoo', { some: 'state' });
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

        // console.log(document.documentElement.clientWidth)
        // console.log(document.documentElement.clientHeight)
        // console.log(`scal = ${screen_scale_width}`) //0.726
        let Model_com_arr = this.state.Model_com_arr
        let change_lang = this.props.onClick

        let heart_height = 88.154*screen_scale_width;
        // let heart_height = 120*screen_scale_width;
        // console.log(`header_height = ${heart_height}`)

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
                            <Menu.Item key="1" style={{backgroundColor:'rgba(0,0,0,0)',fontSize:24*screen_scale_width,fontWeight:1000,}}>Home</Menu.Item>
                            <Menu.Item key="2" style={{fontSize:24*screen_scale_width,fontWeight:1000, color:'#FFFFFF'}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>About Pet</Menu.Item>
                            <Menu.Item key="3" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>Install</Menu.Item>
                            <Menu.Item key="4" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>Document</Menu.Item>
                            <Menu.Item key="5" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>Tutorials</Menu.Item>
                            <Menu.Item key="6" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>Model Zoo</Menu.Item>
                            <Menu.Item key="7" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>Contact Us</Menu.Item>
                            <Menu.Item key="8" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>Github</Menu.Item>
                        </Menu>
                        <div style={{alignSelf:'center', width:'auto', display:"flex",position:"absolute",right:50*screen_scale_width}}>
                            <div style={{textAlign:'center',fontSize:20*screen_scale_width,
                                marginRight:24*screen_scale_width,cursor:'pointer',width:44*screen_scale_width,height:28*screen_scale_width,}} onClick={change_lang} ref={(input) => { this.other_btn = input; }}>中</div>
                            <div style={{textAlign:'center', fontSize:20*screen_scale_width,color:"#FFFFFF",backgroundColor:'#FD8023',borderRadius:20,width:44*screen_scale_width,height:28*screen_scale_width, cursor:'pointer',
                            }}
                            >EN</div>
                        </div>
                    </div>
                </div>
                <div style={{...style.part_wrap,backgroundColor:"#FE7C18"}}>
                    <div style={style.header_banner}>
                        <div style={{marginLeft:100*screen_scale_width, display:'flex',flexDirection:'column',}}>
                            <BannerAnim prefixCls="header" autoPlay>
                                <Element
                                    prefixCls="banner-user-elem"
                                    key="0"
                                    componentProps={{style:{height:465*screen_scale_width, marginTop:(257-77+50)*screen_scale_width}}}
                                >
                                    <BgElement
                                        key="bg"
                                        className="bg"
                                    />
                                    <TweenOne className="banner-user-text" animation={{ y: 30, opacity: 0, type: 'from' }}
                                              >
                                        <div style={{display:'flex',flexDirection:'row'}}>
                                            <span style={{fontSize:110*screen_scale_width,fontWeight:1000,color:'#FFFFFF'}}>Pet</span>
                                            <div style={{display:'flex',flexDirection:'column',justifyContent:"center", marginLeft:30*screen_scale_width}}>
                                                <span style={{fontSize:34*screen_scale_width,color:'#FFFFFF',}}>
                                                    Pytorch efficient toolbox for
                                                </span>
                                                <span style={{fontSize:34*screen_scale_width,color:'#FFFFFF',}}>
                                                    Computer Vision.
                                                </span>
                                            </div>
                                        </div>
                                    </TweenOne>
                                    <TweenOne className="banner-user-text"
                                              animation={{ y: 30, opacity: 0, type: 'from', delay: 100 }}
                                    >
                                        <div style={{display:"flex", flexDirection:'row'}}>
                                            <div>
                                                <Button shape="round" style={{width:200*screen_scale_width, height:68*screen_scale_width,fontSize:20*screen_scale_width,
                                                }} size='small' onClick={this._click_tmp.bind(this,'/Install')}
                                                        className='quick_start_button_1'>
                                                    Get Started
                                                </Button>
                                            </div>
                                            <div style={{marginLeft:30*screen_scale_width}}>
                                                <Button className='home_learn_button_1' shape="round" style={{width:200*screen_scale_width, height:68*screen_scale_width,
                                                    fontSize:20*screen_scale_width,
                                                }} size='small' onClick={this._click_tmp.bind(this,'/About_Pet')}
                                                        // type={this.state.btn_type.btn_1_type} onMouseEnter={this._btnonMouseEnter} onMouseLeave={this._btnonMouseLeave}
                                                >
                                                    Learn More
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
                        <div style={{display:"flex", flexDirection:"column", height:850*screen_scale_width, width:model_width,}}>
                            <div style={{display:'flex',flexDirection:'column', marginTop:120*screen_scale_width,marginLeft:150*screen_scale_width,
                                marginBottom:42*screen_scale_width
                            }}>
                                <div style={{width:72*screen_scale_width, height:2, backgroundColor:'#F98B34'}}></div>
                                <span style={{fontSize:44*screen_scale_width, color:'black', marginTop:20*screen_scale_width}}>About Pet </span>
                                <span style={{fontSize:26*screen_scale_width, width: model_width, marginTop:20*screen_scale_width}}>Pet is  short for Pytorch efficient toolbox for Computer Vision.</span>
                            </div>
                            <OverPack playScale={0}>
                                <QueueAnim
                                    type="bottom"
                                    key="block"
                                    leaveReverse
                                    style={{marginLeft:150*screen_scale_width,display:"flex",flexWrap:'wrap'}}
                                >
                                    <Home_About_pet_component propkey={'about_1'} img={Functions_icon} header={"Functions"}
                                    text1={"Support various tasks of Computer Vision."}
                                    text2={"Provide implementations of latest deep learning algorithms."}
                                    text3={"Aim to help developers to start their own research quickly."}/>
                                    <Home_About_pet_component propkey={'about_2'} left={100*screen_scale_width} img={Features_icon} header={"Features"}
                                    text1={"Modularization & flexible configuration."}
                                    text2={"Implementation of state-of-the-art algorithms in Computer Vision."}
                                    text3={"Clear coding styles for a friendly learning experience."}/>
                                    <Home_About_pet_component propkey={'about_3'} top={87*screen_scale_width} img={Contrast_icon} header={"Contrast"}
                                    text1={"Support various kinds of tasks in Computer Vision."}
                                    text2={"Provide numerous high-quality pre-trained models."}
                                    text3={"Have unique advantages in speed and accuracy."}/>
                                    <Home_About_pet_component propkey={'about_4'} left={200*screen_scale_width} top={87*screen_scale_width} img={Expand_icon} header={'Expansions'}
                                    text1={'Provide basic functions for validating new algorithms and ideas.'}
                                    text2={'Code with uniform format and styles are easy to expansions.'}                                                                    text3={'Update and expand constantly. Custom extension is supported.'}/>
                                </QueueAnim>
                            </OverPack>
                            {/*<div key={'3'} style={{alignSelf:'center', marginTop:50*screen_scale_width, marginBottom:50*screen_scale_width, marginRight:100*screen_scale_width}}>*/}
                            {/*    <Button className='home_learn_button_2' type={this.state.btn_type.btn_2_type} onMouseEnter={this._btnonMouseEnter} onMouseLeave={this._btnonMouseLeave} shape="round" style={{width:200*screen_scale_width, height:68*screen_scale_width,color:'#FC8732',*/}
                            {/*        fontSize:20*screen_scale_width,*/}
                            {/*        marginLeft:120*screen_scale_width,transition: '.25s all'}} size='small' onClick={this._click_tmp.bind(this,'/About_Pet')}>*/}
                            {/*        Learn More*/}
                            {/*    </Button>*/}
                            {/*</div>*/}
                        </div>
                    </div>
                    <div style={{...style.part_wrap}}>
                        <div style={{display:"flex", flexDirection:"column", justifyContent:"space-between", width:model_width,position:"relative"
                        }}>
                            <Home_use_pet en={0} data={home_use_pet_data[0]} progress={this.state.progress}/>
                        </div>
                    </div>
                    <div style={{...style.part_wrap,backgroundColor:"#F7F7F7"}}>
                        <div style={{display:"flex", flexDirection:"column", justifyContent:"space-between", width:model_width,position:"relative"
                        }}>
                            <img src={require('../../asset/HOME_icon/2x/img.png')} style={{width:400*screen_scale_width, height:350*screen_scale_width,position: 'absolute', right: -20*screen_scale_width, top:250*screen_scale_width}}/>
                            <div style={{display:'flex',flexDirection:'column', marginTop:120*screen_scale_width, marginLeft:150*screen_scale_width}}>
                                <div style={{width:72*screen_scale_width, height:2, backgroundColor:'#F98B34'}}></div>
                                <span style={{fontSize:44*screen_scale_width, color:'black', marginTop:30*screen_scale_width}}>Tutorials</span>
                                <span style={{fontSize:26*screen_scale_width, marginTop:39*screen_scale_width,}}>Here are in-depth tutorials for beginners and advanced developers.</span>
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
                                                       header={'Quick start'} text={'Quickly start with a classification network based on Cifar10.'} num={1}/>
                                    <Content_Tutorials img={pri_pic_icon} img_hover={pri_pic_icon_hover} onClick={this._router.bind(this,'/Tutorials',1)}
                                                       header={'Primary tutorial'} text={'Show how to implement deep learning algorithms in Computer Vision.'} num={2}/>
                                    <Content_Tutorials img={int_pic_icon} img_hover={int_pic_icon_hover} onClick={this._router.bind(this,'/Tutorials',2)}
                                                       header={'Intermediate tutorial'} text={'Implement custom network by using components and start your own research.'} num={3}/>
                                    <Content_Tutorials img={adv_pic_icon} img_hover={adv_pic_icon_hover} onClick={this._router.bind(this,'/Tutorials',3)}
                                                       header={'Advance tutorial'} text={'Expanding components to implement the state-of-the-art deep learning algorithms in Computer Vision.'} num={4}/>
                                </TweenOneGroup>
                            </OverPack>
                        </div>
                    </div>
                    <div style={{...style.part_wrap}}>
                        <div style={{display:"flex", flexDirection:"column", justifyContent:"space-between",width:model_width}}>
                            <div style={{display:'flex',flexDirection:'column', marginTop:120*screen_scale_width, marginLeft:150*screen_scale_width}}>
                                <div style={{width:72*screen_scale_width, height:2, backgroundColor:'#F98B34'}}></div>
                                <span style={{fontSize:44*screen_scale_width, color:'black', marginTop:20*screen_scale_width}}>Model&nbsp;&nbsp;Zoo</span>
                                <span style={{fontSize:26*screen_scale_width, width: model_width, marginTop:39*screen_scale_width}}>Provide numerous high-quality pre-trained models. Developers can download for their own research.</span>

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
                                    style={this.state.model_btn_text === 'Show All' ? style.model_zoo_normal:style.model_zoo_more}
                                >
                                    {Model_com_arr}

                                </TweenOneGroup>
                            </OverPack>
                            <div style={{alignSelf:'center',marginBottom:80*screen_scale_width, marginTop:50*screen_scale_width}}>
                                <Button className='home_learn_button_3' type={this.state.btn_type.btn_3_type} style={{width:200*screen_scale_width, height:68*screen_scale_width, transition: '.25s all',fontSize:20*screen_scale_width, color:'#FC8732',}} size='small'
                                        onMouseEnter={this._btnonMouseEnter} onMouseLeave={this._btnonMouseLeave} shape="round" onClick={this._model_more} >
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
                                <span style={{fontSize:44*screen_scale_width, color:'black', marginTop:30*screen_scale_width}}>Community</span>
                                <span style={{fontSize:26*screen_scale_width, marginTop:39*screen_scale_width}}>Join the Pet developer community to contribute, learn, and get your questions answered.</span>
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
                                                       header={'Github'} text1={'Use and discuss about Pet.'}
                                                              text2={'Get source code，report bugs, request'}
                                                              text3={'features, discuss issues, and more.'}
                                                              num={1}/>
                                    <Home_Community_component img={pytorch_icon} onClick={this._jump.bind(this,"pytorch")}
                                                              part_style={{borderBottomWidth:4*screen_scale_width,borderBottomStyle:'solid',borderBottomColor:"#EE583B"}}
                                                       header={'Pytorch'} text1={'Use and discuss about Pytorch.'}
                                                              text2={'Browse and access discussions on deep'}
                                                              text3={'learning with PyTorch.'} num={2}
                                                              style={{marginLeft:30*screen_scale_width}} imgstyle={{width:40*screen_scale_width,height:61*screen_scale_width}}/>
                                    <Home_Community_component img={BUPT_icon} onClick={this._jump.bind(this,'bupt')}
                                                              part_style={{borderBottomWidth:4*screen_scale_width,borderBottomStyle:'solid',borderBottomColor:"#073C86"}}
                                                       header={'BUPT'} text1={'Official release and support.'}
                                                              text2={'Joint research and development, coo-'}
                                                              text3={'peration & consultation.'} num={3} style={{marginLeft:30*screen_scale_width}}/>
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
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/Install",0)}>Get Started</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/About_Pet",0)}>About Pet</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/Doc",0)}>Resource</span>
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