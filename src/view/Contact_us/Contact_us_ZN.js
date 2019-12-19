import React, { Component } from 'react';
import { Layout, Menu, Icon, Button, Table } from 'antd/lib/index';
import { OverPack } from 'rc-scroll-anim';
import QueueAnim from 'rc-queue-anim';
import BannerAnim, { Element,Thumb, Arrow} from 'rc-banner-anim';
import { TweenOneGroup } from 'rc-tween-one';
import TweenOne from 'rc-tween-one';
import {screen_width,model_width, model_height,screen_scale_width,screen_scale_height} from '../../common/parameter/parameters'

import Contact_us_content from "./Contact_us_content";
import Contact_us_Component from './Contact_us_Component'
import {Contact_us_data_CN as Contact_us_data} from './Contact_us_data'

import './Contact_us.less'

import footer_icon1 from "../../asset/HOME_icon/2x/email@2x.png";
import footer_icon2 from "../../asset/HOME_icon/2x/g@2x.png";
import Pet_img from "../../asset/HOME_icon/1x/Pet.png";
import header_logo from "../../asset/HOME_icon/logo/pet-logo2.png";
import footer_logo from "../../asset/HOME_icon/logo/pet-footer3.png";
import Header_banner from "../../asset/HOME_icon/contact_us/contact_us_banner_1.jpg";
import en_1 from "../../asset/HOME_icon/install/1.jpg";
import en_2 from "../../asset/HOME_icon/install/2.png";
import footer_back from "../../asset/HOME_icon/footer.jpg";
import lab_logo from "../../asset/HOME_icon/contact_us/contact_us_lab_logo.png";
import Contact_us_logo_component from "./Contact_us_logo_component";
import lab_banner from "../../asset/HOME_icon/contact_us/contact_us_banner_2.jpg";



const BgElement = Element.BgElement;
const { animType, setAnimCompToTagComp } = BannerAnim;
const {
    Header, Content, Footer, Sider,
} = Layout;

const { SubMenu } = Menu;


let style = {
    wrap:{
        display:'flex',
        flexDirection:'column',
        backgroundColor:'white',
        width:screen_width,
        overflowX: 'hidden',
    },
    part_wrap:{
        display:"flex",
        justifyContent:"center",
        alignItems:"flex-start",
        backgroundColor:'#FFFFFF',
        width:'100%'
    },
    header:{
        width: model_width+1,
        height:'auto',
        display:'flex',
        justifyContent:'space-between'
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
        marginLeft:(418-170)*screen_scale_width
        // marginLeft:358*screen_scale_width
    },
    header_banner:{
        background:`url(${Header_banner}) no-repeat `,
        width: '100%',
        height:936/2*screen_scale_width,
        backgroundSize: '100% 100%',
        display:'flex',
        // flexDirection:'column',
    },
    content:{

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
}

export default class Contact_us_ZN extends React.Component{
    constructor(props) {
        super(props);

        this._click = this._click.bind(this)

        this._handleScroll = this._handleScroll.bind(this)
        this._header_item_onMouseEnter = this._header_item_onMouseEnter.bind(this)
        this._header_item_onMouseLeave = this._header_item_onMouseLeave.bind(this)
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
                if ((i+1)/2 ===7){
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

        // header_wrap.style.background = `rgba(0,0,0,${opaque})`
        header_wrap.style.background = `rgba(61,62,73,${opaque})`
        other_btn.style.color = `rgba(${font_color},${font_color},${font_color},1)`
        for (let i = 1; i<17 ;i = i+2) {
            if ((i+1)/2 ===7){
                continue;
            }
            header.children[1].children[i].style.color = `rgba(${font_color},${font_color},${font_color},1)`
        }
    }

    componentWillUnmount() {
        window.removeEventListener('scroll',this._handleScroll)
        console.log('home scroll listener remove')
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
        let change_lang = this.props.onClick
        let heart_height = 88.154*screen_scale_width;
        let header_img_h = 120*screen_scale_width;
        return (
            <Layout style={style.wrap} className={'Contact_us Contact_us_ZN'}>
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
                            defaultSelectedKeys={['7']}
                            style={style.header_menu}
                            onClick={this._click}
                        >
                            <Menu.Item key="1" style={{backgroundColor:'rgba(0,0,0,0)',fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>首页</Menu.Item>
                            <Menu.Item key="2" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}
                            >关于 PET</Menu.Item>
                            <Menu.Item key="3" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>安装</Menu.Item>
                            <Menu.Item key="4" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>文档</Menu.Item>
                            <Menu.Item key="5" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>教程</Menu.Item>
                            <Menu.Item key="6" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>模型库</Menu.Item>
                            <Menu.Item key="7" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       >联系我们</Menu.Item>
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
                <div style={{...style.part_wrap,...style.header_banner,marginTop:heart_height}}>
                    <div style={style.header} className={'contact_us_header'}>
                        <div style={{display:'flex',flexDirection:'column',
                            marginTop:58*screen_scale_width,marginLeft:91*screen_scale_width
                        }}
                        >
                            <img src={lab_logo} style={{width:526/2*screen_scale_width, height:88/2*screen_scale_width,
                                cursor:'pointer'}}
                                 onClick={this._router.bind(this,"/lab",0)}/>
                            <Contact_us_logo_component en={1} data={Contact_us_data['lab']} style={{marginTop:50*screen_scale_width}}/>
                        </div>
                        <img src={lab_banner} style={{width:960*screen_scale_width,height:300*screen_scale_width,
                            cursor:'pointer',
                            marginTop:168*screen_scale_width}}
                             onClick={this._router.bind(this,"/lab",0)}/>
                    </div>
                </div>
                <div style={style.content}>
                    <div style={{display:"flex", flexDirection:"column", height:'auto', width:model_width, flexWrap:'wrap',backgroundColor:'#FFFFFF',marginTop:59*screen_scale_width}}>
                        {/*<Contact_us_content data={Contact_us_data.ORGANIZER}  en={1}/>*/}
                        <Contact_us_Component data={Contact_us_data.ORGANIZER} top={10*screen_scale_width}/>
                        <Contact_us_Component data={Contact_us_data.CONTRIBUTER} top={10*screen_scale_width}/>
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
                                    <span style={{fontSize:22*screen_scale_width, color:'#D3D6DD', marginBottom:20*screen_scale_width}}>PET</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/",0)}>快速开始</span>
                                    <span style={{cursor:'pointer',fontSize:18*screen_scale_width, color:'#828282',marginBottom:10.3*screen_scale_width}} onClick={this._router.bind(this,"/",0)}>关于 PET</span>
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
                            marginTop:22*screen_scale_width}}>Copyright © PET | 京ICP备19030700号-1 | Song技术支持</span>
                    </div>
                </div>
            </Layout>
        );
    }

}