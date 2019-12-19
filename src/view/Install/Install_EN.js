import React, { Component } from 'react';
import {Layout, Menu, Icon, Button, Table, Anchor} from 'antd/lib/index';
import { OverPack } from 'rc-scroll-anim';
import QueueAnim from 'rc-queue-anim';
import BannerAnim, { Element,Thumb, Arrow} from 'rc-banner-anim';
import { TweenOneGroup } from 'rc-tween-one';
import TweenOne from 'rc-tween-one';
import {screen_width,screen_height,model_width, model_height,screen_scale_width,screen_scale_height} from '../../common/parameter/parameters'

import Install_codePart_component from './Install_codePart_component'
import {Install_codePart_data_EN} from './Install_codePart_data'
import { Template_li } from './li_template'

import './Install.less'

import footer_icon1 from "../../asset/HOME_icon/2x/email@2x.png";
import footer_icon2 from "../../asset/HOME_icon/2x/g@2x.png";
import Pet_img from "../../asset/HOME_icon/1x/Pet.png";
import header_logo from "../../asset/HOME_icon/logo/pet-logo2.png";
import footer_logo from "../../asset/HOME_icon/logo/pet-footer3.png";
import Header_banner from "../../asset/HOME_icon/banner/install.jpg";
import en_1 from "../../asset/HOME_icon/install/1.jpg";
import en_2 from "../../asset/HOME_icon/install/2.png";
import footer_back from "../../asset/HOME_icon/footer.jpg";
import Help from "../../component/md/Doc/Help";


const BgElement = Element.BgElement;
const { animType, setAnimCompToTagComp } = BannerAnim;
const {
    Header, Content, Footer, Sider,
} = Layout;

const { SubMenu } = Menu;
const { Link } = Anchor

const table_1 = [{
    title: 'Requirement and its vision',
    dataIndex: 'Requirement and its vision',
    key: 'Requirement and its vision',
    className:"table_1_column",
    render: (text, record, index) => {
        let row_color = '#666A6D'
        let row_back_color = '#F7F7F7'
        if (index%2===0){
            row_back_color = '#FFFFFF'
        }
        let row_style = {display:'flex',justifyContent:'center',fontSize:22*screen_scale_width,fontWeight:500,textAlign:'center'}
        return (<p style={{...row_style,color:row_color,}}>{text}</p>)
    },
}, {
    title: 'suggest',
    dataIndex: 'suggest',
    key: 'suggest',
    render: (text, record, index) => {
        let row_back_color = '#F7F7F7'
        if (index%2===0){
            row_back_color = '#FFFFFF'
        }
        return (<p style={{display:'flex',justifyContent:'center',fontSize:22*screen_scale_width,fontWeight:500,textAlign:'center',color:'#666A6D'}}>{text}</p>)
    },
}];

let table_1_data = [
    {
        key: '1',
        'Requirement and its vision': ' tpdm',
        suggest:"3.4.0",

    },
    {
        key: '2',
        'Requirement and its vision': ' opencv-python>=3.4.0',
        suggest:"3.4.0",

    },
    {
        key: '3',
        'Requirement and its vision': 'Numpy>=1.13',
        suggest:"1.15.4",

    },{
        key: '4',
        'Requirement and its vision': 'Pyyaml>=3.12',
        suggest:"3.13",

    },{
        key: '5',
        'Requirement and its vision': 'Setuptools>=18.5',
        suggest:"30.0",

    },{
        key: '6',
        'Requirement and its vision': 'Six>=1.9(1.5.2)',
        suggest:"1.11.0",

    },{
        key: '7',
        'Requirement and its vision': ' Matplotlib>=2.1.0(1.3.1)',
        suggest:"2.2.2",

    },{
        key: '8',
        'Requirement and its vision': 'ninja',
        suggest:"--",

    },
    {
        key: '9',
        'Requirement and its vision': ' Cython',
        suggest:"--",

    },
    {
        key: '10',
        'Requirement and its vision': ' scipy',
        suggest:"--",

    },
    {
        key: '11',
        'Requirement and its vision': ' h5py',
        suggest:"--",

    },
    {
        key: '12',
        'Requirement and its vision': 'ninja',
        suggest:"--",

    },
    {
        key: '13',
        'Requirement and its vision': ' scikit-image',
        suggest:"--",

    },];


let style = {
    wrap:{
        display:'flex',
        flexDirection:'column',
        backgroundColor:'white',
        width:screen_width-1,
        // overflowX: 'hidden',
    },
    part_wrap:{
        display:"flex",
        justifyContent:"center",
        alignItems:"center",
        backgroundColor:'#FFFFFF',
        width:'100%',
    },
    header:{
        width: model_width+1,
        height:'auto',
        display:'flex',
        flexDirection:'column',
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
        // justifyContent:'space-between',
        backgroundColor:'rgba(0,0,0,0)',
        marginLeft:(418-170)*screen_scale_width
    },
    header_banner:{
        background:`url(${Header_banner}) no-repeat `,
        width: '100%',
        height:420*screen_scale_width,
        backgroundSize: '100% 100%',
        display:'flex',
        flexDirection:'column',
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
    li_type:{
        listStyle:'disc outside',
        color:'#A05937',
    },
    li_type_text:{
        // fontSize:27*screen_scale_width,
        fontSize:26*screen_scale_width,
        letterSpacing:0.2,
        color:'#595E61',
    }
}

export default class Install_EN extends React.Component{
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
                if ((i+1)/2 ===3){
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
            if ((i+1)/2 ===3){
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
            case 'issue':
                window.location.href = 'https://github.com/BUPT-PRIV/Pet-dev/issues'
                break;
            case 'install':
                window.location.href = 'https://github.com/BUPT-PRIV/Pet-dev/blob/master/INSTALL.md'
                break;
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
        return (
            <Layout style={style.wrap} className={'Install'}>
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
                            defaultSelectedKeys={['3']}
                            style={style.header_menu}
                            onClick={this._click}
                        >
                            <Menu.Item key="1" style={{backgroundColor:'rgba(0,0,0,0)',fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>Home</Menu.Item>
                            <Menu.Item key="2" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>About Pet</Menu.Item>
                            <Menu.Item key="3" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}>Install</Menu.Item>
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
                                marginRight:24*screen_scale_width,cursor:'pointer',width:44*screen_scale_width,height:28*screen_scale_width,opacity:1,
                                }} onClick={change_lang} ref={(input) => { this.other_btn = input; }}>中</div>
                            <div style={{textAlign:'center', fontSize:20*screen_scale_width,color:"#FFFFFF",backgroundColor:'#FD8023',borderRadius:20,width:44*screen_scale_width,height:28*screen_scale_width, cursor:'pointer',
                            }}
                            >EN</div>
                        </div>
                    </div>
                </div>
                <div style={{...style.part_wrap,...style.header_banner,marginTop:heart_height}}>
                    <div style={style.header} className={'about_pet_header'}>

                        <BannerAnim prefixCls="header" autoPlay>
                            <Element
                                prefixCls="banner-user-elem"
                                key="0"
                                componentProps={{style:{height:'auto',marginLeft:97*screen_scale_width,display:'flex',
                                        flexDirection:"column"
                                    }}}
                            >
                                <BgElement
                                    key="bg"
                                    className="bg"
                                />
                                <TweenOne className="banner-user-title" animation={{ y: 30, opacity: 0, type: 'from' }}
                                          component='span'
                                          style={{fontSize:70*screen_scale_width,color:"#FFFFFF",}}
                                >
                                    Install&nbsp;Pet
                                </TweenOne>
                                <TweenOne className="banner-user-text"
                                          animation={{ y: 30, opacity: 0, type: 'from', delay: 100 }}
                                          component='span'
                                          style={{fontSize:30*screen_scale_width,color:"#FFFFFF"}}
                                >
                                    Transform your ideas  into reality by using Pet
                                </TweenOne>
                            </Element>
                        </BannerAnim>

                    </div>
                </div>
                <div style={style.part_wrap}>
                    <div className={'install_contact_dic'} style={{display:"flex", flexDirection:"row", height:'auto', width:model_width,}}>
                        <div style={{display:"flex", flexDirection:"column", height:'auto',width:1220*screen_scale_width,marginLeft:100*screen_scale_width,
                            // marginTop:(129-(174-129)/2)*screen_scale_width,
                            marginTop:40*screen_scale_width,
                            marginBottom:60*screen_scale_width}}>
                            <p style={{fontSize:45*screen_scale_width,color: "#2B2D2E",letterSpacing:1.32*screen_scale_width,}}>
                                Pet-Environment Deployment
                            </p>
                            <span style={{fontSize:26*screen_scale_width, color: "#484B4D",marginTop:30*screen_scale_width}}>
                                Before install Pet, please complete the content of the environment requirements, The environment requirements as shown in below:
                            </span>
                            <span id={'environment'} style={{fontSize:34*screen_scale_width,color:'#A04E37',letterSpacing:0.15*screen_scale_width, marginTop:20*screen_scale_width,}}>
                                The environment Requirements
                            </span>
                            <ul style={{marginLeft:25*screen_scale_width,marginTop:15*screen_scale_width}}>
                                <li style={style.li_type}><span style={style.li_type_text}>ubuntu >= 14.04
</span></li>
                                <li style={style.li_type}><span style={style.li_type_text}>Python >= 3.5
</span></li>
                                <li style={style.li_type}><span style={style.li_type_text}>CUDA >= 9.0</span></li>
                                <li style={style.li_type}><span style={style.li_type_text}>CUDNN >= 7.0.4</span></li>
                                <li style={style.li_type}><span style={style.li_type_text}>GCC(G++) >= 4.9</span></li>
                            </ul>
                            <span id={'environment'} style={{fontSize:26*screen_scale_width,letterSpacing:0.15*screen_scale_width,
                                marginTop:30*screen_scale_width}}>
                                After complete install the environment requirements, Pet can be installed by following steps:
                            </span>
                            <Template_li>
                                <p id={'pytorch'} style={{color: "#484B4D",fontSize:30*screen_scale_width,letterSpacing:0.2*screen_scale_width,marginTop:20*screen_scale_width}}>
                                    Install Pytorch-1.1.0:
                                </p>
                            </Template_li>
                            <Install_codePart_component en={0} data={Install_codePart_data_EN['Install Pytorch']}
                                                        style={{marginTop:20*screen_scale_width}}/>
                            <Template_li>
                                <p id={'torchvision'} style={{color: "#484B4D",fontSize:30*screen_scale_width,letterSpacing:0.2*screen_scale_width,marginTop:20*screen_scale_width}}>
                                    Install torchvision-0.3.0
                                </p>
                            </Template_li>
                            <Install_codePart_component en={0} data={Install_codePart_data_EN['Install torchvision']}
                                                        style={{marginTop:20*screen_scale_width}}/>
                            <Template_li>
                                <p id={'apex'} style={{color: "#484B4D",fontSize:30*screen_scale_width,letterSpacing:0.2*screen_scale_width,marginTop:20*screen_scale_width}}>
                                    Install NVIDIA apex
                                </p>
                            </Template_li>
                            <Install_codePart_component en={0} data={Install_codePart_data_EN['apex']}
                                                        style={{marginTop:20*screen_scale_width}}/>
                            <Template_li >
                                <p id={'pycocotools'} style={{color: "#484B4D",fontSize:30*screen_scale_width,letterSpacing:0.2*screen_scale_width,marginTop:20*screen_scale_width}}>
                                    Install pycocotools
                                </p>
                            </Template_li>
                            <Install_codePart_component en={0} data={Install_codePart_data_EN['pycocotools']}
                                                        style={{marginTop:20*screen_scale_width}}/>
                            <Template_li>
                                <p id={'install_Pet'} style={{marginTop:50*screen_scale_width,fontSize:30*screen_scale_width,color: "#2B2D2E",letterSpacing:0.15}}>
                                    Download and install Pet
                                </p>
                            </Template_li>
                            <Install_codePart_component en={0} data={Install_codePart_data_EN['Clone Pet']}
                                                        style={{marginTop:20*screen_scale_width}}/>
                            <p style={{color: "#A05937",fontSize:26*screen_scale_width,letterSpacing:0.2*screen_scale_width,marginTop:50*screen_scale_width}}>
                                PS: There will get the error such as "x86_64-linux-gpu-gcc" . Please solve it as follows:
                            </p>
                            <Install_codePart_component en={0} data={Install_codePart_data_EN['ps']}
                                                        style={{marginTop:20*screen_scale_width}}/>
                            <p style={{color: "#A05937",fontSize:26*screen_scale_width,letterSpacing:0.2*screen_scale_width,marginTop:50*screen_scale_width}}>
                                Installation has finished! Starting your Pet.
                            </p>
                        </div>
                        <Anchor className={'dataNav'}
                                offsetTop={heart_height+10}
                                style={{marginLeft:100*screen_scale_width,
                                    marginTop:50*screen_scale_width,
                                    width:380/0.75*screen_scale_width,zIndex:0,
                                }} affix={true}>
                            <span style={{fontSize:34*screen_scale_width,color:'#3765A0',letterSpacing:0.15*screen_scale_width,marginTop:20*screen_scale_width,marginLeft:20*screen_scale_width}}>
                                Installation steps of Pet
                            </span>
                            <Link href={'#environment'} title={'The environment requirements'}/>
                            <Link href={'#pytorch'}  title={'Pytorch-1.1'}/>
                            <Link href={'#torchvision'}  title={'torchvision'}/>
                            <Link href={'#apex'} title={'NVIDIA apex'} />
                            <Link href={'#pycocotools'} title={'pycocotools'} />
                            <Link href={'#install_Pet'} title={'Install Pet'} />
                            <div style={{width:440*screen_scale_width,height:'auto',border:'1px solid #FF8722',marginTop:20*screen_scale_width,display:'flex',flexDirection:"column",marginLeft:20*screen_scale_width}}>
                                {/*// part1*/}
                                <span style={{marginTop:19*screen_scale_width,marginLeft:24*screen_scale_width,fontSize:24*screen_scale_width,color: "#484B4D",}}>
                                    During the installation process,
                                </span>
                                <span style={{marginLeft:24*screen_scale_width,fontSize:24*screen_scale_width,color: "#484B4D",}}>
                                    please refer to the
                                </span>
                                <div style={{marginLeft:24*screen_scale_width,marginTop:20*screen_scale_width,fontSize:24*screen_scale_width,color:'#FFFFFF',backgroundColor:'#FF8722',width:320*screen_scale_width,height:44*screen_scale_width,textAlign:'center',cursor:'pointer',borderRadius:4,lineHeight:`${44*screen_scale_width}px`
                                }}
                                     onClick={this._jump.bind(this,"install")}>
                                    Installation and precautions
                                </div>
                                {/*// part2*/}
                                <span style={{marginTop:19*screen_scale_width,marginLeft:24*screen_scale_width,marginRight:24*screen_scale_width,fontSize:24*screen_scale_width,color: "#484B4D",}}>
                                    Problems and discussions encountered during installation，please refer to the
                                </span>
                                {/*<span style={{marginLeft:24*screen_scale_width,fontSize:24*screen_scale_width,color: "#484B4D",}}>*/}
                                {/*    please refer to the*/}
                                {/*</span>*/}
                                <div style={{marginLeft:24*screen_scale_width,marginTop:20*screen_scale_width,marginBottom:28*screen_scale_width,fontSize:24*screen_scale_width,color:'#FFFFFF',backgroundColor:'#FF8722',width:180*screen_scale_width,height:44*screen_scale_width,textAlign:'center',cursor:'pointer',borderRadius:4,lineHeight:`${44*screen_scale_width}px`
                                }}
                                     onClick={this._jump.bind(this,"issue")}>
                                    Github issues
                                </div>
                            </div>
                        </Anchor>
                    </div>
                </div>
                <div style={{...style.part_wrap, background:`url(${footer_back}) no-repeat `,backgroundSize: '100% 100%',zIndex:1}}>
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
        );
    }

}