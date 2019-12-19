import React, { Component } from 'react';
import { Layout, Menu, Icon } from 'antd/lib/index';
import { Button } from 'antd/lib/index';
import {Model_com_1, Model_com_2,Model_com_3,Model_com_4} from './Home_model_zoo_component'

import './Home.less'
import { OverPack } from 'rc-scroll-anim';
import BannerAnim, { Element } from 'rc-banner-anim';
import TweenOne from 'rc-tween-one';
import 'rc-banner-anim/assets/index.css';

import Pet_img from "../asset/HOME_icon/1x/Pet.png";
import Header_banner from "../asset/HOME_icon/banner.png"
import quic_pic_icon from '../asset/HOME_icon/1x/Quick start.png'
import quic_pic_icon_hover from '../asset/HOME_icon/1x/Quick start Copy 2.png'

const BgElement = Element.BgElement;

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

const style = {
    wrap:{
        display:'flex',
        flexDirection:'column',
        justifyContent:'space-around',
        backgroundColor:'white',
        width:screen_width
    },
    header_banner:{
        // overflowX: 'hidden',
        background:`url(${Header_banner}) no-repeat`,
        width: '100%',
        // height:500,
        height:1000*screen_scale_height,
        backgroundSize: '100% 100%',
        display:'flex',
        flexDirection:'column',
        justifyContent:'space-between',
    },
    header_logo:{
        height:'auto',
        width:240,
        background:`url(${Pet_img}) center no-repeat`,
        backgroundSize:'cover%'
    },
    footer_logo:{
        height:53*screen_scale_height,
        width:152*screen_scale_width,
        background:`url(${Pet_img}) center no-repeat`,
        backgroundSize:'cover%',
        marginLeft:150*screen_scale_width,
        marginTop:74*screen_scale_height
    },
    header_menu:{
        lineHeight: '64px',
        display:'flex',
        width:'auto',
        justifyContent:'space-between',
        backgroundColor:'rgba(0,0,0,0)'
    },
    content:{

    },
    content_part:{

    },
    Tutorials_part_quick:{
        display:"flex", flexDirection:"column", justifyContent:"center",alignItems:'center',width:370*screen_scale_width,height:430*screen_scale_height, border:'1px solid rgba(211,214,221,0.6)', color: '#000'
    },
    Tutorials_part_quick_hover:{
        display:"flex", flexDirection:"column", justifyContent:"center",alignItems:'center',width:370*screen_scale_width,height:430*screen_scale_height, border:'2px solid rgba(251,139,35,1)', color: 'rgba(251,139,35,1)',paddingBottom:100
    },
    Tutorials_part_Primary:{
        display:"flex", flexDirection:"column", justifyContent:"center",alignItems:'center',width:370*screen_scale_width,height:430*screen_scale_height, border:'1px solid rgba(211,214,221,0.6)', color: '#000'
    },
    Tutorials_part_Primary_hover:{
        display:"flex", flexDirection:"column", justifyContent:"center",alignItems:'center',width:370*screen_scale_width,height:430*screen_scale_height, border:'2px solid rgba(251,139,35,1)', color: 'rgba(251,139,35,1)',paddingBottom:100
    },
    Tutorials_part_Intermediate:{
        display:"flex", flexDirection:"column", justifyContent:"center",alignItems:'center',width:370*screen_scale_width,height:430*screen_scale_height, border:'1px solid rgba(211,214,221,0.6)', color: '#000'
    },
    Tutorials_part_Intermediate_hover:{
        display:"flex", flexDirection:"column", justifyContent:"center",alignItems:'center',width:370*screen_scale_width,height:430*screen_scale_height, border:'2px solid rgba(251,139,35,1)', color: 'rgba(251,139,35,1)',paddingBottom:100
    },
    Tutorials_part_Advanced:{
        display:"flex", flexDirection:"column", justifyContent:"center",alignItems:'center',width:370*screen_scale_width,height:430*screen_scale_height, border:'1px solid rgba(211,214,221,0.6)', color: '#000'
    },
    Tutorials_part_Advanced_hover:{
        display:"flex", flexDirection:"column", justifyContent:"center",alignItems:'center',width:370*screen_scale_width,height:430*screen_scale_height, border:'2px solid rgba(251,139,35,1)', color: 'rgba(251,139,35,1)',paddingBottom:100
    },
    Tutorials_part_line:{
        width:20, height:2, backgroundColor:'#6A6F73', marginBottom:15
    }
};


export default class Home extends Component{
    // 构造
    constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            Model_com_arr:[<Model_com_1 />, <Model_com_2 />,<Model_com_3 />,<Model_com_4 />],
            model_btn_text:'show all',
            quick_pic :quic_pic_icon,
            quick_hover:false,
            primary_hover:false,
            intermediate_hover:false,
            advanced_hover:false

        };

        this._click = this._click.bind(this);
        this._model_more = this._model_more.bind(this);
        this._onMouseEnter_quick = this._onMouseEnter_quick.bind(this);
        this._onMouseLeave_quick = this._onMouseLeave_quick.bind(this);

        this._onMouseEnter_primary = this._onMouseEnter_primary.bind(this);
        this._onMouseLeave_primary = this._onMouseLeave_primary.bind(this);

        this._onMouseEnter_Intermediate = this._onMouseEnter_Intermediate.bind(this);
        this._onMouseLeave_Intermediate = this._onMouseLeave_Intermediate.bind(this);

        this._onMouseEnter_Advanced = this._onMouseEnter_Advanced.bind(this);
        this._onMouseLeave_Advanced = this._onMouseLeave_Advanced.bind(this);

        //  TODO 组件数据化
        let Tutorials_part_arr = [{
            pic:quic_pic_icon,
            text:"Quick start",
            props:{
                onMouseEnter:this._onMouseEnter_quick,
                onMouseLeave:this._onMouseLeave_quick,
            },
            style:style.Tutorials_part_quick,
            style_hover:style.Tutorials_part_quick_hover
        },{
            pic:quic_pic_icon,
            text:"Primary tutorial",
        }];
    }

    _onMouseEnter_quick(e){
        this.setState({
            quick_hover :true
        })
    }

    _onMouseLeave_quick(e){
        this.setState({
            quick_hover :false
        })
    }

    _onMouseEnter_primary(e){
        this.setState({
            primary_hover :true
        })
    }

    _onMouseLeave_primary(e){
        this.setState({
            primary_hover :false
        })
    }

    _onMouseEnter_Intermediate(e){
        this.setState({
            intermediate_hover :true
        })
    }

    _onMouseLeave_Intermediate(e){
        this.setState({
            intermediate_hover :false
        })
    }

    _onMouseEnter_Advanced(e){
        this.setState({
            advanced_hover :true
        })
    }
    _onMouseLeave_Advanced(e){
        this.setState({
            advanced_hover :false
        })
    }


    _model_more(event){
        if (this.state.model_btn_text === 'show all'){
            this.setState({
                Model_com_arr:[<Model_com_1 />, <Model_com_2 />,<Model_com_3 />,<Model_com_4 />,<Model_com_1 />, <Model_com_2 />,<Model_com_3 />,<Model_com_4 />],
                model_btn_text:"pack up"
            })
        }else {
            this.setState({
                Model_com_arr:[<Model_com_1 />, <Model_com_2 />,<Model_com_3 />,<Model_com_4 />],
                model_btn_text:"show all"
            })
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
                console.log(`set state ${key}`);
                console.log(`set state ${keyPath}`);
                break;
            case '6':
                this.props.history.push('/Model_Zoo', { some: 'state' });
                break;
            case '7':
                console.log(`set state ${key}`);
                console.log(`set state ${keyPath}`);
                break;
            case '11':
                this.props.history.push('/Model_Zoo', { some: 'state' });
                break;
            default:
                console.log(`set key ${key}`);
                console.log(`set keyPath ${keyPath}`);
                break;
        }
    }



    render() {
        console.log(document.documentElement.clientHeight)
        console.log(document.documentElement.clientWidth)
        let Model_com_arr = this.state.Model_com_arr



        return (
            <Layout style={style.wrap}>
                <div style={style.header_banner}>
                    <div style={{display:'flex',
                        flexDirection:'row', justifyContent:'space-between', heighit:'64',width:'100%',backgroundColor:'none'}}>
                        <div className="logo" style={style.header_logo}/>
                        <Header className="Content_layout_Header" style={{height: 'auto',flex:1, textAlign:'center', backgroundColor:'rgba(0,0,0,0)', marginLeft:100}}>
                            <Menu
                                theme="light"
                                mode="horizontal"
                                defaultSelectedKeys={['1']}
                                style={style.header_menu}
                                onClick={this._click}
                            >
                                <Menu.Item key="1" style={{backgroundColor:'rgba(0,0,0,0)'}}>Home</Menu.Item>
                                <Menu.Item key="2" >About Pet</Menu.Item>
                                <Menu.Item key="3" >Install</Menu.Item>
                                <Menu.Item key="4" >Doc</Menu.Item>
                                <Menu.Item key="5" >Tutorials</Menu.Item>
                                <SubMenu key="6" title="Model Zoo">
                                    <Menu.Item key="11" >Model Zoo</Menu.Item>
                                    <Menu.Item key="12" >Classification</Menu.Item>
                                    <Menu.Item key="13" >Detection</Menu.Item>
                                    <Menu.Item key="14" >Segmentation</Menu.Item>
                                    <Menu.Item key="15" >Posture</Menu.Item>
                                </SubMenu>
                                <Menu.Item key="7" >contact us</Menu.Item>
                            </Menu>
                        </Header>
                        <div style={{alignSelf:'center', width:50, display:"flex"}}>
                            <p style={{textAlign:'center', paddingTop:10}}>中</p>
                        </div>
                    </div>
                    <div style={{paddingLeft:120, display:'flex',flexDirection:'column'}}>
                        <BannerAnim prefixCls="banner-user" autoPlay>
                            <Element
                                prefixCls="banner-user-elem"
                                key="0"
                                componentProps={{style:{height:500, marginTop:250, marginLeft:60}}}
                            >
                                <BgElement
                                    key="bg"
                                    className="bg"
                                />
                                <TweenOne className="banner-user-title" animation={{ y: 30, opacity: 0, type: 'from' }}>
                                    <span style={{fontSize:24}}>A flexble and effcient</span>
                                </TweenOne>
                                <TweenOne className="banner-user-text"
                                          animation={{ y: 30, opacity: 0, type: 'from', delay: 100 }}
                                >
                                    <span style={{fontSize:48}}>APIs for Pytorch</span>
                                </TweenOne>
                                <TweenOne className="banner-user-text"
                                          animation={{ y: 30, opacity: 0, type: 'from', delay: 100 }}
                                >
                                    <div style={{display:"flex", flexDirection:'row'}}>
                                        <div>
                                            <Button type="primary" shape="round" style={{width:100, height:38}} size='small'>
                                                Get Started
                                            </Button>
                                        </div>
                                        <div style={{marginLeft:30}}>
                                            <Button className='home_learn_button_1' shape="round" style={{width:100, height:38, color:'#EC6730', }} size='small'>
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
                <div style={style.content}>

                    <div style={{display:"flex", flexDirection:"column", justifyContent:"space-between", marginLeft:150*screen_scale_width}}>
                        <div style={{display:'flex',flexDirection:'column', marginTop:50}}>
                            <div style={{width:50, height:2, backgroundColor:'#F98B34'}}></div>
                            <span style={{fontSize:24, color:'black', marginTop:20}}>About Pet</span>
                            <p style={{fontSize:14, width: 750, marginTop:20}}>The pytorch implementation, based on the uplayer of
                                encapsulation, provides functions and flexible invocation,which has unique advantages over other frameworks.</p>
                        </div>
                        <div style={{display:'flex',flexDirection:'column',}}>
                            <div style={{display:'flex',justifyContent:"space-between"}}>
                                <div style={{display:'flex', marginLeft:40}}>
                                    <img src={require('../asset/HOME_icon/1x/function.png')} style={{width:81*screen_scale_width, height:95*screen_scale_height, alignSelf:'center',marginBottom:20}}/>
                                    <div style={{display:'flex', flexDirection:"column", marginLeft:20}}>
                                        <span style={{fontSize:18, fontWeight:500,color:'black'}}>Function</span>
                                        <p style={{fontSize:14,width:400}}>Pet is short for PytorchEveryThing, which is a versatile and flexible interface on Pytorch.</p>
                                    </div>
                                </div>
                                <div style={{display:'flex', marginRight:40}}>
                                    <img src={require('../asset/HOME_icon/1x/Features.png')} style={{width:87*screen_scale_width, height:87*screen_scale_height,alignSelf:'center', marginBottom:20}}/>
                                    <div style={{display:'flex', flexDirection:"column",marginLeft:20}}>
                                        <span style={{fontSize:18, fontWeight:500,color:'black'}}>Features</span>
                                        <p style={{fontSize:14, width:400}}>Pet can clearly show the training process and details of different tasks, and it is very easy to learn. </p>
                                    </div>
                                </div>
                            </div>
                            <div style={{display:'flex',justifyContent:"space-between"}}>
                                <div style={{display:'flex', marginLeft:40}}>
                                    <img src={require('../asset/HOME_icon/1x/Contrast.png')} style={{width:81*screen_scale_width, height:95*screen_scale_height, alignSelf:'center', marginBottom:20}}/>
                                    <div style={{display:'flex', flexDirection:"column", marginLeft:20}}>
                                        <span style={{fontSize:18, fontWeight:500,color:'black'}}>Contrast</span>
                                        <p style={{fontSize:14,width:400}}>Compared with other deep learning frameworks, Pet has own unique advantages in speed and accuracy.</p>
                                    </div>
                                </div>
                                <div style={{display:'flex', marginRight:40}}>
                                    <img src={require('../asset/HOME_icon/1x/Expand.png')} style={{width:81*screen_scale_width, height:95*screen_scale_height,alignSelf:'center', marginBottom:20}}/>
                                    <div style={{display:'flex', flexDirection:"column",marginLeft:20}}>
                                        <span style={{fontSize:18, fontWeight:500,color:'black'}}>Expand</span>
                                        <p style={{fontSize:14, width:400}}>Pet is easy to expand and implement the latest and best papers and algorithms</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div style={{alignSelf:'center', marginTop:50, marginBottom:50, marginRight:100}}>
                            <Button className='home_learn_button_2' style={{width:100, height:38, color:'#EC6730',}} size='small'>
                                Learn more
                            </Button>
                        </div>
                    </div>
                    <div style={{display:"flex", flexDirection:"column", justifyContent:"space-between", backgroundColor:'#F7F7F7'}}>
                        <div style={{display:'flex',flexDirection:'column', marginTop:50, marginLeft:150*screen_scale_width}}>
                            <div style={{width:50, height:2, backgroundColor:'#F98B34'}}></div>
                            <span style={{fontSize:24, color:'black', marginTop:20}}>Tutorials</span>
                            <p style={{fontSize:14, width: 750, marginTop:20}}>Pet is used for all kinds of mainstream tasks in the field of computer vision</p>
                        </div>
                        <div style={{display:"flex", justifyContent:"space-around", marginBottom:50, marginLeft:150*screen_scale_width, marginRight:150*screen_scale_width}}>

                            <div className='home_content_part_div' style={this.state.quick_hover? style.Tutorials_part_quick_hover:style.Tutorials_part_quick}
                                 onMouseEnter={this._onMouseEnter_quick}
                                 onMouseLeave={this._onMouseLeave_quick}>
                                <img src={this.state.quick_hover ? quic_pic_icon_hover : quic_pic_icon} style={{width:30, height:30,alignSelf:'center', marginBottom:15}}/>
                                <div style={this.state.quick_hover ? Object.assign({},style.Tutorials_part_line,{backgroundColor:"#FB8B23"}):style.Tutorials_part_line }></div>
                                <span style={{fontWeight:800,fontSize:14}}>Quick start</span>
                            </div>
                            <div style={this.state.primary_hover? style.Tutorials_part_Primary_hover:style.Tutorials_part_Primary}
                                 onMouseEnter={this._onMouseEnter_primary}
                                 onMouseLeave={this._onMouseLeave_primary}>
                                <img src={require('../asset/HOME_icon/1x/Primary Copy 2.png')} style={{width:30, height:30,alignSelf:'center', marginBottom:15}}/>
                                <div style={this.state.primary_hover ? Object.assign({},style.Tutorials_part_line,{backgroundColor:"#FB8B23"}):style.Tutorials_part_line }></div>
                                <span style={{fontWeight:800,fontSize:14}}>Primary tutorial</span>
                            </div>

                            <div style={this.state.intermediate_hover? style.Tutorials_part_Intermediate_hover:style.Tutorials_part_Intermediate}
                                 onMouseEnter={this._onMouseEnter_Intermediate}
                                 onMouseLeave={this._onMouseLeave_Intermediate}>
                                <img src={require('../asset/HOME_icon/1x/Intermediate.png')} style={{width:30, height:30,alignSelf:'center', marginBottom:15}}/>
                                <div style={this.state.intermediate_hover ? Object.assign({},style.Tutorials_part_line,{backgroundColor:"#FB8B23"}):style.Tutorials_part_line }></div>
                                <span style={{fontWeight:800,fontSize:14}}>Intermediate tutorial</span>
                            </div>

                            <div style={this.state.advanced_hover? style.Tutorials_part_Advanced_hover:style.Tutorials_part_Advanced}
                                 onMouseEnter={this._onMouseEnter_Advanced}
                                 onMouseLeave={this._onMouseLeave_Advanced}>
                                <img src={require('../asset/HOME_icon/1x/Advanced.png')} style={{width:30, height:30,alignSelf:'center', marginBottom:15}}/>
                                <div style={this.state.advanced_hover ? Object.assign({},style.Tutorials_part_line,{backgroundColor:"#FB8B23"}):style.Tutorials_part_line }></div>
                                <span style={{fontWeight:800,fontSize:14}}>Advanced start</span>
                            </div>

                        </div>

                    </div>
                    <div style={{display:"flex", flexDirection:"column", justifyContent:"space-between",marginLeft:150*screen_scale_width}}>
                        <div style={{display:'flex',flexDirection:'column', marginTop:50, marginLeft:25}}>
                            <div style={{width:50, height:2, backgroundColor:'#F98B34'}}></div>
                            <span style={{fontSize:24, color:'black', marginTop:20}}>Model Zoo</span>
                            <p style={{fontSize:14, width: 750, marginTop:20}}>One of the advantages of Pet is that we provide thousands of high-quality pre-training models,including various popular network structures and excellent work.The expansion of model zoo is still a focus of our work, and we hope that Pet users can help us to improve constantly.</p>

                        </div>
                        <div style={{display:"flex", justifyContent:"space-around",flexWrap:"wrap"}}>

                            {Model_com_arr}

                        </div>
                        <div style={{alignSelf:'center', marginBottom:50}}>
                            <Button className='home_learn_button_2' style={{width:100, height:38, color:'#EC6730',marginRight:100}} size='small'
                                    onClick={this._model_more}>
                                {this.state.model_btn_text}
                            </Button>
                        </div>
                    </div>
                </div>
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
        )
    }
}