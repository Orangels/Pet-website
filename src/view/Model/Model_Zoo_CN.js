import React, { Component } from 'react';
import { Layout, Menu, Icon } from 'antd/lib/index';
import Echartstest from "../../component/charts/echarts-splattering";
import TableTest from "../../component/table/table";
import Model_Zoo_content_compent  from "../../component/Model_Zoo_content_compent"
import {
    Classification_Image_charts_data,
    Classification_Cifar_charts_data,
    Classification_Cifar_table_data,
    Classification_Image_table_data,
    Classification_Cifar_detail_table_data,
    Classification_Image_detail_table_data,
    Classification_Image_3rd_table_data,
    Detect_VOC_charts_data,
    Detect_COCO_charts_data,
    Detect_VOC_table_data,
    Detect_COCO_table_data,
    Detect_VOC_detail_table_data,
    Detect_COCO_detail_table_data,
    DSN_Arr_Classification,DSN_Arr_Detection,DSN_Arr_Segmentation,DSN_Arr_Pose} from '../../common/data/Doc/EN/model_zoo_data'

import {
    Classification_cifar_content,
    Classification_image_content,
    Detection_coco_content,
    Detection_voc_content
} from '../../component/table/table_content'

import { model_content_data_CN as model_content_data } from './model_content_data'

import ModeL_title from './ModeL_title'

import './Model_Zoo.less'
import ReactCSSTransitionGroup from 'react-addons-css-transition-group'
import {
    BrowserRouter as Router,
    Route,
    Link,
    Redirect
} from 'react-router-dom'
import {model_height, model_width, screen_scale_width, screen_width} from "../../common/parameter/parameters";
import header_logo from "../../asset/HOME_icon/logo/pet-logo2.png";
import Header_banner from "../../asset/HOME_icon/TOP.jpg";
import footer_logo from "../../asset/HOME_icon/logo/pet-footer3.png";
import footer_back from "../../asset/HOME_icon/footer.jpg";
import footer_icon1 from "../../asset/HOME_icon/2x/email@2x.png";
import footer_icon2 from "../../asset/HOME_icon/2x/g@2x.png";
import workplace from "../../asset/HOME_icon/workplace/workplace.jpg";

const {
    Header, Content, Footer, Sider,
} = Layout;

const { SubMenu } = Menu;

const style = {

    wrap:{
        display:'flex',
        flexDirection:'column',
        backgroundColor:'white',
        width:screen_width,
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
        marginTop:16*screen_scale_width,
        marginLeft:119*screen_scale_width,
        cursor:'pointer',
    },
    header_menu:{
        lineHeight: `${88.154*screen_scale_width}px`,
        display:'flex',
        width:(958+170)*screen_scale_width,
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
        // fontSize:24*screen_scale_width,
        fontSize:22*screen_scale_width,
        // textColor:"#8E8E8E"
        color: '#323232',
        backgroundColor:'#F8F8F8'
        // backgroundColor:'red'
    },
    left_item:{
        fontWeight:500,
        // fontSize:22*screen_scale_width,
        fontSize:18*screen_scale_width,
        // textColor:"#8E8E8E"
        // color: '#4A84D0'
        // color:'#EEAC57',
    },
    right_item:{
        fontWeight:500,
        fontSize:15,
    },
    slider_header:{
        height:80,
        background:'#323232',
        margin:16,
        borderWidth:2,
        borderStyle:"solid",
        borderColor:"orange"
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

    }
};


const routes = [
    { path: '/Classification',
        component: Echartstest
    },
    { path: '/Detection',
        component: Echartstest,
    },
    { path: '/Segmentation',
        component: TableTest
    },
    { path: '/Pose Estimation',
        component: TableTest,
    }
];


// const Placeholder_Component = ()=>(
//     <span style={{fontSize:58, alignSelf:'center', marginTop:100*screen_scale_width}}>
//         施工中...
//     </span>
// )

const Placeholder_Component = ()=>(
    <img src={workplace} style={{width:800*screen_scale_width, height:600*screen_scale_width,marginTop:50*screen_scale_width}}/>
)

class Model_Zoo_EN extends Component{
    // 构造
    constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            data:0,
            current_content:2
        };
        this._click = this._click.bind(this)
        this._click_data = this._click_data.bind(this)
        this._click_nav = this._click_nav.bind(this)
        this._select  = this._select.bind(this)
        this._onMouseEnter = this._onMouseLeave.bind(this)
        this._onMouseLeave = this._onMouseLeave.bind(this)

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
                if ((i+1)/2 ===6){
                    continue;
                }
                if (header.children[1].children[i].className.indexOf('active')!==-1){
                    continue
                }
                header.children[1].children[i].style.color = `rgba(${font_color},${font_color},${font_color},1)`
            }
        }

    }

    componentWillMount() {
        console.log(`will mount ${this.props.selectKey}`)
        switch (this.props.selectKey) {
            case 10:
                this.setState({
                    current_content:1
                })
                break
            case 20:
                this.setState({
                    current_content:12
                })
                break
            case 30:
                this.setState({
                    current_content:21
                })
                break
            case 40:
                this.setState({
                    current_content:31
                })
                break
            case 50:
                this.setState({
                    current_content:41
                })
                break
            case 60:
                this.setState({
                    current_content:51
                })
                break
            case 70:
                this.setState({
                    current_content:61
                })
                break
            default:
                this.setState({
                    current_content:51
                })
                break
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
            if ((i+1)/2 ===6){
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


    _click_nav(item, key, keyPath) {
        var { key, keyPath} = item;
        key = Number(key);
        this.setState({
            current_content:key
        },function () {
            console.log(this.state.current_content)
            console.log(`charts type ${Math.floor(this.state.current_content/10)}`)
        });

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
                this.props.history.push('/Model_Zoo', { some: 'state' });
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

    _click_data(item, key, keyPath) {
        var { key, keyPath} = item;
        key = Number(key)
        // var temp = "";
        // console.log(item)
        // for(var i in item){//用javascript的for/in循环遍历对象的属性
        //     temp += i+":"+item[i]+"\n";
        // }
        // alert(temp);
        // console.log(this.state.current_content);
        // console.log(key);
        this.setState({
            current_content:Number(Math.floor(this.state.current_content/2)*2 + key)
        },function () {
            console.log(this.state.current_content)
        });

    }

    _select( item, key, keyPath, selectedKeys, domEvent){
        var { selectedKeys } = item;

    }

    _onMouseEnter( eventKey, domEvent ){
        console.log('_onMouseEnter')
        console.log(eventKey)
        console.log(domEvent)
    }

    _onMouseLeave( eventKey, domEvent ){
        console.log('_onMouseLeave')
        console.log(eventKey)
        console.log(domEvent)
    }


    render() {
        let slider_left_width = 310*screen_scale_width;
        let slider_right_width = 150;
        let header_img_h = 120*screen_scale_width;
        let charts_data = Detect_VOC_charts_data;
        let table_data = Classification_Cifar_table_data;
        let detail_table_data = Classification_Cifar_detail_table_data;
        let DSN_arr = [];
        // let table_component = Classification_cifar_content;
        let table_component = Classification_cifar_content;
        let model_content_data_current = model_content_data.classification.cifar.Model_title_data
        switch (this.state.current_content) {
            // classification
            case 1:
                // charts_data = Classification_Cifar_charts_data;
                // table_data = Classification_Cifar_table_data;
                // detail_table_data = Classification_Cifar_detail_table_data;
                // DSN_arr = DSN_Arr_Classification;
                // table_component = Classification_cifar_content;
                // model_content_data_current = model_content_data.classification.cifar.Model_title_data
                // break;

                charts_data = Classification_Image_charts_data;
                table_data = Classification_Image_table_data;
                // detail_table_data = Classification_Image_detail_table_data;
                detail_table_data = [];
                DSN_arr = DSN_Arr_Classification;
                table_component = Classification_image_content;
                model_content_data_current = model_content_data.classification.imageNet.Model_title_data
                break;
            case 2:
                charts_data = Classification_Image_charts_data;
                table_data = Classification_Image_table_data;
                // detail_table_data = Classification_Image_detail_table_data;
                detail_table_data = [];
                DSN_arr = DSN_Arr_Classification;
                table_component = Classification_image_content;
                model_content_data_current = model_content_data.classification.imageNet.Model_title_data
                break;
            // detect
            case 11:
                charts_data = Detect_VOC_charts_data;
                table_data = Detect_VOC_table_data;
                detail_table_data = Detect_VOC_detail_table_data;
                DSN_arr = DSN_Arr_Detection;
                table_component = Detection_voc_content;
                break;
            case 12:
                charts_data = Detect_COCO_charts_data;
                table_data = Detect_COCO_table_data;
                detail_table_data = Detect_COCO_detail_table_data;
                DSN_arr = DSN_Arr_Detection;
                table_component = Detection_coco_content;
                model_content_data_current = model_content_data.detection.coco.Model_title_data
                break;
            case 21:
                charts_data = Detect_VOC_charts_data;
                table_data = Classification_Cifar_table_data;
                break;
            case 22:
                charts_data = Detect_VOC_charts_data;
                table_data = Classification_Cifar_table_data;
                break;
        }

        let change_lang = this.props.onClick
        let heart_height = 88.154*screen_scale_width;

        let select_key = `${this.state.current_content}`
        let open_key = `${parseInt(this.state.current_content / 10)*10}`


        return (
            <Layout style={{...style.wrap, }} className={'Model_Zoo Model_Zoo_CN'}>
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
                            defaultSelectedKeys={['6']}
                            style={style.header_menu}
                            onClick={this._click}
                            className={'Doc-header'}
                        >
                            <Menu.Item key="1" style={{backgroundColor:'rgba(0,0,0,0)',fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>首页</Menu.Item>
                            <Menu.Item key="2" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>关于 Pet</Menu.Item>
                            <Menu.Item key="3" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>安装</Menu.Item>
                            <Menu.Item key="4" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>文档</Menu.Item>
                            <Menu.Item key="5" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}
                                       onMouseEnter={this._header_item_onMouseEnter} onMouseLeave={this._header_item_onMouseLeave}>教程</Menu.Item>
                            <Menu.Item key="6" style={{fontSize:24*screen_scale_width,fontWeight:1000,}}>模型库</Menu.Item>
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
                <img src={require('../../asset/HOME_icon/banner/model zoo.jpg')} style={{height:header_img_h,marginTop:heart_height}} />
                <Layout style={{...style.part_wrap,backgroundColor:'#F7F7F7'}}>
                    <div style={{display:"flex", flexDirection:"row", height:'auto', width:model_width,}}>
                        <div style={{
                            overflow: "visible", height: model_height,position: 'sticky', left: 0, top:heart_height,
                            width:slider_left_width,
                            // width:slider_left_width,
                            marginTop:40*screen_scale_width
                        }}    className={'left-sider'}
                        >
                            <Menu theme="light" mode="inline"  defaultOpenKeys={[open_key]} defaultSelectedKeys={[select_key]} onClick={this._click_nav}
                                  onSelect={this._select}
                            >

                                <SubMenu key={'0'}  title={<span style={style.subMenu_item}> 分类</span>}>
                                    {/*{DSN_Arr_Classification.map((item,i)=>{*/}
                                    {/*    return (*/}
                                    {/*        <Menu.Item key={`${1+i}`} style={style.left_item}>{item}</Menu.Item>*/}
                                    {/*    )*/}
                                    {/*})}*/}
                                    <Menu.Item key={`${1}`} style={style.left_item}>ImageNet</Menu.Item>
                                </SubMenu>
                                <SubMenu key={'10'} title={<span style={style.subMenu_item}>检测</span>}>
                                    {/*{DSN_Arr_Detection.map((item,i)=>{*/}
                                    {/*    return (*/}
                                    {/*        <Menu.Item key={`${11+i}`} style={style.left_item}>{item}</Menu.Item>*/}
                                    {/*    )*/}
                                    {/*})}*/}
                                    <Menu.Item key={`${11+1}`} style={style.left_item}>MS COCO</Menu.Item>
                                </SubMenu>
                                <SubMenu key={'20'} title={<span style={style.subMenu_item}> 语义分割</span>}>
                                    {DSN_Arr_Segmentation.map((item,i)=>{
                                        return (
                                            <Menu.Item key={`${21+i}`} style={style.left_item}>{item}</Menu.Item>
                                        )
                                    })}
                                </SubMenu>
                                <SubMenu key={'30'} title={<span style={style.subMenu_item}>关键点</span>}>
                                    {DSN_Arr_Pose.map((item,i)=>{
                                        return (
                                            <Menu.Item key={`${31+i}`} style={style.left_item}>{item}</Menu.Item>
                                        )
                                    })}
                                </SubMenu>
                                <SubMenu key={'40'} title={<span style={style.subMenu_item}>人脸</span>}>
                                    {DSN_Arr_Pose.map((item,i)=>{
                                        return (
                                            <Menu.Item key={`${41+i}`} style={style.left_item}>{item}</Menu.Item>
                                        )
                                    })}
                                </SubMenu>
                                <SubMenu key={'50'} title={<span style={style.subMenu_item}>人体部位分割</span>}>
                                    {DSN_Arr_Pose.map((item,i)=>{
                                        return (
                                            <Menu.Item key={`${51+i}`} style={style.left_item}>{item}</Menu.Item>
                                        )
                                    })}
                                </SubMenu>
                                <SubMenu key={'60'} title={<span style={style.subMenu_item}>密集姿态</span>}>
                                    {DSN_Arr_Pose.map((item,i)=>{
                                        return (
                                            <Menu.Item key={`${61+i}`} style={style.left_item}>{item}</Menu.Item>
                                        )
                                    })}
                                </SubMenu>
                            </Menu>
                        </div>

                        <Layout className="Content_layout" style={{ paddingLeft:40*screen_scale_width,paddingRight:100*screen_scale_width, backgroundColor:"#FFFFFF", width:model_width-slider_left_width, display:'flex', justifyContent:'center',
                            flexDirection:'row'
                        }}>
                            {
                                this.state.current_content < 20 ? (
                                    <Content className="Content_layout_Content" style={{display:'flex', flexDirection:'column', height:'auto',
                                        marginBottom:20}}>
                                        <ModeL_title en={1} data={model_content_data_current} />
                                        <Echartstest style={{ width: 'auto', height: document.body.clientHeight-200,marginBottom:20*screen_scale_width}} charts_data={ charts_data } type={Math.floor(this.state.current_content/10)}/>
                                        <TableTest style={{ width: 'auto', height: 'auto', marginTop:51*screen_scale_width}}
                                                   data_3rd={Classification_Image_3rd_table_data}
                                                   table_data={ table_data } detail_table_data={detail_table_data} type={Math.floor(this.state.current_content/10)} en={1}/>

                                    </Content>
                                ):(<Placeholder_Component />)
                            }
                        </Layout>
                    </div>
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



export default Model_Zoo_EN;