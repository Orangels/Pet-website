import React from 'react';
import { Button } from 'antd';
import BannerAnim, { Element,Thumb, Arrow} from 'rc-banner-anim';
import TweenOne from 'rc-tween-one';
import 'rc-banner-anim/assets/index.css';
import {model_width, model_height,screen_scale_width,screen_scale_height} from '../../common/parameter/parameters'

// let screen_width = document.documentElement.clientWidth;
// // 798
// let screen_height = document.documentElement.clientHeight;
//
// let model_width = screen_width > 1920 ? 1920 : screen_width;
// let model_height = screen_height > 1080 ? 1080 : screen_height;
//
// let screen_scale_width = model_width/1920;
// let screen_scale_height = model_height/1080;




const BgElement = Element.BgElement;
const { animType, setAnimCompToTagComp } = BannerAnim;


animType.custom = (elem, type, direction, animData) => {
    console.log(`custom animType, type:${type}`); // eslint-disable-line no-console
    let _y;
    const props = { ...elem.props };
    let children = props.children;
    if (type === 'enter') {
        _y = direction === 'next' ? '100%' : '-100%';
    } else {
        _y = direction === 'next' ? '-10%' : '10%';
        children = React.Children.toArray(children).map(setAnimCompToTagComp);
    }
    return React.cloneElement(elem, {
        ...props,
        animation: {
            ...animData,
            y: _y,
            type: type === 'enter' ? 'from' : 'to',
        },
    }, children);
};

const style = {
    Tutorials_part_quick:{
        display:"flex",
        flexDirection:"column",
        justifyContent:"center",
        alignItems:'center',
        // width:370*screen_scale_width,
        // height:430*screen_scale_height,
        width:370*screen_scale_width,
        height:420*screen_scale_height,
        border:`${1*screen_scale_width}px solid rgba(194,194,194,1)`,
        color: '#000',
        borderRadius: 4,
        // marginBottom: 24,

    },
    Tutorials_part_quick_hover:{
        display:"flex",
        flexDirection:"column",
        justifyContent:"center",
        alignItems:'center',
        // width:370*screen_scale_width,
        // height:430*screen_scale_height,
        width:370*screen_scale_width,
        height:420*screen_scale_height,
        border:`${3*screen_scale_width}px solid rgba(251,139,35,1)`,
        // color: 'rgba(251,139,35,1)',
        cursor:'pointer',
        borderRadius: 4,
    },
    Tutorials_part_line:{
        width:24*screen_scale_width, height:3*screen_scale_width, backgroundColor:'#6A6F73', marginBottom:15
    },
    Tutorials_part_line_over:{
        width:24*screen_scale_width, height:3*screen_scale_width, backgroundColor:'#FB8B23', marginBottom:15
    }
};


export default class Content_2_compent extends React.Component {
    constructor(props) {
        super(props);
        this._onMouseEnter = this._onMouseEnter.bind(this);
        this._onMouseLeave = this._onMouseLeave.bind(this);
    }

    _onMouseEnter(e) {
        this.banner.slickGoTo(1)
    }

    _onMouseLeave(e){
        this.banner.slickGoTo(0)
    }

    render(){

        const { ...props } = this.props;

        return (
            <BannerAnim prefixCls="banner-user-ls" onMouseEnter={this._onMouseEnter} onMouseLeave={this._onMouseLeave}
                        ref={(c) => { this.banner = c; }}
                        type="custom"
                        duration={0}
                        style={{width:370*screen_scale_width,
                            height:420*screen_scale_height,display:'flex',flexDirection:"column"}}
                        onClick={props.onClick}
            >
                <Element
                    prefixCls="banner-user-elem"
                    key="0"
                >
                    <BgElement
                        key="bg_1"
                        className="bg_1"
                    />
                    <TweenOne className="banner-user-title" animation={{ y: 30, opacity: 0, type: 'from' }} style={style.Tutorials_part_quick}>

                        <img src={props.img} style={{width:75*screen_scale_width, height:63*screen_scale_height,alignSelf:'center', marginBottom:15}}/>
                        <div style={style.Tutorials_part_line}></div>
                        <span style={{fontWeight:800,fontSize:20*screen_scale_width, color:"#414447"}}>{props.header}</span>

                    </TweenOne>
                </Element>
                <Element
                    prefixCls="banner-user-elem"
                    key="1"
                >
                    <BgElement
                        key="bg"
                        className="bg"
                    />
                    <TweenOne className="banner-user-title" animation={{ y: 30, opacity: 0, type: 'from' }} style={style.Tutorials_part_quick_hover}>
                            <img src={props.img_hover} style={{width:75*screen_scale_width, height:63*screen_scale_height,alignSelf:'center', marginBottom:15}}/>
                            <div style={style.Tutorials_part_line_over}></div>
                            <span style={{fontWeight:500,fontSize:16*screen_scale_width}}>{props.text}</span>
                        <Button className='home_learn_button_1' type="primary" style={{width:130*screen_scale_width, height:44*screen_scale_height, color:'#FFFFFF',
                            marginTop:30
                        }} size='small' onClick={props.onClick}>
                            Learn More
                        </Button>
                    </TweenOne>
                </Element>
                <Thumb prefixCls="user-thumb" key="thumb" component={TweenOne}
                >
                </Thumb>
                <Arrow arrowType="prev" key="prev" prefixCls="user-arrow"/>
                <Arrow arrowType="next" key="next" prefixCls="user-arrow"/>
            </BannerAnim>);
    }
}

