import React from "react";
import TweenOne from 'rc-tween-one';
import BannerAnim from "rc-banner-anim";
import {model_width, model_height,screen_scale_width,screen_scale_height} from '../../common/parameter/parameters'

let getDelay = (e, b) => (e % b) * 100 + Math.floor(e / b) * 100 + b * 100;

// let screen_scale_width = 1;
// let screen_scale_height = 1;

// const delay = getDelay(i, 24 / 6);
const liAnim = {
    opacity: 0,
    type: 'from',
    ease: 'easeOutQuad',
};

const style = {
    normal:{
        display:'flex',
        cursor:'pointer',
        backgroundColor:"#FFFFFF",
        width:520*screen_scale_width,
        height:260*screen_scale_width,
        marginBottom:90*screen_scale_width,
        transform: "translateY(0)",
        transition: '.3s all',
    },
    hover:{
        display:'flex',
        cursor:'pointer',
        backgroundColor:"#FFFFFF",
        width:520*screen_scale_width,
        height:260*screen_scale_width,
        marginBottom:90*screen_scale_width,
        boxShadow: "0 5px 8px rgba(0, 0, 0, 0.15)",
        transform: "translateY(-5px)",
        transition: '.3s all',
        opacity:1
    }
};

export default class Home_Community_component extends React.Component{
    constructor(props) {
        super(props);
        this.state = {
            hover:false
        }
        this._onMouseEnter = this._onMouseEnter.bind(this);
        this._onMouseLeave = this._onMouseLeave.bind(this);
    }

    _onMouseEnter(e) {
        this.setState({
            hover:true
        })
    }

    _onMouseLeave(e){

        this.setState({
            hover:false
        })

    }

    componentDidMount() {

        // this.setState({
        //     hover:true
        // })

    }

    componentWillReceiveProps(nextProps, nextContext) {

    }

    render() {
        let {...props} = this.props

        let header_font_size = this.props.header_font_size || 30;
        let text_font_size = this.props.text_font_size || 22;

        let wrap_style = {display:"flex", justifyContent:"start",alignItems:'center'}
        let div_style = {display:'flex',cursor:'pointer',backgroundColor:"#FFFFFF",width:520*screen_scale_width,height:260*screen_scale_width,marginBottom:90*screen_scale_width}
        let custom_style = props.style
        let part_style = props.part_style
        let img_style = props.imgstyle
        return (
            <TweenOne
                animation={{...liAnim, delay:getDelay(props.num,5)}}
                key={`Content_Tutorials_${props.num}`}
                style={{...wrap_style, ...custom_style}}
            >
                <TweenOne
                    animation={{
                        x: '-=10',
                        opacity: 0,
                        type: 'from',
                        ease: 'easeOutQuad',
                    }}
                    key="img"
                    onClick={props.onClick}
                    onMouseEnter={this._onMouseEnter}
                    onMouseLeave={this._onMouseLeave}
                    style={this.state.hover?{...style.hover,...part_style}:{...style.normal,...part_style}}
                >
                    <img src={props.img} style={{...{width:57*screen_scale_width, height:57*screen_scale_height,marginLeft: 30*screen_scale_width, marginTop: 30*screen_scale_width,marginRight:22*screen_scale_width},...img_style}}/>
                    <div style={{display:"flex",flexDirection:"column",marginTop: 30*screen_scale_width,}}>
                        <span style={{fontSize:header_font_size*screen_scale_width,marginBottom:18*screen_scale_width}}>{props.header}</span>
                        <span style={{fontSize:text_font_size*screen_scale_width,fontWeight:1000}}>{props.text1}</span>
                        <span style={{fontSize:text_font_size*screen_scale_width,}}>{props.text2}</span>
                        <span style={{fontSize:text_font_size*screen_scale_width,}}>{props.text3}</span>
                    </div>
                </TweenOne>
            </TweenOne>
        )
    }

}
