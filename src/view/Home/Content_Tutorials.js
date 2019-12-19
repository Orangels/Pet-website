import React from "react";
import TweenOne from 'rc-tween-one';
import BannerAnim from "rc-banner-anim";
import arrows_icon from '../../asset/HOME_icon/2x/arrows@2x.png'
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
        flexDirection:'row',
        background:'#FFFFFF',
        color:'#000000',
        width:470*screen_scale_width,
        height:68*screen_scale_width,
        borderRadius:4,
        cursor:'pointer',
        transition: '.25s all',
        position:"relative",
        alignItems:"center",
    },
    hover:{
        display:'flex',
        flexDirection:'row',
        background:'#FC8732',
        color:'#FFFFFF',
        width:470*screen_scale_width,
        height:68*screen_scale_width,
        borderRadius:4,
        cursor:'pointer',
        transition: '.25s all',
        position:"relative",
        alignItems:"center",
    }
};

export default class Content_Tutorials extends React.Component{
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
        let {onClick, img, img_hover,header,text,num} = this.props;

        let header_font_size = this.props.header_font_size || 34;
        let text_font_size = this.props.text_font_size || 24;

        return (
            <TweenOne
                animation={{...liAnim, delay:getDelay(num,5)}}
                key={`Content_Tutorials_${num}`}
                style={{display:"flex", justifyContent:"start",alignItems:'center',marginBottom:20*screen_scale_width}}
            >
                <TweenOne
                    animation={{
                        x: '-=10',
                        opacity: 0,
                        type: 'from',
                        ease: 'easeOutQuad',
                    }}
                    key="img"
                    onClick={onClick}
                    onMouseEnter={this._onMouseEnter} onMouseLeave={this._onMouseLeave}
                >
                    <div style={this.state.hover ? style.hover : style.normal}>

                        <img src={this.state.hover ? img_hover : img} style={{width:44*screen_scale_width, height:44*screen_scale_height, marginLeft:30*screen_scale_height}}/>
                        <span style={{fontSize:header_font_size*screen_scale_width, marginLeft:30*screen_scale_height,}}>{header}</span>

                        {this.state.hover? <img src={arrows_icon} style={{width:20*screen_scale_width, height:17*screen_scale_height,
                            position:'absolute', top: 28*screen_scale_width,right: 20*screen_scale_width,}}/>:null}
                    </div>
                </TweenOne>
                <TweenOne
                    key="content_Tutorials_h2"
                    animation={{ ...liAnim, x: '+=10', delay: getDelay(1,4) + 200 }}
                    component="span"
                    style={{marginLeft:30*screen_scale_width}}
                >
                    <span ref={(input) => { this.text = input; }} style={this.state.hover ? {fontSize:(text_font_size+2)*screen_scale_width,fontWeight:500,color:'#202D58',transition: '.25s all'} : {fontSize:text_font_size*screen_scale_width,transition: '.25s all'
                        }}>{text}</span>
                </TweenOne>
            </TweenOne>
        )
    }

}
