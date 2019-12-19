import React from "react";
import TweenOne from 'rc-tween-one';
import {model_width, model_height,screen_scale_width,screen_scale_height} from '../../common/parameter/parameters'

let getDelay = (e, b) => (e % b) * 100 + Math.floor(e / b) * 100 + b * 100;


const liAnim = {
    opacity: 0,
    type: 'from',
    ease: 'easeOutQuad',
};

export const Model_com_1 = ({onClick}) => (
    <TweenOne
        animation={{...liAnim, delay:getDelay(1,4)}}
        key={'Model_com_1'}
        style={{display:"flex", flexDirection:"column", justifyContent:"start",alignItems:'center',width:270*screen_scale_width,height:200/0.86*screen_scale_width,  marginRight:25,marginTop:50*screen_scale_width,cursor:'pointer',}} onClick={onClick}
    >
        <TweenOne
            animation={{
                x: '-=10',
                opacity: 0,
                type: 'from',
                ease: 'easeOutQuad',
            }}
            key="img"
        >
            <img src={require('../../asset/HOME_icon/2x/分类@2x.png')} style={{width:80*screen_scale_width, height:80*screen_scale_width,alignSelf:'center', marginBottom:28*screen_scale_width,}}/>
        </TweenOne>
        <TweenOne
            key="h2_com1_1"
            animation={{ ...liAnim, x: '+=10', delay: getDelay(1,4) + 100 }}
            component="h2"
            style={{marginBottom:22*screen_scale_width}}
        >
            <span style={{fontWeight:500,fontSize:38*screen_scale_width, }}>Classification</span>
        </TweenOne>
        <TweenOne
            key="h2_com2_2"
            animation={{ ...liAnim, x: '+=10', delay: getDelay(1,4) + 200 }}
            component="span"
            style={{marginLeft:40,width:357*screen_scale_width}}
        >
            <span style={{fontSize:22*screen_scale_width, display:"flex", textAlign:'center'}}>Based on cifar 10,ImageNet datasets</span>
        </TweenOne>
    </TweenOne>
);

export const Model_com_2 = ({onClick}) => (
    <TweenOne
        animation={{...liAnim, delay:getDelay(2,4)}}
        key={'Model_com_2'}
        style={{display:"flex", flexDirection:"column", justifyContent:"start",alignItems:'center',width:270*screen_scale_width,height:200/0.86*screen_scale_width,  marginRight:25,marginTop:50*screen_scale_width, marginLeft:130*screen_scale_width,cursor:'pointer',}} onClick={onClick}
    >
        <TweenOne
            animation={{
                x: '-=10',
                opacity: 0,
                type: 'from',
                ease: 'easeOutQuad',
            }}
            key="img"
        >
            <img src={require('../../asset/HOME_icon/2x/主动监测@2x.png')} style={{width:80*screen_scale_width, height:80*screen_scale_width,alignSelf:'center', marginBottom:28*screen_scale_width}}/>
        </TweenOne>
        <TweenOne
            key="h2_com2_1"
            animation={{ ...liAnim, x: '+=10', delay: getDelay(2,4) + 100 }}
            component="h2"
            style={{marginBottom:22*screen_scale_width}}
        >
            <span style={{fontWeight:500,fontSize:38*screen_scale_width}}>Detection</span>
        </TweenOne>
        <TweenOne
            key="h2_com2_2"
            animation={{ ...liAnim, x: '+=10', delay: getDelay(2,4) + 200 }}
            component="span"
            style={{marginLeft:40,width:380*screen_scale_width}}
        >
            <span style={{fontSize:22*screen_scale_width, width:380*screen_scale_width, textAlign:"center",display:"flex"}}>Based on cifar10、ImageNet datasets.</span>
        </TweenOne>
    </TweenOne>
);

export const Model_com_3 = ({onClick}) => (
    <TweenOne
        animation={{...liAnim, delay:getDelay(3,4)}}
        key={'Model_com_1'}
        style={{display:"flex", flexDirection:"column", justifyContent:"start",alignItems:'center',width:270*screen_scale_width,height:200/0.86*screen_scale_width,  marginRight:25*screen_scale_width,marginTop:50*screen_scale_width,marginLeft:130*screen_scale_width,cursor:'pointer',}} onClick={onClick}
    >
        <TweenOne
            animation={{
                x: '-=10',
                opacity: 0,
                type: 'from',
                ease: 'easeOutQuad',
            }}
            key="img"
        >
            <img src={require('../../asset/HOME_icon/2x/分隔@2x.png')} style={{width:80*screen_scale_width, height:80*screen_scale_width,alignSelf:'center', marginBottom:28*screen_scale_width,}}/>
        </TweenOne>
        <TweenOne
            key="h2_com3_1"
            animation={{ ...liAnim, x: '+=10', delay: getDelay(3,4) + 100 }}
            component="span"
            style={{marginBottom:22*screen_scale_width}}
        >
            <span style={{fontWeight:500,fontSize:38*screen_scale_width, marginBottom:15}}>Segmentation</span>
        </TweenOne>
        <TweenOne
            key="h2_com3_2"
            animation={{ ...liAnim, x: '+=10', delay: getDelay(3,4) + 200 }}
            component="span"
            style={{width:355*screen_scale_width}}
        >
            <span style={{fontSize:22*screen_scale_width, width:355*screen_scale_width,marginLeft:40*screen_scale_width,
                textAlign:'center',textAlignLast:'center',display:"flex"}}>Based on ADE20k、MSCOCO、Cityscape、VOC datasets</span>
        </TweenOne>
    </TweenOne>
);

export const Model_com_4 = ({onClick}) => (
    <TweenOne
        animation={{...liAnim, delay:getDelay(4,5)}}
        key={'Model_com_1'}
        style={{display:"flex", flexDirection:"column", justifyContent:"start",alignItems:'center',width:270*screen_scale_width,height:200/0.86*screen_scale_width, textAlign:'center', marginRight:25,marginTop:50*screen_scale_width,marginLeft:130*screen_scale_width,cursor:'pointer',}} onClick={onClick}
    >
        <TweenOne
            animation={{
                x: '-=10',
                opacity: 0,
                type: 'from',
                ease: 'easeOutQuad',
            }}
            key="img"
        >
            <img src={require('../../asset/HOME_icon/2x/姿势@2x.png')} style={{width:80*screen_scale_width, height:80*screen_scale_width,alignSelf:'center', marginBottom:28*screen_scale_width,}}/>
        </TweenOne>
        <TweenOne
            key="h2_com4_1"
            animation={{ ...liAnim, x: '+=10', delay: getDelay(4,5) + 100 }}
            component="h2"
            style={{marginBottom:22*screen_scale_width}}
        >
            <span style={{fontWeight:500,fontSize:38*screen_scale_width, marginBottom:15}}>Posture</span>
        </TweenOne>
        <TweenOne
            key="h2_com4_2"
            animation={{ ...liAnim, x: '+=10', delay: getDelay(4,5) + 200 }}
            component="span"
            style={{width:327*screen_scale_width}}
        >
            <span style={{fontSize:22*screen_scale_width, width:327*screen_scale_width,display:"flex", textAlign:'center'}}>Based on MSCOCO datasets</span>
        </TweenOne>
    </TweenOne>
);


const style = {
    wrap:{
        display:"flex",
        flexDirection:"column",
        justifyContent:"start",
        alignItems:'center',
        height:250/0.86*screen_scale_width,
        width:380*screen_scale_width,
        cursor:'pointer',
    },
    normal:{
        width:80*screen_scale_width, height:80*screen_scale_width,
        marginTop:55*screen_scale_width,marginBottom:(46-9)*screen_scale_width,
        cursor:'pointer',
    },
    hover:{
        width:80*screen_scale_width, height:80*screen_scale_width,
        marginTop:55*screen_scale_width,marginBottom:(46-9)*screen_scale_width,
        cursor:'pointer',
        // transform: "translateY(-5px)",
        // transition: '.3s all',
    },
    wrap_normal:{
        display:"flex",
        flexDirection:"column",
        justifyContent:"start",
        alignItems:'center',
        height:390*screen_scale_width,
        width:390*screen_scale_width,
        cursor:'pointer',
        borderWidth:1*screen_scale_width,
        borderStyle:'solid',
        borderColor:"#D3D6DD",
        borderRadius:4,

        // transform: "translateY(0)",
        // transition: '.3s all',
    },
    wrap_hover:{
        display:"flex",
        flexDirection:"column",
        justifyContent:"start",
        alignItems:'center',
        height:390*screen_scale_width,
        width:390*screen_scale_width,
        cursor:'pointer',
        opacity:1,

        backgroundColor:'#FD8733',
        color:'#FFFFFF',
        borderWidth:1*screen_scale_width,
        borderStyle:'solid',
        borderColor:"#FD8733",
        borderRadius:4,

        // boxShadow: "0 5px 8px rgba(0, 0, 0, 0.15)",
        // transform: "translateY(-5px)",
        // transition: '.3s all',
    }
};


export default class Home_model_zoo_component extends React.Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            hover:false
        };
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

      render() {
          let {...props} = this.props;

          let header_font_size = props.header_font_size || 34;
          let text_font_size = props.text_font_size || 26;

          let content_text = props.text
            // console.log(content_text)
          let content_text_component = content_text.map((item, i)=>{
              return (
                  <span style={{fontSize:text_font_size*screen_scale_width,display:"block", textAlign:'center',textAlignLast:'center', letterSpacing:0.5}}>
                      {item}
                  </span>
              )
          })

          return (
              <TweenOne
                  animation={{...liAnim, delay:getDelay(props.num,5)}}
                  key={`Model_com_1_${props.num}`}
                  style={props.img?(this.state.hover?style.wrap_hover:style.wrap_normal):style.wrap}
                  // style={style.wrap}
                  onClick={props.onClick}
                  onMouseEnter={this._onMouseEnter}
                  onMouseLeave={this._onMouseLeave}
              >
                  <TweenOne
                      animation={{
                          x: '-=10',
                          opacity: 0,
                          type: 'from',
                          ease: 'easeOutQuad',
                      }}
                      key={`home_model_zoo_${props.num}`}
                  >
                      {props.img ? <img src={props.img} style={this.state.hover?style.hover:style.normal}/> : null}
                  </TweenOne>
                  <TweenOne
                      key={`home_model_zoo_h_${props.num}`}
                      animation={{ ...liAnim, x: '+=10', delay: getDelay(props.num,5) + 100 }}
                      component="div"
                      style={{marginBottom:(20-9)*screen_scale_width,}}
                  >
                      <span style={{fontWeight:500,fontSize:header_font_size*screen_scale_width, }}>{props.title}</span>
                  </TweenOne>
                  <TweenOne
                      key={`home_model_zoo_span_${props.num}`}
                      animation={{ ...liAnim, x: '+=10', delay: getDelay(props.num,5) + 200 }}
                      component="div"
                      style={{width:360*screen_scale_width}}
                  >
                      {content_text_component}
                      {/*<span style={{fontSize:text_font_size*screen_scale_width,display:"block", textAlign:'center',textAlignLast:'center', letterSpacing:0.5}}>{props.text}</span>*/}
                      {/*<span style={{fontSize:text_font_size*screen_scale_width,display:"block", textAlign:'center',textAlignLast:'center',letterSpacing:0.5}}>{props.text2}</span>*/}
                  </TweenOne>
              </TweenOne>
          )
      }

}





