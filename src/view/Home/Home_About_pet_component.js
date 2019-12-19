import React from 'react'
import {screen_scale_height, screen_scale_width} from "../../common/parameter/parameters";


const style = {
    normal:{
        display:'flex',
    },
    hover:{
        display:'flex',
    }
};


export default class Home_About_pet_component extends React.Component{
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
          let {...props} = this.props
          let img_width = 38;
          let img_height = 38;
          let marginTop_var = 15
              // console.log(props.key);
          if (props.header === 'Functions' || props.header === '功能'){
              img_height = 45.6 //72/60*38
              marginTop_var = 10
          }
          let header_font_size = props.header_font_size || 30;
          let text_font_size = props.text_font_size || 26;

          return (
              <div key={props.propkey} style={{...style.normal,marginLeft:props.left,marginTop:props.top}}>
                  <img src={props.img} style={{width:img_width*screen_scale_width, height:img_height*screen_scale_width, marginTop:marginTop_var*screen_scale_width}}/>
                  <div style={{display:"flex",flexDirection:"column", marginLeft:20*screen_scale_width}}>
                      <span style={{fontSize:header_font_size*screen_scale_width,color:'#414447'}}>{props.header}</span>
                      <span style={{fontSize:text_font_size*screen_scale_width,colr:'#6D7276', }}>{props.text1}</span>
                      <span style={{fontSize:text_font_size*screen_scale_width,colr:'#6D7276', }}>{props.text2}</span>
                      <span style={{fontSize:text_font_size*screen_scale_width,colr:'#6D7276', }}>{props.text3}</span>
                  </div>
              </div>
          )
      }
}