import React, { Component } from 'react';
import {model_width, screen_scale_width} from "../../common/parameter/parameters";


const style = {
    wrap_div:{
        display:'flex',
        flexDirection:"column",
        backgroundColor:'#EDEDED',
        paddingTop:30*screen_scale_width,
        paddingLeft:30*screen_scale_width,
        overflowX:'auto',
        whiteSpace:"nowrap"
    },
    div_text:[
        {
            color:"#666A6D",
            fontSize:19*screen_scale_width,
            letterSpacing:1*screen_scale_width,
            // letterSpacing:10,
            marginBottom:10*screen_scale_width,
            whiteSpace: "pre"
        },
        {
            color:"#666A6D",
            fontSize:19*screen_scale_width,
            // letterSpacing:0.14,
            letterSpacing:1*screen_scale_width,
            marginBottom:10*screen_scale_width,
            whiteSpace: "pre"
        }
    ]
}

export default class Install_codePart_component extends Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {};
      }

      render() {
          let {...props} = this.props
          let { data } = props;
          let length = data.length
          let en = props.en
          data = data.map((item,i)=>{
                let bottom = style.div_text[en].marginBottom
                if (i === length-1){
                    bottom = style.wrap_div.paddingTop
                }
                return (
                    <span style={{...style.div_text[en],marginBottom:bottom}}>
                        {item}
                    </span>
                )
              })

          return (
              <div style={{...style.wrap_div,...props.style}}>
                  {data}
              </div>
          )
      }
}