import React, { Component } from 'react';
import {screen_scale_width} from "../../common/parameter/parameters";


const style = {
    text:[
        {
            fontSize:34*screen_scale_width,
            color:'#2B2D2E',
        },
        {
            fontSize:34*screen_scale_width,
            color:'#2B2D2E',
        },
]
}


export default class Contact_us_logo_component extends Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {};
      }

      render() {
          let {data ,en} = this.props
          data = data.map((item, i)=>{
              return (
                  <span style={style.text[en]}>
                      {item}
                  </span>
              )
          });
          return(
              <div style={{...{
                      marginTop:17*screen_scale_width, whiteSpace:'pre'
                  }, ...this.props.style}}>
                  {data}
              </div>
          )
      }

}