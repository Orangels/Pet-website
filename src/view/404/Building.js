import React, { Component } from 'react';
import workplace from '../../asset/HOME_icon/workplace/workplace.jpg'
import {screen_scale_width, model_width} from "../../common/parameter/parameters";

export default class Building extends Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {};
      }

      render() {

          return (
              <div style={{width:model_width, display:'flex', justifyContent:'center'}}>
                  <img src={workplace} style={{width:800*screen_scale_width, height:600*screen_scale_width}}/>
              </div>
          )
      }
}