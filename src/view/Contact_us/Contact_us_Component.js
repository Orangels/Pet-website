import React, { Component } from 'react';
import {model_width, screen_scale_height, screen_scale_width} from "../../common/parameter/parameters";
import mail_icon from '../../asset/HOME_icon/contact_us/email.png'

const Singal_Component = ({data, index}) => {
    let left = (index%4)===0 ? 15*screen_scale_width : 210*screen_scale_width
    return (
        <div style={{display:'flex', flexDirection:"column", marginLeft:left, alignItems:'center', marginBottom:30*screen_scale_width}}>
            <img src={data.img} style={{width:150*screen_scale_width, height:150*screen_scale_width, borderRadius:150*screen_scale_width/2}}/>
            <span style={{fontSize:28*screen_scale_width, color: '#484B4D', fontWeight:500, marginTop:5*screen_scale_width}}>
                {data['name']}
            </span>
            <span style={{fontSize:22*screen_scale_width, color: '#484B4D', marginTop:15*screen_scale_width}}>
                {data['title']}
            </span>
            <div style={{display:'flex', marginTop:0*screen_scale_width, alignItems:'center'}}>
                <img src={mail_icon} style={{width:20*screen_scale_width, height:20*screen_scale_width}}/>
                <span style={{fontSize:22*screen_scale_width,color:'#B2B2B2', marginLeft:12*screen_scale_width}}>
                    {data['mail']}
                </span>
            </div>
        </div>
    )
}


export default class Contact_us_Component extends Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {};
      }

      render() {

          let {...props} = this.props
          let { data } = props
          let children = data['children']
          let top = props.top || 40*screen_scale_width

          let children_Component_arr = children.map((item,i)=>{
              return (
                  <Singal_Component  data={item} index={i}/>
              )
          })

          return (
              <div style={{display:'flex',flexDirection:'column', marginTop:top,marginLeft:150*screen_scale_width,
                  marginRight:150*screen_scale_width,marginBottom:47*screen_scale_width,
              }}>
                  <div style={{width:72*screen_scale_width, height:2, backgroundColor:'#F98B34'}}></div>
                  <span style={{fontSize:44*screen_scale_width, color:'black', marginTop:5*screen_scale_width}}>{data['title']}</span>
                  <div style={{display:'flex', flexWrap:'wrap-reverse', marginTop:15*screen_scale_width, flexDirection:'row'}}>
                      {children_Component_arr}
                  </div>
              </div>
          )
      }
}