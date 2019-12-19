import React from 'react'
import {screen_scale_width} from "../../common/parameter/parameters";

const style = {
    title_style:[
        {
            fontSize:58*screen_scale_width,
            color:'#2B2D2E',
            letterSpacing:1.32*screen_scale_width,
            marginTop:40*screen_scale_width
        },
        {
            fontSize:40*screen_scale_width,
            color:'#2B2D2E',
            letterSpacing:0.91*screen_scale_width,
            marginTop:40*screen_scale_width
        }
    ],
    text_style:[
        {
            // fontSize:27*screen_scale_width,
            fontSize:16,
            color:'#484B4D',
            marginTop:20*screen_scale_width,
            textAlign:'justify'
        },
        {
            fontSize:22*screen_scale_width,
            // fontSize:14,
            color:'#484B4D',
            marginTop:20*screen_scale_width
        }
    ],
    title_1_style:[
        {
            fontSize:34*screen_scale_width,
            letterSpacing:0.15*screen_scale_width,
            color:'#484B4D',
            marginTop:50*screen_scale_width,
            marginBottom:40*screen_scale_width
        },
        {
            fontSize:28*screen_scale_width,
            letterSpacing:0.13*screen_scale_width,
            color:'#484B4D',
            marginTop:50*screen_scale_width,
            marginBottom:40*screen_scale_width
        }
    ],
    block_wrap:[
        {
            display:'flex',
            justifyContent:'space-between',
            marginBottom:50*screen_scale_width
        },
        {
            display:'flex',
            justifyContent:'space-between',
            marginBottom:50*screen_scale_width
        }
    ],
    block:[
        {
            display:'flex',
            flexDirection:'column',
            justifyContent:'space-around',
            border:`${1*screen_scale_width}px solid rgba(160,89,55,1)`,
            height:150*screen_scale_width,
            width:470*screen_scale_width,
            padding:`${26*screen_scale_width}px 0px ${17*screen_scale_width}px ${34*screen_scale_width}px`,
            whiteSpace:'pre',
            fontFamily:'-apple-system'
        },
        {
            display:'flex',
            flexDirection:'column',
            justifyContent:'space-around',
            border:`${1*screen_scale_width}px solid rgba(160,89,55,1)`,
            height:150*screen_scale_width,
            width:470*screen_scale_width,
            padding:`${26*screen_scale_width}px 0px ${17*screen_scale_width}px ${34*screen_scale_width}px`,
            whiteSpace:'pre',
            fontFamily:'-apple-system'
        }
    ],
    block_text:[
        {
            // fontSize:22*screen_scale_width,
            fontSize:16,
            letterSpacing:0.15*screen_scale_width,
            color: '#666A6D'
        },
        {
            // fontSize:22*screen_scale_width,
            fontSize:14,
            letterSpacing:0.15*screen_scale_width,
            color: '#666A6D'
        }
    ]
}

class ModeL_title_block extends React.Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {};
      }

      render() {
          let {...props} = this.props
          let data = props.data
          let en = props.en
          let text = data.text
          console.log(data)
          let text_Component = text.map((item, i)=>{
              return (
                  <span style={style.block_text[en]}>{item}</span>
              )
          })

          return (
              <div style={style.block[en]}>
                  <span style={{...style.block_text[en], color:'#A05937'}}>{data.title}</span>
                  {text_Component}
              </div>
          )
      }

}


export default class ModeL_title extends React.Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {};
      }



      render() {

          let {...props} = this.props
          let data = props.data
          let title = data.title
          let text = data.text
          let title_1 = data.title_1
          let en = props.en

          let components_arr = data.block
          components_arr = components_arr.map((item,i)=>{
              return (
                  <ModeL_title_block data={item} en={en}/>
              )
          })



          return (
              <div style={{display:'flex',flexDirection:"column", width:1470*screen_scale_width}}>
                  <span style={style.title_style[en]}>{title}</span>
                  <span style={style.text_style[en]}>{text}</span>
                  <span style={style.title_1_style[en]}>{title_1}</span>
                  <div style={style.block_wrap[en]}>
                      {components_arr}
                  </div>
              </div>
          )
      }

}