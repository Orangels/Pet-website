import React, { Component } from 'react';
import {screen_scale_width} from "../../common/parameter/parameters";
import { Progress } from 'antd'

const style = {
    wrap_div:[
        {
            display:'flex',
            flexDirection:"column",
            backgroundColor:'#F7F7F7',
            paddingTop:33*screen_scale_width,
            paddingLeft:31*screen_scale_width,
            height:192*screen_scale_width,
            width:800*screen_scale_width,
            overflowX:'auto',
            whiteSpace:"nowrap",
            // borderBottom:'1px solid #A05937',
            marginLeft: 50*screen_scale_width,
            fontSize:21*screen_scale_width,
            position:"relative",
        },
        {
            display:'flex',
            flexDirection:"column",
            backgroundColor:'#F7F7F7',
            paddingTop:33*screen_scale_width,
            paddingLeft:31*screen_scale_width,
            height:192*screen_scale_width,
            width:800*screen_scale_width,
            overflowX:'auto',
            whiteSpace:"nowrap",
            // borderBottom:'1px solid #A05937',
            marginLeft: 20*screen_scale_width,
            fontSize:21*screen_scale_width,
            position:"relative",
        }
    ],
    text:[
        {
            width:850*screen_scale_width,
            fontSize:26*screen_scale_width,
            fontFamily:'freight-sans, sans-serif',
            alignSelf:'center',
            letterSpacing:'1px'
        },
        {
            width:850*screen_scale_width,
            letterSpacing:1,
            fontSize:20*screen_scale_width,
            fontFamily:'freight-sans, sans-serif',
            alignSelf:'center'

        }
    ]
};


class Code_component extends Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            progress:0
        };
      }

    componentWillReceiveProps(nextProps, nextContext) {
        let progress = nextProps.progress || 0
        this.setState({
            progress:progress
        })
    }

      render() {
          let {en} = this.props
          return (
              <div style={style.wrap_div[en]}>
                  <span style={{marginBottom:10*screen_scale_width}}>
                    <span style={{color:'#2AE9E9'}}>
                        python&ensp;
                    </span>
                    <span style={{color:'#F59727'}}>
                      -m&ensp;
                    </span>
                    <span>
                      torch.distributed.launch&ensp;
                    </span>
                    <span style={{color:'#F59727'}}>
                      --nproc_per_node&ensp;
                    </span>
                    <span style={{color:'#EB55BD'}}>
                      =&ensp;
                    </span>
                      <span>
                          8&ensp;\
                      </span>
                  </span>
                  <span style={{marginBottom:10*screen_scale_width}}>
                      tools/rnn/train_net.py&ensp;\
                  </span>
                  <span style={{marginBottom:10*screen_scale_width}}>
                      <span style={{color:'#F59727'}}>
                          --cfg&ensp;
                      </span>
                      <span>
                          cfgs/rnn/mscoco/e2e_faster_rcnn_R-50-FPN_1x.yaml
                      </span>
                  </span>
                  <Progress percent={this.state.progress} showInfo={false} style={{
                      position:"absolute",
                      bottom:-10*screen_scale_width,
                      left:0,
                  }}
                            strokeWidth={2}
                            strokeColor={'#A05937'}
                  className={'Home_code_progress'}/>
              </div>
          )
      }

}


export default class Home_use_pet extends Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            progress:0
        };
      }

    componentWillReceiveProps(nextProps, nextContext) {
          let progress = nextProps.progress || 0
          this.setState({
              progress:progress
          })
    }

      render() {
          let {...props} = this.props
          let { data } = props
          let top = props.top || 40*screen_scale_width
          let en = props.en || 0

          return (
              <div style={{display:'flex',flexDirection:'column',}}>
                  <div style={{display:'flex',flexDirection:'column', marginTop:top,marginLeft:150*screen_scale_width,
                      marginRight:150*screen_scale_width,marginBottom:47*screen_scale_width,
                  }}>
                      <div style={{width:72*screen_scale_width, height:2, backgroundColor:'#F98B34'}}></div>
                      <span style={{fontSize:44*screen_scale_width, color:'black', marginTop:5*screen_scale_width}}>{data['title']}</span>
                      <div style={{display:'flex', marginTop:23*screen_scale_width}}>
                          {/*<div style={style.text[en]}>*/}
                          {/*    {data['text']}*/}
                          {/*</div>*/}
                          <div style={{display:'flex',flexDirection:'column'}}>
                              {data['text'].map((item,i)=>{
                                  return (
                                      <span style={style.text[en]}>
                                      {item}
                                  </span>
                                  )
                              })}
                          </div>
                          <Code_component en={en} progress={this.state.progress}/>
                      </div>
                  </div>
              </div>
          )
      }

}