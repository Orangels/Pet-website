import React from 'react';
import './animation_css.less'
import Texty from 'rc-texty'
import 'rc-texty/assets/index.css'
import {screen_scale_width} from "../common/parameter/parameters";
import About_Pet_content_component from './AboutPet/About_Pet_content_component'
import {About_Pet_data} from './AboutPet/About_Pet_data'

import ai_img_1 from "../asset/HOME_icon/about_Pet/1.png";
import ai_img_2 from "../asset/HOME_icon/about_Pet/2.png";
import ai_img_3 from "../asset/HOME_icon/about_Pet/3.png";
import ai_img_4 from "../asset/HOME_icon/about_Pet/4.png";

export default class Rotate_demo extends React.Component {
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            rotate:0,
            position:[0,90,180,270],
            text:'Ant Motion',
        };
        this.handleClick = this.handleClick.bind(this)
      }

      handleClick(e){
          e.stopPropagation()
          let content_text = ['Ant Motion 0', 'Ant Design 1', 'Pytorch 2', 'Pet 3']
          let dom = e.target
          let next_rotate = Number(dom.getAttribute('position_style'))

          let symbol = next_rotate > 0 ? 1 : -1
          let rotate_tag = 360 * symbol
          let clockwise_rotate = rotate_tag - next_rotate%360
          let anticlockwise_rotate = next_rotate
          next_rotate = Math.abs(clockwise_rotate) < Math.abs(anticlockwise_rotate) ? clockwise_rotate : (anticlockwise_rotate)*-1

          content_text = content_text[Number(dom.innerHTML)]

          const { rotate } =this.state;
          let next_position = this.state.position.map((item,i)=>{
              return (item+next_rotate)%360
          });


          console.log(next_rotate)
          console.log(next_position)
          console.log(rotate+next_rotate)
          // rotate 这里不 % 360 是为了旋转顺序  270 => 0 会出现方向错误
          this.setState({
              rotate: rotate+next_rotate,
              position:next_position,
              text:content_text
          },()=>{
              // console.log(this.state.rotate)
              // console.log(Number(dom.value))
          });

      }


      render() {
          let scale = 0.7
          let wrap_radius = 1350/2*screen_scale_width*scale
          let content_radius = 220/2*screen_scale_width*scale
          let duration = 1.5

          let color_arr = new Array(4)
          let img_arr_size = new Array(4)
          let content_tag = 1


          for (let i =0, length = this.state.position.length ; i < length ; i++){
              color_arr[i] = this.state.position[i] === 0 ? '#FF8722' : '#E4E4E4'
              img_arr_size[i] = this.state.position[i] === 0 ? `200% 100%` : `0% 0%`
              content_tag = this.state.position[i] === 0 ? i+1 : content_tag
          }


          return (
              <div style={{width:wrap_radius*2, height:wrap_radius*2, borderRadius:wrap_radius, border:'1px solid gray', marginTop:100, marginLeft:410*screen_scale_width,
                  position:'relative',
                  overflow:"visible",
                  transform:`rotate(${this.state.rotate}deg)`,
                  transition: `${duration}s all`,
                  display:'flex',justifyContent:"center",alignItems:'center',}}
                   >
                  {/*<Texty style={{*/}
                  {/*    transform:`rotate(${this.state.rotate*-1}deg)`,*/}
                  {/*    transition: `${duration}s all`,*/}
                  {/*}}>{this.state.text}</Texty>*/}
                  <About_Pet_content_component en={0} data={About_Pet_data[`part_${content_tag}`]}
                                               left={0*screen_scale_width}
                                               top={0*screen_scale_width}
                                               style={{
                                                   transform:`rotate(${this.state.rotate*-1}deg)`,
                                                   transition: `${duration}s all`,
                                                   marginBottom:30*screen_scale_width
                                               }}
                                               />
                  <div style={{
                      backgroundColor:color_arr[0],
                      position:'absolute',
                      width:content_radius*2,
                      height:content_radius*2,
                      borderRadius:content_radius,
                      top:-content_radius,
                      left:wrap_radius-content_radius,
                      display:'flex',justifyContent:"center",alignItems:'center',
                      transform:`rotate(${this.state.rotate*-1}deg)`,
                      transition: `${duration}s all`,}}
                      >
                      <div style={{
                          display:'flex',justifyContent:"center",alignItems:'center',
                          width:content_radius*2,height:content_radius*2,borderRadius:content_radius,
                          background:`url(${ai_img_1}) center no-repeat`,
                          backgroundSize:img_arr_size[0],
                          backgroundPosition:'0% 50%'
                        }
                      }
                           position_style={this.state.position[0]}
                           onClick={this.handleClick}
                      >
                          {this.state.position[0] != 0 ? 0 : null}
                      </div>
                  </div>
                  <div style={{position:'absolute',
                      width:content_radius*2,
                      height:content_radius*2,
                      borderRadius:content_radius,
                      backgroundColor:color_arr[1],
                      top:wrap_radius-content_radius,left:wrap_radius*2-content_radius,
                      display:'flex',justifyContent:"center",alignItems:'center',
                      transform:`rotate(${this.state.rotate*-1}deg)`,
                      transition: `${duration}s all`,
                  }}
                        >
                      <div style={{
                          display:'flex',justifyContent:"center",alignItems:'center',
                          width:content_radius*2,height:content_radius*2,borderRadius:content_radius,
                          background:`url(${ai_img_2}) center no-repeat`,
                          backgroundSize:img_arr_size[1],
                          backgroundPosition:'0% 50%'
                      }
                      }
                           position_style={this.state.position[1]}
                           onClick={this.handleClick}
                      >
                          {this.state.position[1] != 0 ? 1 : null}
                      </div>
                  </div>
                  <div style={{position:'absolute', width:content_radius*2,height:content_radius*2,borderRadius:content_radius,backgroundColor:color_arr[2],
                      top:wrap_radius*2-content_radius,left:wrap_radius-content_radius,
                      display:'flex',justifyContent:"center",alignItems:'center',
                      transform:`rotate(${this.state.rotate*-1}deg)`,
                      transition: `${duration}s all`,}}
                        >
                      <div style={{
                          display:'flex',justifyContent:"center",alignItems:'center',
                          width:content_radius*2,height:content_radius*2,borderRadius:content_radius,
                          background:`url(${ai_img_3}) center no-repeat`,
                          backgroundSize:img_arr_size[2],
                          backgroundPosition:'0% 50%'
                      }
                      }
                           position_style={this.state.position[2]}
                           onClick={this.handleClick}
                      >
                          {this.state.position[2] != 0 ? 2 : null}
                      </div>
                  </div>
                  <div style={{position:'absolute', width:content_radius*2,height:content_radius*2,borderRadius:content_radius,backgroundColor:color_arr[3],
                      top:wrap_radius-content_radius,left:-content_radius,
                      display:'flex',justifyContent:"center",alignItems:'center',
                      transform:`rotate(${this.state.rotate*-1}deg)`,
                      transition: `${duration}s all`,}}
                       >
                      <div style={{
                          display:'flex',justifyContent:"center",alignItems:'center',
                          width:content_radius*2,height:content_radius*2,borderRadius:content_radius,
                          background:`url(${ai_img_4}) center no-repeat`,
                          backgroundSize:img_arr_size[3],
                          backgroundPosition:'0% 50%'
                      }
                      }
                           position_style={this.state.position[3]}
                           onClick={this.handleClick}
                      >
                          {this.state.position[3] != 0 ? 3 : null}
                      </div>
                  </div>
              </div>
          )
      }
}