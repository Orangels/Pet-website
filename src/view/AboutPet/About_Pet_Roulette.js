import React from 'react';

import Texty from 'rc-texty'
import 'rc-texty/assets/index.css'
import {screen_scale_width} from "../../common/parameter/parameters";
import About_Pet_content_component from './About_Pet_content_component'
import { About_Pet_data, About_Pet_data_CN } from './About_Pet_data'

import ai_img_1 from "../../asset/HOME_icon/about_Pet/functions.png";
import ai_img_2 from "../../asset/HOME_icon/about_Pet/features.png";
import ai_img_3 from "../../asset/HOME_icon/about_Pet/comparious.png";
import ai_img_4 from "../../asset/HOME_icon/about_Pet/expand.png";

export default class About_Pet_Roulette extends React.Component {
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
        this._onMouseEnter = this._onMouseEnter.bind(this)
        this._onMouseLeave = this._onMouseLeave.bind(this)

    }

    _onMouseEnter(e){
        console.log(`mouse_enter`)
        this.handleClick(e)
        // let dom = e.target
        // let parnentDom = dom.parentNode
        // let next_rotate = Number(dom.getAttribute('position_style'))
        // if (next_rotate !== 0){
        //     let radius = 250/2*screen_scale_width*0.7
        //     parnentDom.style.width = `${radius*2}px`
        //     parnentDom.style.height = `${radius*2}px`
        //     parnentDom.style.borderRadius = `${radius}px`
        //
        //     // console.log(parnentDom)
        //     // console.log(dom)
        // }
    }

    _onMouseLeave(e){
        console.log(`mouse_leave`)

        // let dom = e.target
        // let parnentDom = dom.parentNode
        // let next_rotate = Number(dom.getAttribute('position_style'))
        //
        //
        // let radius = 220/2*screen_scale_width*0.7
        // parnentDom.style.width = `${radius*2}px`
        // parnentDom.style.height = `${radius*2}px`
        // parnentDom.style.borderRadius = `${radius}px`

    }

    handleClick(e){
        // e.stopPropagation()
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


        // console.log(next_rotate)
        // console.log(next_position)
        // console.log(rotate+next_rotate)
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

        let en = this.props.en

        let scale = 0.7
        // let wrap_radius = 1350/2*screen_scale_width*scale
        // let content_radius = 220/2*screen_scale_width*scale
        let wrap_radius = 1550/2*screen_scale_width*scale
        let content_radius = 350/2*screen_scale_width*scale
        let duration = 1.5

        let color_arr = new Array(4)
        let img_arr_size = new Array(4)
        let content_tag = 1
        let content_arr = [About_Pet_data, About_Pet_data_CN]

        for (let i =0, length = this.state.position.length ; i < length ; i++){
            color_arr[i] = this.state.position[i] === 0 ? '#FF8722' : '#E4E4E4'
            img_arr_size[i] = this.state.position[i] === 0 ? `100% 100%` : `0% 0%`
            content_tag = this.state.position[i] === 0 ? i+1 : content_tag
        }

        let component_content_data = content_arr[en][`part_${content_tag}`]


        return (
            <div style={{width:wrap_radius*2, height:wrap_radius*2, borderRadius:wrap_radius, border:'1px solid #D3D6DD',
                // marginTop:100,
                marginTop:content_radius,
                marginBottom:content_radius-50,
                // marginLeft:410*screen_scale_width,
                alignSelf:'center',
                position:'relative',
                overflow:"visible",
                transform:`rotate(${this.state.rotate}deg)`,
                transition: `${duration}s all`,
                display:'flex',justifyContent:"center",alignItems:'center',
            }}
            >
                <About_Pet_content_component en={en} data={component_content_data}
                                             // left={0*screen_scale_width}
                                             left={0*screen_scale_width}
                                             top={0*screen_scale_width}
                                             style={{
                                                 transform:`rotate(${this.state.rotate*-1}deg)`,
                                                 transition: `${duration}s all`,
                                                 // marginBottom:30*screen_scale_width,
                                                 // marginLeft:20*screen_scale_width
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
                    transition: `${duration*0.3}s all`,
                }}
                >
                    <div style={{
                        display:'flex',justifyContent:"center",alignItems:'center',
                        width:content_radius*2,height:content_radius*2,borderRadius:content_radius,
                        background:`url(${ai_img_1}) center no-repeat`,
                        backgroundSize:img_arr_size[0],
                        // backgroundPosition:'0% 50%'
                        fontSize:34*screen_scale_width
                    }}
                         position_style={this.state.position[0]}
                         onClick={this.handleClick}
                         onMouseEnter={this._onMouseEnter}
                         onMouseLeave={this._onMouseLeave}
                    >
                        {this.state.position[0] != 0 ? content_arr[en]['part_1']['title'] : null}
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
                    transition: `${duration*0.3}s all`,
                }}
                >
                    <div style={{
                        display:'flex',justifyContent:"center",alignItems:'center',
                        width:content_radius*2,height:content_radius*2,borderRadius:content_radius,
                        background:`url(${ai_img_2}) center no-repeat`,
                        backgroundSize:img_arr_size[1],
                        fontSize:34*screen_scale_width
                    }
                    }
                         position_style={this.state.position[1]}
                         onClick={this.handleClick}
                         onMouseEnter={this._onMouseEnter}
                         onMouseLeave={this._onMouseLeave}
                    >
                        {this.state.position[1] != 0 ? content_arr[en]['part_2']['title'] : null}
                    </div>
                </div>
                <div style={{position:'absolute', width:content_radius*2,height:content_radius*2,borderRadius:content_radius,backgroundColor:color_arr[2],
                    top:wrap_radius*2-content_radius,left:wrap_radius-content_radius,
                    display:'flex',justifyContent:"center",alignItems:'center',
                    transform:`rotate(${this.state.rotate*-1}deg)`,
                    transition: `${duration*0.3}s all`,}}
                >
                    <div style={{
                        display:'flex',justifyContent:"center",alignItems:'center',
                        width:content_radius*2,height:content_radius*2,borderRadius:content_radius,
                        background:`url(${ai_img_3}) center no-repeat`,
                        backgroundSize:img_arr_size[2],
                        fontSize:34*screen_scale_width
                    }
                    }
                         position_style={this.state.position[2]}
                         onClick={this.handleClick}
                         onMouseEnter={this._onMouseEnter}
                         onMouseLeave={this._onMouseLeave}
                    >
                        {this.state.position[2] != 0 ? content_arr[en]['part_3']['title'] : null}
                    </div>
                </div>
                <div style={{position:'absolute', width:content_radius*2,height:content_radius*2,borderRadius:content_radius,backgroundColor:color_arr[3],
                    top:wrap_radius-content_radius,left:-content_radius,
                    display:'flex',justifyContent:"center",alignItems:'center',
                    transform:`rotate(${this.state.rotate*-1}deg)`,
                    transition: `${duration*0.3}s all`,}}
                >
                    <div style={{
                        display:'flex',justifyContent:"center",alignItems:'center',
                        width:content_radius*2,height:content_radius*2,borderRadius:content_radius,
                        background:`url(${ai_img_4}) center no-repeat`,
                        backgroundSize:img_arr_size[3],
                        fontSize:34*screen_scale_width
                    }
                    }
                         position_style={this.state.position[3]}
                         onClick={this.handleClick}
                         onMouseEnter={this._onMouseEnter}
                         onMouseLeave={this._onMouseLeave}
                    >
                        {this.state.position[3] != 0 ? content_arr[en]['part_4']['title'] : null}
                    </div>
                </div>
            </div>
        )
    }
}