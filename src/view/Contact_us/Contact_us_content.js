import React, { Component } from 'react';
import {screen_scale_width} from "../../common/parameter/parameters";
import tag from '../../asset/HOME_icon/contact_us/tag.png'


const style = {
    img:[
        {
            width:309*0.8*screen_scale_width,
            // height:216*0.8*screen_scale_width,
            alignSelf:'center'
        },
        {
            width:309*0.8*screen_scale_width,
            // height:216*0.8*screen_scale_width,
            alignSelf:'center'
        }
    ],
    text:[
        {
            fontSize:22*screen_scale_width,
            color:'#484B4D',
            marginLeft:72*screen_scale_width,
            fontWeight: 500,
            textAlign:'justify',
            // lineHeight:'40px'
            alignSelf:'center'
        },
        {
            fontSize:20*screen_scale_width,
            color:'#484B4D',
            marginLeft:72*screen_scale_width,
            alignSelf:'center'
            // lineHeight:'40px'
        }
    ],
    introduce:[
        {
            fontSize:29*screen_scale_width,
            color:'#2B2D2E',
            marginTop:35*screen_scale_width,
            height:33
        },
        {
            fontSize:27*screen_scale_width,
            color:'#2B2D2E',
            marginTop:35*screen_scale_width,
            height:33
        }
    ]
}


export default class Contact_us_content extends Component{
    constructor(props) {
        super(props);

    }

    render() {

        let {...props} = this.props
        let { data } = props
        let children = data['children']
        let introduce = data['introduce']
        let top = props.top || 40*screen_scale_width
        let en = props.en || 0

        return (
            <div style={{display:'flex',flexDirection:'column',}}>
                <div style={{display:'flex',flexDirection:'column', marginTop:top,marginLeft:150*screen_scale_width,
                    marginRight:150*screen_scale_width,marginBottom:47*screen_scale_width,
                }}>
                    <div style={{width:72*screen_scale_width, height:2, backgroundColor:'#F98B34'}}></div>
                    <span style={{fontSize:44*screen_scale_width, color:'black', marginTop:5*screen_scale_width}}>{data['title']}</span>
                    <span style={style.introduce[en]}>
                        {introduce}
                    </span>
                    <div style={{width:'100%',height:320*screen_scale_width,
                        // backgroundColor:'#F7F7F7',
                        padding:`${64*screen_scale_width}px ${138*screen_scale_width}px ${40*screen_scale_width}px ${0*screen_scale_width}px`,
                        display:'flex', flexDirection:'row'}}>
                        <img src={tag} style={style.img[en]}/>
                        <span style={style.text[en]}>
                        {children}
                    </span>
                    </div>
                </div>

            </div>
        )
    }

}