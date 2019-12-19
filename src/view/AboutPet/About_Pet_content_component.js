import React, { Component } from 'react';
import {model_width, screen_scale_height, screen_scale_width} from "../../common/parameter/parameters";
import Texty from 'rc-texty'

const style = {
    part_wrap:{
        display:"flex",
        justifyContent:"center",
        alignItems:"center",
        backgroundColor:'#FFFFFF',
        // width:'auto'
    },
    component_wrap:{
        display:"flex",
        // width:'auto',
        // flexDirection:'row',
        flexDirection:'column',
        marginTop:40*screen_scale_width
    },
    component:[
        {
            display:"flex",
            // width:542*screen_scale_width,
            width:726*screen_scale_width,
            flexDirection:"column",
        },
        {
            display:"flex",
            // width:542*screen_scale_width,
            width:680*screen_scale_width,
            // width:726*screen_scale_width,
            flexDirection:"column",
        }
    ],
    component_title:[
        {
            color: "#2B2D2E",
            letterSpacing:'0.15px',
            fontSize: 28*screen_scale_width,
            marginBottom: 20*screen_scale_width,
            transition: `.5s all`,
        },
        {
            color: "#2B2D2E",
            letterSpacing:'0.15px',
            fontSize: 26*screen_scale_width,
            marginBottom: 20*screen_scale_width,
            transition: `.5s all`,
        },
    ],
    component_text:[
        {
            color: "#484B4D",
            letterSpacing:1/0.75*screen_scale_width,
            // fontSize: 26*screen_scale_width,
            fontSize: 16/0.75*screen_scale_width,
            // height:120*screen_scale_width,
            // textAlign:'justify'
            // wordBreak:"break-all",
            // hyphens:'auto',
            wordBreak:'keep-all ',
            transition: `.5s all`,
        },{
            color: "#484B4D",
            letterSpacing:0.2*screen_scale_width,
            fontSize: 20*screen_scale_width,
            height:58.67*screen_scale_width,
            // wordBreak:"break-all",
            // hyphens:'auto',
            transition: `.5s all`,
        }
    ],
    img:{
        width:840*screen_scale_width,
        height:500*screen_scale_height,
        // transition: '.3s all',
    }
}


class About_part_component extends Component{
    constructor(props) {
        super(props);

    }
    render() {
        let {...props} = this.props
        let { data } = props
        let title = data.title
        let content = data.children
        let header = props.header
        let index = props.index
        let en = props.en

        content = content.map((item,i) => {
            return (
                <span className={`About_part_component_${header}_${index}_text_${i}`} key={`About_part_component_text_span_${i}`}style={style.component_text[en]}>{item}</span>
            )
        });

        return (
            <div  style={{...style.component[en],marginLeft:props.marginLeft, marginBottom:40*screen_scale_width}}>
                <span style={style.component_title[en]}>{title}</span>
                {content}
            </div>
        )
    }


}


export default class About_Pet_content_component extends Component{
    constructor(props) {
        super(props);
        this._component_img_onMouseEnter = this._component_img_onMouseEnter.bind(this)
        this._component_img_onMouseLeave = this._component_img_onMouseLeave.bind(this)

    }

    _component_img_onMouseEnter(e){
        let dom = e.target
        dom.style.backgroundSize = '150% 150%'
    }

    _component_img_onMouseLeave(e){
        let dom = e.target
        dom.style.backgroundSize = '100% 100%'
    }

    render() {
        let {...props} = this.props
        let { data } = props;
        let components_arr = data.children
        let bottom = props.Bottom || 0

        let en = props.en

        //  实际 scale 是 1-scale
        let scale = 0.25
        let title_size = [58,44]

        components_arr = components_arr.map((item,i)=>{
            let left = 51*screen_scale_width
            return (
                <About_part_component key={`About_part_component_dev_${i}`} data={item} marginLeft={left} header={data.title} index={i} en={en}/>
            )
        })

        return (
                <div style={{...{display:'flex',flexDirection:'column', marginTop:props.top,width:840*screen_scale_width, marginLeft:props.left,}, ...props.style }}
                className={'About_Pet_count_component'}>
                    <div style={{alignSelf:'center', color:'#FF8722', fontSize:34*screen_scale_width}}>
                            {data.title}
                    </div>
                    <div style={{...style.component_wrap,marginBottom:bottom*(1-en*scale)}}>
                        {components_arr}
                    </div>
                </div>


        )

    }


}