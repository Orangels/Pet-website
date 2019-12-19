import {screen_scale_width} from "../../common/parameter/parameters";
import React from "react";

<div style={{
    overflow: 'auto',
    height:'90vh', position: 'sticky', top:heart_height,marginLeft:59*screen_scale_width,
    width:541*screen_scale_width,marginTop:(100)*screen_scale_width,
    display:'flex',flexDirection:"column",
}}
>
                            <span style={{fontSize:26*screen_scale_width,color:'#3765A0',letterSpacing:0.15*screen_scale_width,marginTop:20*screen_scale_width}}>
                                安装内容
                            </span>
    <ul style={{marginLeft:25*screen_scale_width,marginTop:15*screen_scale_width}}>
        <li style={style.li_type}><a href={'#environment'} style={style.li_type_text}>Pet 环境</a></li>
        <li style={style.li_type}><a href={'#dependencies'} style={style.li_type_text}>Pet 依赖</a></li>
        <li style={style.li_type}><a href={'#torchvision'} style={style.li_type_text}>Pytorch-1.1, torchvision</a></li>
        <li style={style.li_type}><a href={'#apex'} style={style.li_type_text}>NVIDIA apex</a></li>
        <li style={style.li_type}><a href={'#pycocotools'} style={style.li_type_text}>pycocotools</a></li>
        <li style={style.li_type}><a href={'#install_Pet'} style={style.li_type_text}>安装 Pet</a></li>
    </ul>
    <div style={{width:440*screen_scale_width,height:'auto',border:'1px solid #FF8722',marginTop:20*screen_scale_width,display:'flex',flexDirection:"column",}}>
        {/*// part1*/}
        <span style={{marginTop:19*screen_scale_width,marginLeft:24*screen_scale_width,fontSize:24*screen_scale_width,color: "#484B4D",}}>
                                    安装过程中,具体参见
                                </span>
        <div style={{marginLeft:20*screen_scale_width,marginTop:20*screen_scale_width,fontSize:20*screen_scale_width,color:'#FFFFFF',backgroundColor:'#FF8722',width:320*screen_scale_width,height:44*screen_scale_width,textAlign:'center',cursor:'pointer',borderRadius:4,lineHeight:`${44*screen_scale_width}px`
        }}
             onClick={this._jump.bind(this,"install")}>
            安装及注意事项
        </div>
        {/*// part2*/}
        <span style={{marginTop:19*screen_scale_width,marginLeft:24*screen_scale_width,marginRight:24*screen_scale_width,fontSize:20*screen_scale_width,color: "#484B4D",}}>
                                    安装中遇到问题及讨论意见
                                </span>
        {/*<span style={{marginLeft:24*screen_scale_width,fontSize:24*screen_scale_width,color: "#484B4D",}}>*/}
        {/*    please refer to the*/}
        {/*</span>*/}
        <div style={{marginLeft:24*screen_scale_width,marginTop:20*screen_scale_width,marginBottom:28*screen_scale_width,fontSize:20*screen_scale_width,color:'#FFFFFF',backgroundColor:'#FF8722',width:180*screen_scale_width,height:44*screen_scale_width,textAlign:'center',cursor:'pointer',borderRadius:4,lineHeight:`${44*screen_scale_width}px`
        }}
             onClick={this._jump.bind(this,"issue")}>
            Github issues
        </div>
    </div>
</div>