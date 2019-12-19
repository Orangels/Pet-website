import React from 'react'
import {screen_scale_width} from "../../common/parameter/parameters";


export const Template_li = (props) => (
    <ul className={'Template_li'} style={{marginTop:30*screen_scale_width,marginLeft:25*screen_scale_width}}>
        <li style={{listStyle:'disc outside',}}>
            {props.children}
        </li>
    </ul>

)