import React, { Component } from 'react';
import { Layout, Menu, Icon, Tooltip } from 'antd';
import {system_param, screen_width,model_width, model_height,screen_scale_width,screen_scale_height} from '../../../common/parameter/parameters'

const { SubMenu } = Menu;


const style = {
    subMenu_item:[
        {
            fontWeight:500,
            fontSize:16/0.75*screen_scale_width,
            // fontSize:18*screen_scale_width,
            // textColor:"#8E8E8E"
            color: '#262626',
            // backgroundColor:'#FFFFFF',
            backgroundColor:'#F7F7F7',
            // marginTop:20*screen_scale_width,
            fontFamily:'freight-sans, sans-serif',
        },
        {
            fontWeight:500,
            fontSize:14/0.75*screen_scale_width,
            // fontSize:18*screen_scale_width,
            // textColor:"#8E8E8E"
            color: '#262626',
            // backgroundColor:'#FFFFFF',
            backgroundColor:'#F7F7F7',
            // marginTop:20*screen_scale_width,
            fontFamily:'PingFangSC-Regular',
        }
    ],
    left_item:[
        {
            fontWeight:500,
            // fontSize:22*screen_scale_width,
            fontSize:14/0.75*screen_scale_width,
            transition: '0.25s all',
            marginTop:20*screen_scale_width,
            fontFamily:'freight-sans, sans-serif',
            lineHeight:1.5,
            // backgroundColor:'#F7F7F7',
            paddingLeft:'100px'
        },
        {
            fontWeight:500,
            fontSize:12/0.75*screen_scale_width,
            transition: '0.25s all',
            marginTop:20*screen_scale_width,
            fontFamily:'PingFangSC-Regular',
            lineHeight:1.5,
            // backgroundColor:'#F7F7F7',
            paddingLeft:'100px'
        }
    ],
    left_item_1:{
        fontWeight:500,
        fontSize:12,
        position:'relative',
        width:600
    },
    right_item:{
        fontWeight:500,
        fontSize:15,
    },
}


export default class Sider_Menu extends React.Component{
    constructor(props) {
        super(props);
        this._onMouseEnter = this._onMouseEnter.bind(this)
        this._onMouseLeave = this._onMouseLeave.bind(this)
        this._textSize = this._textSize.bind(this)
    }

    _textSize(fontSize, text) {
        var span = document.createElement("span");
        var result = {};
        result.width = span.offsetWidth;
        result.height = span.offsetWidth;
        span.style.visibility = "hidden";
        span.style.fontSize=fontSize
        document.body.appendChild(span);
        if (typeof span.textContent != "undefined")
            span.textContent = text;
        else span.innerText = text;
        result.width = span.offsetWidth - result.width;
        result.height = span.offsetHeight - result.height;
        span.parentNode.removeChild(span);
        return result;
    }

    _onMouseEnter(eventKey){
        let dom = eventKey.domEvent.target
        let text = dom.innerHTML


        let text_length = this._textSize(12,text)
        // dom.style.width = `${500*screen_scale_width}px`
        // dom.style.width = `${text_length.width+40*screen_scale_width}px`
        // dom.style.backgroundColor = '#FCE8D5'
        // dom.style.transition= '0.25s all'
    }

    _onMouseLeave(eventKey){
        let dom = eventKey.domEvent.target
        // dom.style.width = ''
        // dom.style.backgroundColor = '#FFFFFF'
        // dom.style.transition= '0.25s all'

    }

    getChildrenToRender(data,key,nest, en) {

        return data.map((item,i) => {
            if (typeof(item)=="object"){
                for (let item_key in item){

                    return (
                        <SubMenu key={`${key}_${i}`} style={{backgroundColor:'#F7F7F7',}}
                                 title={
                                     <span>
                                         <span style={style.subMenu_item[en]}>{item_key}</span>
                                     </span>
                                 }
                        >
                            {this.getChildrenToRender(item[item_key],`${key}_${(i+1)*10}`,nest+1, en)}
                        </SubMenu>
                    )
                }
            }else {

                // let submenu_style = nest===0 ? {...style.subMenu_item[en], marginBottom:10*screen_scale_width} : style.left_item[en]
                let submenu_style = nest===0 ? {...style.subMenu_item[en]} : style.left_item[en]

                return (
                    item.length > 18 ? (
                        <Menu.Item className={'Sider_Menu_test'}
                            // onMouseEnter={this._onMouseEnter}
                            // onMouseLeave={this._onMouseLeave}
                            key={`${key}_${i}`} style={submenu_style}>
                            {item}
                        </Menu.Item>
                    ) : (
                        <Menu.Item key={`${key}_${i}`} style={submenu_style}
                                   className={'Sider_Menu_test'}
                                   // onMouseEnter={this._onMouseEnter}
                                   // onMouseLeave={this._onMouseLeave}
                        >
                            {item}
                        </Menu.Item>
                    )
                )
            }
        });

    }


    render() {
        let {...props} = this.props
        const { data } = props;
        let en = props.en
        const childrenToRender = this.getChildrenToRender(data.dataSource,data.key,0, en);

        return(
            <Menu theme="light" mode="inline"
                  onClick={props.onClick}
                  onOpenChange={props.onOpenChange}
                  style={{paddingBottom:20,}}
                  openKeys={props.openKeys}
                  selectedKeys={props.selectedKeys}
                  style={{backgroundColor:'#F7F7F7'}}
            >
                {childrenToRender}
            </Menu>
        )
    }

}

