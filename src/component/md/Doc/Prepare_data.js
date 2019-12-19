import React, { Component } from 'react';
import { Collapse, Icon } from 'antd/lib/index';
import Md_content_component from '../common/Md_content_component'
import Md_content_component_EN from '../common/Md_content_component_EN'
import {
    prepare_data,
    prepare_data_CN,
} from '../../../common/data/component_data'
import '../style/prepare_data.less'


const Panel = Collapse.Panel;



class Prepare_data extends Component {
    // 构造
    constructor(props) {
        super(props);
        // 初始状态
        this.state = {};
    }

    render() {
        let {...props} = this.props
        let en = props.en
        let data_arr = [prepare_data, prepare_data_CN]


        return (
            en === 0 ? (
                <Md_content_component_EN data={data_arr[en]} offsetTop={props.offsetTop} type={props.type}/>
            ) : (
                <Md_content_component data={data_arr[en]} offsetTop={props.offsetTop} type={props.type} />
            )
        )

    }
}

export default Prepare_data