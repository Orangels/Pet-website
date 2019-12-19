import React, { Component } from 'react';
import Help from "./Help";
import { Collapse, Icon } from 'antd/lib/index';
import { Table, Button } from 'antd/lib/index';
import Md_content_component from '../common/Md_content_component'
import '../style/opt.less'

import {opt_data, opt_data_CN} from '../../../common/data/component_data'
import Md_content_component_EN from "../common/Md_content_component_EN";

const Panel = Collapse.Panel;



class Optimization_iteration extends Component {
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {};
      }

    render() {
        let {...props} = this.props
        let en = props.en || 0
        let data_arr = [opt_data, opt_data_CN]

        return (
            en === 0 ? (
                <Md_content_component_EN data={data_arr[en]} offsetTop={props.offsetTop} type={props.type}/>
            ) : (
                <Md_content_component data={data_arr[en]} offsetTop={props.offsetTop} type={props.type} />
            )
        )

    }
}

export default Optimization_iteration