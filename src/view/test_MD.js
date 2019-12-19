import React, { Component } from 'react';
import { Collapse, Icon } from 'antd/lib/index';
import Md_content_component from '../component/md/common/Md_content_component'
import { test_MD_data  } from './test_MD_data'
import Md_content_component_EN from "../component/md/common/Md_content_component_EN";

class Test_MD extends Component {
    // 构造
    constructor(props) {
        super(props);
        // 初始状态
        this.state = {};
    }

    render() {

        let {...props} = this.props
        let en = props.en
        let data_arr = [test_MD_data, test_MD_data]

        return (
            en === 0 ? (
                <Md_content_component_EN data={data_arr[en]} offsetTop={props.offsetTop} type={props.type}/>
            ) : (
                <Md_content_component data={data_arr[en]} offsetTop={props.offsetTop} type={props.type} />
            )
        )

    }
}

export default Test_MD