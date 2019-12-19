import React, { Component } from 'react';
import { Layout, Menu, Icon } from 'antd/lib/index';
import Echartstest from "./charts/echarts-splattering";
import TableTest from "./table/table";
import Model_Zoo from "../view/Model/Model_Zoo";



class Model_Zoo_content_compent extends Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {};
      }


      render() {
          let { slider_left_width, slider_right_width, charts_data, table_data, detail_table_data, type, table_component} = this.props;
          return (<div style={{padding: 24, background: '#FFFFFFF', textAlign: 'center',height: 'auto'}}>
              <Echartstest style={{ width: document.body.clientWidth-slider_left_width-slider_right_width-20, height: document.body.clientHeight-200, paddingRight:50}} charts_data={ charts_data } type={type}/>
              <TableTest style={{width: document.body.clientWidth-slider_left_width-slider_right_width-20, height: 'auto', paddingRight:50, marginTop:20}} table_data={ table_data } detail_table_data={detail_table_data} type={type} content_style={{backgroundColor:"#E4F4F6", marginTop:10, textAlign:'left', padding:15, width: document.body.clientWidth-slider_left_width-slider_right_width-50}} table_component={table_component}/>
          </div>)
      }
}

export default Model_Zoo_content_compent;
