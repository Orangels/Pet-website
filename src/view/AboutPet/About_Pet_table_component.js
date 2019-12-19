import React, { Component } from 'react';
import {model_width, screen_scale_width} from "../../common/parameter/parameters";
import { Tabs, Table } from 'antd';
import workplace from '../../asset/HOME_icon/workplace/workplace.jpg'

import {table_1, table_2,
    table_1_data,table_2_data, table_3_data, table_4_data, table_5_data, table_6_data, table_7_data, table_8_data,} from "./About_Pet_table_data";

const { TabPane } = Tabs;

{/*<span style={{fontSize:58, alignSelf:'center', marginTop:100*screen_scale_width,}} >*/}
{/*        施工中...*/}
{/*    </span>*/}


const Placeholder_Component = ()=>(
    <img src={workplace} style={{width:800*screen_scale_width, height:600*screen_scale_width}}/>
)


const style = {
    part_wrap:{
        display:"flex",
        justifyContent:"center",
        alignItems:"center",
        backgroundColor:'#FFFFFF',
        width:'100%'
    },
    component_wrap:{
        display:"flex",
        width:'100%',
        flexDirection:'row',
        marginTop:40*screen_scale_width
    },
}

export default class About_Pet_table_component extends Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            btn_state:{
                btn_1:true,
                btn_2:true,
                btn_3:true,
                btn_4:true,
            }
        };
        this._changeTab = this._changeTab.bind(this)
      }


    _changeTab(activeKey){
          // console.log(activeKey)
    }

      render() {
          let {...props} = this.props
          let en = props.en
          // if (props.en !== undefined){
          //     en = props.en
          // }

          let title = [
              ['Compare in task support','Compare in speed','Compare in accuracy','Compare in memory cost'],
              ['任务支持','速度比对','精度比对','显存比对']
          ]

          let table_data = [
              [table_1_data,table_2_data,table_3_data,table_4_data],
              [table_5_data,table_6_data,table_7_data,table_8_data]
          ]

          let table = [
              table_1,table_2
          ]

          console.log(`about table en ${en}`)
          console.log(title[en][0])
          console.log(title[en][1])

          return (
              <div style={{...style.part_wrap,...props.wrap_style, marginTop:props.top}} key={'About_Pet_table_component_div_1'}>
                  <div style={{display:"flex", flexDirection:"column", height:'auto', width:model_width,}} key={'About_Pet_table_component_div_2'}>
                      <div style={{display:'flex',flexDirection:'column', marginLeft:100*screen_scale_width,marginBottom:100*screen_scale_width
                      }} key={'About_Pet_table_component_div_3'}>
                          <Tabs type="card" onChange={this._changeTab} key={'About_Pet_table_component_tabs'}>
                              <TabPane tab={title[en][0]} key="about_pet_tab_1" className={'about_pet_tab_1'} >
                                  <Table className={'About_Pet_table'} key={'About_Pet_table_component_tab_1_table'}
                                  style={{width:1720*screen_scale_width,height:612*screen_scale_width}}
                                         columns={table[en]} dataSource={table_data[en][0]} rowClassName={'About_Pet_table_row'}
                                         pagination={false} bordered
                                         />
                              </TabPane>
                              <TabPane tab={title[en][1]} key="about_pet_tab_2" className={"about_pet_tab_2"}
                                       style={{display:'flex',
                                           justifyContent: 'center'}}>
                                  {/*<Table className={'About_Pet_table'}*/}
                                  {/*       style={{width:1720*screen_scale_width,height:612*screen_scale_width}}*/}
                                  {/*       columns={table[en]} dataSource={table_data[en][1]} rowClassName={'About_Pet_table_row'} pagination={false} bordered />*/}
                                  <Placeholder_Component  />
                              </TabPane>
                              <TabPane tab={title[en][2]} key="about_pet_tab_3" className={"about_pet_tab_3"}
                                       style={{display:'flex',
                                           justifyContent: 'center'}}>
                                  {/*<Table className={'About_Pet_table'}*/}
                                  {/*       style={{width:1720*screen_scale_width,height:612*screen_scale_width}}*/}
                                  {/*       columns={table[en]} dataSource={table_data[en][2]} rowClassName={'About_Pet_table_row'} pagination={false} bordered />*/}
                                  <Placeholder_Component />
                              </TabPane>
                              <TabPane tab={title[en][3]} key="about_pet_tab_4" className={"about_pet_tab_4"} style={{display:'flex',
                              justifyContent: 'center'}}>
                                  {/*<Table className={'About_Pet_table'}*/}
                                  {/*       style={{width:1720*screen_scale_width,height:612*screen_scale_width}}*/}
                                  {/*       columns={table[en]} dataSource={table_data[en][3]} rowClassName={'About_Pet_table_row'} pagination={false} bordered />*/}
                                  <Placeholder_Component />
                              </TabPane>
                          </Tabs>
                      </div>
                  </div>
              </div>
          )
      }

}