import React from 'react';
import QueueAnim from 'rc-queue-anim';
import { Row, Col } from 'antd';
import OverPack from 'rc-scroll-anim/lib/ScrollOverPack';
import { getChildrenToRender } from './utils';


//  1397
let screen_width = document.documentElement.clientWidth;
// 798
let screen_height = document.documentElement.clientHeight;
let screen_scale_width = screen_width/1920;
let screen_scale_height = screen_height/1080;

let a = require('../asset/HOME_icon/1x/Contrast.png');

const test_com = ()=>(
    <Col>
        <div>
            <img src={require(a)} style={{width:81*screen_scale_width, height:95*screen_scale_height, alignSelf:'center', marginBottom:20}}/>
        </div>
        <h3 style={{fontSize:18, fontWeight:500,color:'black'}}>Contrast</h3>
        <div style={{fontSize:14,width:400}}>Compared with other deep learning frameworks, Pet has own unique advantages in speed and accuracy.</div>
    </Col>
)

let prop_test = {
    name: 'block0',
    className: 'block',
    // md: 8,
    // xs: 24,
    children: {
        icon: {
            className: 'icon',
            children:
                'https://zos.alipayobjects.com/rmsportal/WBnVOjtIlGWbzyQivuyq.png',
        },
        title: { className: 'content0-title', children: '一站式业务接入' },
        content: { children: '支付、结算、核算接入产品效率翻四倍' },
    },
}

const test_arr = [<Col key={"1"} {...prop_test}>
    <div>
        <img src={a} style={{width:81*screen_scale_width, height:95*screen_scale_height, alignSelf:'center', marginBottom:20}}/>
    </div>
    <h3 style={{fontSize:18, fontWeight:500,color:'black'}}>Contrast</h3>
    <div style={{fontSize:14,width:400}}>Compared with other deep learning frameworks, Pet has own unique advantages in speed and accuracy.</div>
</Col>,<Col key={"2"} {...prop_test}>
    <div>
        <img src={a} style={{width:81*screen_scale_width, height:95*screen_scale_height, alignSelf:'center', marginBottom:20}}/>
    </div>
    <h3 style={{fontSize:18, fontWeight:500,color:'black'}}>Contrast</h3>
    <div style={{fontSize:14,width:400}}>Compared with other deep learning frameworks, Pet has own unique advantages in speed and accuracy.</div>
</Col>];


class Content extends React.PureComponent {
  getBlockChildren = (data) =>
    data.map((item, i) => {
      const children = item.children;
      return (
        <Col key={i.toString()} {...item}>
            <div>
                <img src={require('../asset/HOME_icon/1x/Contrast.png')} style={{width:81*screen_scale_width, height:95*screen_scale_height, alignSelf:'center', marginBottom:20}}/>
            </div>
            <h3 style={{fontSize:18, fontWeight:500,color:'black'}}>Contrast</h3>
            <div style={{fontSize:14,width:400}}>Compared with other deep learning frameworks, Pet has own unique advantages in speed and accuracy.</div>
        </Col>
      );
    });



  render() {
    const { ...props } = this.props;
    const { dataSource } = props;
    delete props.dataSource;
    delete props.isMobile;
    const listChildren = this.getBlockChildren(dataSource.block.children);
    return (
      <div {...props} {...dataSource.wrapper}>
        <div {...dataSource.page}>
          <div {...dataSource.titleWrapper}>
            {dataSource.titleWrapper.children.map(getChildrenToRender)}
          </div>
          <OverPack {...dataSource.OverPack}>
            <QueueAnim
              type="bottom"
              key="block"
              leaveReverse
              {...dataSource.block}
            >
                {test_arr}
            </QueueAnim>
          </OverPack>
        </div>
      </div>
    );
  }
}

export default Content;
