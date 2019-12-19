import React from 'react';
import Test_MD from './test_MD'

export default class Test_show_MD extends React.PureComponent{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {};
      }

      render() {
          return (
              <Test_MD style={{height: 'auto',backgroundColor:'#F8F8F8'}} en={0}/>
          )
      }
}