import React, { Component } from 'react';
import {
    BrowserRouter as Router,
    Route,
    Link,
    withRouter
} from 'react-router-dom'
import PropTypes from 'prop-types';
import Echartstest  from './charts/echarts-splattering'
import TableTest from './table/table'
import Model_Zoo from "../view/Model/Model_Zoo";


// 一些程序员喜欢把路由配置集中到一个地方，要知道路由的配置其实只是普通的数据
// 在把数据映射到组件上这方面，React 非常强大，并且，这里的 <Route> 就是一
// 个普通的组件。

////////////////////////////////////////////////////////////
// 我们先来定义route相关的组件。
const Main = () => <h2>主页</h2>

const Redbull = () => <h2>红牛</h2>

const Snacks = ({ routes }) => (
    <div>
        <h2>小吃</h2>
        <ul>
            <li><Link to="/snacks/spicy">辣条</Link></li>
            <li><Link to="/snacks/chips">薯片</Link></li>
        </ul>

        {routes.map((route, i) => (
            <RouteWithSubRoutes key={i} {...route}/>
        ))}
    </div>
)

const Spicy = () => <h3>辣条</h3>
const Chips = () => <h3>薯片</h3>

////////////////////////////////////////////////////////////
// 这里是路由的配置。
// const routes = [
//     { path: '/redbull',
//         component: Echartstest
//     },
//     { path: '/snacks',
//         component: Snacks,
//         routes: [
//             { path: '/snacks/spicy',
//                 component: Spicy
//             },
//             { path: '/snacks/chips',
//                 component: Chips
//             }
//         ]
//     }
// ]

const routes = [
    { path: '/redbull',
        component: Echartstest
    },
    { path: '/snacks',
        component: TableTest,
    }
]


// 把 <Route> 组件像这样包一层，然后在需要使用 <Route> 的地方使用 <RouteWithSubRoutes>
// 自路由可以加到任意路由组件上。
const RouteWithSubRoutes = (route) => (
    <Route path={route.path} render={props => (
        // 把自路由向下传递来达到嵌套。
        <route.component {...props} routes={route.routes}/>
    )}/>
)

class RouteConfigExample extends Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {};
        this._onclick = this._onclick.bind(this)

      }
    static propTypes = {
        history: PropTypes.object.isRequired
    }
      _onclick(path){
          this.props.history.push(path, { some: 'state' });
      }
      render() {
          return(<Router>
              <div>
                  <ul>
                      <li onClick={this._onclick.bind(this,"/")}><Link to="/">表格</Link></li>
                      <li onClick={this._onclick.bind(this,"/Animation_route")}><Link to="/Animation_route">图表</Link></li>
                  </ul>

                  {routes.map((route, i) => (
                      <RouteWithSubRoutes key={i} {...route}/>
                  ))}
              </div>
          </Router>)
      }

}

export default withRouter(RouteConfigExample);