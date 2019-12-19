import React, { Component } from 'react';
import ReactCSSTransitionGroup from 'react-addons-css-transition-group'
import {
    BrowserRouter as Router,
    Route,
    Link,
    Redirect
} from 'react-router-dom'
import Echartstest from "./charts/echarts-splattering";
import TableTest from "./table/table";


const routes = [
    { path: '/Echart_1',
        component: Echartstest
    },
    { path: '/Echart_2',
        component: Echartstest,
    },
    { path: '/Form_1',
        component: TableTest
    },
    { path: '/Form_2',
        component: TableTest,
    }
]


const RouteWithSubRoutes = (route) => (
    <Route path={route.path} render={props => (
        // 把自路由向下传递来达到嵌套。
        <route.component {...props} style={{ width: document.body.clientWidth, height: document.body.clientHeight}} routes={route.routes}/>
    )}/>
)


class AnimationExample extends Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {};
      }

      render() {
          return(<Router>
              <Route render={({ location }) => (
                  <div style={styles.fill}>
                      <Route exact path="/" render={() => (
                          <Redirect to="/Echart_1"/>
                      )}/>

                      <ul style={styles.nav}>
                          <NavLink to="/Echart_1">Echart_1</NavLink>
                          <NavLink to="/Echart_2">Echart_2</NavLink>
                          <NavLink to="/Form_1">Form_1</NavLink>
                          <NavLink to="Form_2">Form_2</NavLink>
                      </ul>

                      <div style={styles.content}>
                          <ReactCSSTransitionGroup
                              transitionName="fade"
                              transitionEnterTimeout={300}
                              transitionLeaveTimeout={300}
                          >
                              {/*
                这里和使用 ReactCSSTransitionGroup 没有区别，
                唯一需要注意的是要把你的地址（location）传入
                「Route」里使它可以在动画切换的时候匹配之前的
                地址。
            */}
                              {routes.map((route, i) => (
                                  <RouteWithSubRoutes key={i} {...route}/>
                              ))}
                          </ReactCSSTransitionGroup>
                      </div>
                  </div>
              )}/>
          </Router>)
      }

}


const NavLink = (props) => (
    <li style={styles.navItem}>
        <Link {...props} style={{ color: 'inherit' }}/>
    </li>
)

const HSL = ({ match: { params } }) => (
    <div style={{
        ...styles.fill,
        ...styles.hsl,
        background: `hsl(${params.h}, ${params.s}%, ${params.l}%)`
    }}>hsl({params.h}, {params.s}%, {params.l}%)</div>
)

const styles = {}

styles.fill = {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0
}

styles.content = {
    ...styles.fill,
    top: '40px',
    textAlign: 'center'
}

styles.nav = {
    padding: 0,
    margin: 0,
    position: 'absolute',
    top: 0,
    height: '40px',
    width: '100%',
    display: 'flex'
}

styles.navItem = {
    textAlign: 'center',
    flex: 1,
    listStyleType: 'none',
    padding: '10px'
}

styles.hsl  = {
    ...styles.fill,
    color: 'white',
    paddingTop: '20px',
    fontSize: '30px'
}

export default AnimationExample