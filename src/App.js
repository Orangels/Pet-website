import React, { Component } from 'react';
import ReactCSSTransitionGroup from 'react-addons-css-transition-group'
import {
    BrowserRouter as Router,
    Route,
    Link,
    Redirect,
    Switch,
    HashRouter
} from 'react-router-dom'

import Model_Zoo from './view/Model/Model_Zoo'

import Document_CN from './view/Doc/Document_CN'

import Document from './view/Doc/Document'
import Home from './view/Home/Home'
import About_Pet from './view/AboutPet/About_Pet'
import Install from './view/Install/Install'
import Test from './view/test_view'
import Test_show_MD from './view/test_show_MD'
import Markdown_ori from './view/Markdown_ori'
import Theme_show from './view/Theme_show'
import HomeDemo from './Home/index'
import Tutorials from './view/Tutorials/Tutorials'
import Contact_us from './view/Contact_us/Contact_us'
import Building from './view/404/Building'

import Demo from './view/test_view'
import Rotate_demo from './view/test_web_demo'

// import Router_main from './view/router_test/router_main'

import { createBrowserHistory } from 'history';

import './App.css'


const history = createBrowserHistory();

// Get the current location.
const location = history.location;

export const NotFound404 = (props) => (
    <div className="whoops-404">
        <h1>没有页面可以匹配</h1>
    </div>
)

const style = {
    wrap:{
        display: 'flex',
        backgroundColor: 'orange',
        flexDirection: 'column',
        justifyContent:'space-between',
        height:1200

    },
    div_1:{
        display: 'flex',
        flex:2,
        backgroundColor: 'red',
        flexDirection:'row',
        justifyContent:'space-between',
        height:200
    },
    div_2:{
        display: 'flex',
        flex:3,
        backgroundColor: 'green',
        flexDirection:'row',
        justifyContent:'space-between',
        height:200
    },
    div_3:{
        display: 'flex',
        flex:4,
        backgroundColor: 'blue',
        flexDirection:'row',
        justifyContent:'space-between',
        height:200
    }
};

const routes = [
    { path: '/',
        component: Home
    },
    {
        path:'/About_Pet',
        component:About_Pet,
    },
    {
      path:'/Install',
      component:Install
    },
    { path: '/Model_Zoo',
        component: Model_Zoo,
    },
    { path: '/Doc',
        component: Document,
    },
    {
      path:'/Tutorials',
      component:Tutorials
    },
    {
      path:'/Contact_us',
      component:Contact_us
    },
    {
        path:'/demo',
        component:Demo
    },
    {
        path:'/demo_2',
        component:Rotate_demo
    },
    {
        path:'/lab',
        component:Building
    },
    // { path: '/debug',
    //     component: Theme_show,
    // },
    { path: '/test',
       component: Markdown_ori,
        // component:Router_main
    },
    // { path: '/tmp',
    //     component: HomeDemo,
    // },
    // {
    //     component:NotFound404
    // }
];


const RouteWithSubRoutes = (route, index) => {
    return (
        <Route path={route.path} exact render={props => (
            // 把自路由向下传递来达到嵌套。
            <route.component {...props} style={{ width: document.body.clientWidth, height: document.body.clientHeight}} />
        )}/>
    )
};



class App extends Component {

  // 构造
    constructor(props) {
      super(props);
      // 初始状态
      this.state = {};
    }

    render() {

    let component_arr = [
        <Route path='/' exact render={props => (
            <Home {...props} style={{ width: document.body.clientWidth, height: document.body.clientHeight}} />
        )}/>,
        <Route path='/About_Pet'  render={props => (
            <About_Pet {...props} style={{ width: document.body.clientWidth, height: document.body.clientHeight}} />
        )}/>,
        <Route path='/Install'  render={props => (
            <Install {...props}  style={{ width: document.body.clientWidth, height: document.body.clientHeight}} />
        )}/>,
        <Route path='/Model_Zoo'  render={props => (
            <Model_Zoo {...props} style={{ width: document.body.clientWidth, height: document.body.clientHeight}} />
        )}/>,
        <Route path='/Doc' render={props => (
            <Document {...props} style={{ width: document.body.clientWidth, height: document.body.clientHeight}} />
        )}/>,
        <Route path='/Tutorials'  render={props => (
            <Tutorials {...props} style={{ width: document.body.clientWidth, height: document.body.clientHeight}} />
        )}/>,
        <Route path='/Contact_us'  render={props => (
            <Contact_us {...props} style={{ width: document.body.clientWidth, height: document.body.clientHeight}} />
        )}/>,
    ]

    let component_arr_map = routes.map((route, i) => {
        let exact = i === 0 ? true : false

        return (

            <Route path={route.path} exact={exact} render={props => (
                // 把自路由向下传递来达到嵌套。
                <route.component {...props} style={{ width: document.body.clientWidth, height: document.body.clientHeight}} />
            )}/>
        )
    })


    return (
        <Router>
            <Switch>
                {component_arr_map}
            </Switch>
        </Router>
            )
    }
}
export default App;
