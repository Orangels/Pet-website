import React from 'react'
import { HashRouter, Route, Switch, Redirect, BrowserRouter as Router, } from 'react-router-dom'

// 引入展示组件
import { About, Contact, Home, Products, Events, NotFound404, Details, } from './pages';


let routes = [
    {
        path:'/',
        component:Home
    },
    {
        path:'/about',
        component:About
    },
    {
        path:'/contact',
        component:Contact
    },
    {
        path:'/products',
        component:Products
    },
    {
        path:'/events',
        component:Events
    },
    // {
    //     component:NotFound404
    // },

]


const RouteWithSubRoutes = (route) => {
    console.log(route.path)
    return (
        <Route path={route.path} exact component={route.component}/>
    )
};

// const RouteWithSubRoutes = (route) => (
    {/*<Route path={route.path} component={route.component} />*/}
// )





    function Router_main() {
    console.log(routes)
    let component_arr = routes.map((route, i) => (
            <RouteWithSubRoutes key={i} {...route} />
        ))

  return (
      <HashRouter>
        <div>
          <Switch>
            <Route path='/' exact component={Home}/>
            <Route path='/about'  component={About}/>
            <Route path='/contact'  component={Contact}/>
            <Route path='/products'  component={Products}/>
            <Route path='/events'  component={Events}/>
            <Redirect from='/history'  to='about/history'></Redirect>
            <Route path='/details/:type'  component={Details}></Route>
            <Route component={NotFound404}/>
              {/*{component_arr}*/}
          </Switch>
        </div>
      </HashRouter>
  )
}

export default Router_main