import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import Basis_router from './component/basic-router'
import Nested_router from './component/Nested-router'
import Echartstest  from './component/charts/echarts-splattering'
import TableTest from './component/table/table'
import RouteConfigExample from './component/RouteConfigExample'
import AnimationExample from './component/AnimationExample'
import Model_Zoo from './view/Model/Model_Zoo'

import * as serviceWorker from './serviceWorker';


ReactDOM.render(<App />, document.getElementById('root'));
// ReactDOM.render(<Echartstest />, document.getElementById('root'));

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
