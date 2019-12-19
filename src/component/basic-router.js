import React, { Component } from 'react';
// import "antd/dist/antd.css";
import { BrowserRouter as Router, Route, Link } from 'react-router-dom'
// import App from '../App'

const Index = () => (
    <h2>Home</h2>
);

const About = () => (
    <h2>About</h2>
);

const Users = () => (
    <h2>Users</h2>
);


class Basis_router extends Component {

    render() {
        return (
            <Router>
                <div>
                    <nav>
                        <ul>
                            <li>
                                <Link to="/">Home</Link>
                            </li>
                            <li>
                                <Link to="/about/">About</Link>
                            </li>
                            <li>
                                <Link to="/users/">Users</Link>
                            </li>
                        </ul>
                    </nav>

                    <Route path="/" exact component={Index} />
                    <Route path="/about/" component={About} />
                </div>
            </Router>
        )
    }

}

export default Basis_router;