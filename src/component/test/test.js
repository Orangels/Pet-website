import React, { Component } from 'react';
import Table from 'antd/lib/table';
import Button from 'antd/lib/button';
import QueueAnim from 'rc-queue-anim';
import PropTypes from 'prop-types';
import { TweenOneGroup } from 'rc-tween-one';
import {Model_com_1_ori, Model_com_2_ori,Model_com_3_ori,Model_com_4_ori} from '../../view/Home/Home_model_zoo_component'


const TableContext = React.createContext(false);

export default class Animate_test extends React.Component {
    static propTypes = {
        className: PropTypes.string,
    };

    static defaultProps = {
        className: 'table-enter-leave-demo',
    };

    constructor(props) {
        super(props);
        this.enterAnim = [
            {
                opacity: 0, x: 30, backgroundColor: '#fffeee', duration: 0,},
            {
                height: 0,
                duration: 200,
                type: 'from',
                delay: 250,
                ease: 'easeOutQuad',
                // onComplete: this.onEnd,
            },
            {
                opacity: 1, x: 0, duration: 250, ease: 'easeOutQuad',
            },
            { delay: 1000, backgroundColor: '#fff' },
        ];
        this.pageEnterAnim = [
            {
                opacity: 0, duration: 0,
            },
            {
                height: 0,
                duration: 150,
                type: 'from',delay: 150,
                ease: 'easeOutQuad',
                // onComplete: this.onEnd,
            },
            {
                opacity: 1, duration: 150, ease: 'easeOutQuad',
            },
        ];
        this.leaveAnim = [
            { duration: 250, opacity: 0 },
            { height: 0, duration: 200, ease: 'easeOutQuad' },
        ];
        this.pageLeaveAnim = [
            { duration: 150, opacity: 0 },
            { height: 0, duration: 150, ease: 'easeOutQuad' },
        ];

        // 动画标签，页面切换时改用 context 传递参数；
        this.animTag = ($props) => {
            return (
                <TableContext.Consumer>
                    {(isPageTween) => {
                        return (
                            <TweenOneGroup
                                component="tbody"
                                enter={!isPageTween ? this.enterAnim : this.pageEnterAnim}
                                leave={!isPageTween ? this.leaveAnim : this.pageLeaveAnim}
                                appear={false}
                                exclusive
                                {...$props}
                            />
                        );
                    }}
                </TableContext.Consumer>
            );};

        this.state = {
            show:false,
            // Model_com_arr:[<Model_com_1 key={'1'}/>, <Model_com_2 key={'2'}/>,<Model_com_3 key={'3'}/>,<Model_com_4 key={'4'}/>],
        };
    }

    onEnd = (e) => {
        const dom = e.target;
        dom.style.height = 'auto';
    }

    onAdd = () => {
        this.setState({
            show:!this.state.show
        });
        // if (this.state.show){
        //     this.setState({
        //         // Model_com_arr:[<Model_com_1 key={'1'}/>, <Model_com_2 key={'2'}/>,<Model_com_3 key={'3'}/>,<Model_com_4 key={'4'}/>,<Model_com_1 key={'5'}/>, <Model_com_2 key={'6'}/>,<Model_com_3 key={'7'}/>,<Model_com_4 key={'8'}/>],
        // })}
        // else {
        //     this.setState({
        //         // Model_com_arr:[<Model_com_1 key={'1'}/>, <Model_com_2 key={'2'}/>,<Model_com_3 key={'3'}/>,<Model_com_4 key={'4'}/>],
        //     })
        // }
    };

    render() {
        let Model_com_arr = this.state.Model_com_arr
        return (
            <div >
                <Button type="primary" onClick={this.onAdd}>Add</Button>
                <QueueAnim
                    type="bottom"
                    key="block"
                    leaveReverse
                    style={{display:"flex",flexDirection:"column", height:314}}
                >
                    {this.state.show? <Model_com_1_ori key={'1'}/>:<Model_com_2_ori key={'2'}/>}
                </QueueAnim>
            </div>
        );
    }
}