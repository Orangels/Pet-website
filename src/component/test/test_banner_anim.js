import React from 'react';
import BannerAnim, { Element,Thumb, Arrow} from 'rc-banner-anim';
import TweenOne from 'rc-tween-one';
import 'rc-banner-anim/assets/index.css';
import './test_banner_anim.less'


const BgElement = Element.BgElement;
const { animType, setAnimCompToTagComp } = BannerAnim;

animType.custom = (elem, type, direction, animData) => {
    console.log(`custom animType, type:${type}`); // eslint-disable-line no-console
    let _y;
    const props = { ...elem.props };
    let children = props.children;
    if (type === 'enter') {
        _y = direction === 'next' ? '100%' : '-100%';
    } else {
        _y = direction === 'next' ? '-10%' : '10%';
        children = React.Children.toArray(children).map(setAnimCompToTagComp);
    }
    return React.cloneElement(elem, {
        ...props,
        animation: {
            ...animData,
            y: _y,
            type: type === 'enter' ? 'from' : 'to',
        },
    }, children);
};


class Demo extends React.Component {
    constructor(props) {
        super(props);
        this._onMouseEnter = this._onMouseEnter.bind(this);
        this._onMouseLeave = this._onMouseLeave.bind(this);
    }

    _onMouseEnter(e) {
        this.banner.slickGoTo(1)
    }

    _onMouseLeave(e){
        this.banner.slickGoTo(0)
    }

    render(){
        return (
            <BannerAnim prefixCls="banner-user" onMouseEnter={this._onMouseEnter} onMouseLeave={this._onMouseLeave}
                        ref={(c) => { this.banner = c; }}
                        type="custom" style={{height:500, width:300}}>
                <Element
                    prefixCls="banner-user-elem"
                    key="0"
                >
                    <BgElement
                        key="bg"
                        className="bg"
                        style={{
                            background: '#364D79',
                            border:'2px solid rgba(251,139,35,1)',
                        }}
                    />
                    <TweenOne className="banner-user-title" animation={{ y: 30, opacity: 0, type: 'from' }}>
                        Ant Motion Banner
                    </TweenOne>
                    <TweenOne className="banner-user-text"
                              animation={{ y: 30, opacity: 0, type: 'from', delay: 100 }}
                    >
                        The Fast Way Use Animation In React
                    </TweenOne>
                </Element>
                <Element
                    prefixCls="banner-user-elem"
                    key="1"
                >
                    <BgElement
                        key="bg"
                        className="bg"
                        style={{
                            background: '#64CBCC',
                        }}
                    />
                    <TweenOne className="banner-user-title" animation={{ y: 30, opacity: 0, type: 'from' }}>
                        Ant Motion Banner
                    </TweenOne>
                    <TweenOne className="banner-user-text"
                              animation={{ y: 30, opacity: 0, type: 'from', delay: 100 }}
                    >
                        The Fast Way Use Animation In React
                    </TweenOne>
                </Element>
                <Element
                    prefixCls="banner-user-elem"
                    key="1"
                >
                    <BgElement
                        key="bg"
                        className="bg"
                        style={{
                            background: '#64CBCC',
                        }}
                    />
                    <TweenOne className="banner-user-title" animation={{ y: 30, opacity: 0, type: 'from' }}>
                        Ant Motion Banner
                    </TweenOne>
                    <TweenOne className="banner-user-text"
                              animation={{ y: 30, opacity: 0, type: 'from', delay: 100 }}
                    >
                        The Fast Way Use Animation In React
                    </TweenOne>
                </Element>
                <Thumb prefixCls="user-thumb" key="thumb" component={TweenOne}
                >
                </Thumb>
                <Arrow arrowType="prev" key="prev" prefixCls="user-arrow"/>
                <Arrow arrowType="next" key="next" prefixCls="user-arrow"/>
            </BannerAnim>);
    }
}

export default Demo;