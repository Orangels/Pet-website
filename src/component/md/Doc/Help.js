import React, { Component } from 'react';
import { Card, Form } from 'antd/lib/index';
import ReactMarkdown from 'react-markdown';
import codeBlock from './CodeBlock';
import ShellBlock from './ShellBlock'
import YamlBlock from './YamlBlock'

import AppMarkdown from './MSCOCO2017.md';
import {screen_scale_width} from "../../../common/parameter/parameters";
// import 'github-markdown-css';


const Text_component = ({text})=>(
    <p className={'Text_component'} style={{color:'#FBA339', marginRight:30*screen_scale_width, float:'left'}}>{text}</p>
);


class Help extends Component {
    state = {
        markdown: '',
        text:null,
        border:false,
        type: 'code'
    };

    componentWillMount() {
        let { style, ...props } = this.props;
        this.setState({
            markdown:props.markdown_text,
            style:style,
            text:props.text,
            border:props.border,
            type:props.type
        })

    }

    componentWillReceiveProps(nextProps, nextContext) {
        // this._detailData(nextProps.table_data)
        this.setState({
            markdown:nextProps.markdown_text,
            style:nextProps.style,
            text:nextProps.text,
            border:nextProps.border,
            type:nextProps.type
        })
    }

    render() {
        const { markdown } = this.state;

        let Block = null;

        switch (this.state.type) {
            case "code":
                Block = codeBlock;
                break;
            case "shell":
                Block = ShellBlock;
                break;
            case "yaml":
                Block = YamlBlock;
                break;
            default:
                Block = ShellBlock;
        }

        return (
            <div className='advancedForm' style={{width:'auto'}}>
                {/*<Card className='card' bordered={false} style={this.state.border ? this.state.style : null} size='small'>*/}
                    {/*<div style={this.state.text ? this.state.style : null}>*/}
                    <div style={this.state.style}>
                        {this.state.text ? <Text_component text={this.state.text} />:null}
                            <ReactMarkdown
                                style={{float:'left'}}
                                className="markdown-body"
                                source={markdown}
                                escapeHtml={false}
                                renderers={{
                                    code: Block,
                                }}
                            />
                            <div style={{clear:'both'}}></div>
                    </div>
                {/*</Card>*/}
            </div>
        );
    }
}

export default Form.create()(Help);
