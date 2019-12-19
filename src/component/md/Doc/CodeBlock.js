import React from 'react';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { atelierDuneLight } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import { Form } from 'antd/lib/index';

class CodeBlock extends React.PureComponent {
    render() {
        const { value , type } = this.props;

        return (
            <SyntaxHighlighter language="" style={atelierDuneLight}>
                {value}
            </SyntaxHighlighter>
        );
    }
}

export default Form.create()(CodeBlock);
