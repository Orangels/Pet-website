import React from 'react';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { a11yLight } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import { Form } from 'antd/lib/index';

class ShellBlock extends React.PureComponent {
    render() {
        const { value , type } = this.props;

        return (
            <SyntaxHighlighter language="" style={a11yLight} className='code_shell'>
                {value}
            </SyntaxHighlighter>
        );
    }
}

export default Form.create()(ShellBlock);
