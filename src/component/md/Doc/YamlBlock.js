import React from 'react';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { atelierEstuaryLight, atelierPlateauLight } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import { Form } from 'antd/lib/index';

class YamlBlock extends React.PureComponent {
    render() {
        const { value , type } = this.props;

        return (
            <SyntaxHighlighter language="" style={atelierPlateauLight}>
                {value}
            </SyntaxHighlighter>
        );
    }
}

export default Form.create()(YamlBlock);