import React from 'react'
import ReactMarkdown from 'react-markdown'
import AppMarkdown from './test.md';

export default class Markdown_ori extends React.Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            markdown:''
        };
      }

    componentWillMount() {
        fetch(AppMarkdown)
            .then(res => res.text())
            .then(text => this.setState({ markdown: text }));
    }

      render() {
          return <ReactMarkdown source={this.state.markdown} />
      }

}