import React from 'react'
import Document_EN from './Document_EN'
import Document_CN from './Document_CN'

export default class About_Pet extends React.Component{
    // 构造
    constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            en:true
        };
        this.changeLanguage = this.changeLanguage.bind(this)
    }

    componentDidMount() {
        let is_en = localStorage.getItem('en')
        switch (is_en) {
            case 'true':
                this.setState({
                    en:true
                })
                break;
            case 'false':
                this.setState({
                    en:false
                })
                break;
            default:
                this.setState({
                    en:false
                })
                break;
        }
    }

    changeLanguage(e){
        document.documentElement.scrollTop = document.body.scrollTop =0;
        e.stopPropagation()
        this.setState({
            en:!this.state.en
        },()=>{
            localStorage.setItem('en',this.state.en);
        })
    }

    render() {
        return (
            this.state.en === true ? <Document_EN onClick={this.changeLanguage} {...this.props}/> : <Document_CN onClick={this.changeLanguage} {...this.props}/>
        )
    }

}

