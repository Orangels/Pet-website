import React from 'react'
import Contact_us_ZN from './Contact_us_ZN'
import Contact_us_EN from './Contact_us_EN'

export default class Contact_us extends React.Component{
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
            this.state.en === true ? <Contact_us_EN onClick={this.changeLanguage} {...this.props}/> : <Contact_us_ZN onClick={this.changeLanguage} {...this.props}/>
        )
    }

}

