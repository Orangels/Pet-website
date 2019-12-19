import React from 'react'
import Model_Zoo_EN from './Model_Zoo_EN'
import Model_Zoo_CN from './Model_Zoo_CN'


export default class Model_Zoo extends React.Component{
    // 构造
    constructor(props) {
        super(props);
        // 初始状态
        this.state = {
            en:true,
            // selectKey:10
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

        // this.setState({
        //     selectKey:this.props.history.location.state.some.selectedKeys
        // })
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
        console.log(`Model Zoo select key ${this.state.selectKey}`)
        return (
            this.state.en === true ? <Model_Zoo_EN onClick={this.changeLanguage} {...this.props} selectKey={this.props.history.location.state.some.selectedKeys}/> : <Model_Zoo_CN onClick={this.changeLanguage} {...this.props} selectKey={this.props.history.location.state.some.selectedKeys}/>
        )
    }

}

