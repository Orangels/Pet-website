import React from 'react'
import Home_Thin from './Home_Thin'
import Home_CN from './Home_CN'

export default class Home extends React.Component{
    // 构造
      constructor(props) {
        super(props);
        // 初始状态
          let is_en = localStorage.getItem('en')
          switch (is_en) {
              case 'true':
                  this.state = {
                      en:true
                  };
                  break;
              case 'false':
                  this.state = {
                      en:false
                  };
                  break;
              default:
                  this.state = {
                      en:true
                  };
                  break;
          }
        // this.state = {
        //     en:true
        // };
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
              let is_en = localStorage.getItem('en')
          })
      }

      render() {
          return (
              this.state.en === true ? <Home_Thin onClick={this.changeLanguage} {...this.props}/> : <Home_CN onClick={this.changeLanguage} {...this.props}/>
          )
      }

}

