// import React, { Component } from 'react';
// import '../style/MSCOCO2017.less'
// import Help from "./Help";
// import Optmization_iteration from './Optimization_iteration.md'
// import Md_content_component from "./common/Md_content_component";
// import {test_md_data} from "../../common/data/Doc/EN/test_md";
//
//
//
// export default class Test_md extends Component{
//     // 构造
//       constructor(props) {
//         super(props);
//         // 初始状态
//         this.state = {markdown:''};
//       }
//
//     componentWillMount() {
//         let { style, ...props } = this.props;
//
//         fetch(Optmization_iteration)
//             .then(res => res.text())
//             .then(text => this.setState({ markdown: text }));
//     }
//
//       render() {
//           return (<Help markdown_text={this.state.markdown}/>)
//       }
//
//
//     // constructor(props) {
//     //     super(props);
//     //     // 初始状态
//     //     this.state = {};
//     // }
//     //
//     // render() {
//     //     let str = '## 在MSCOCO2017数据集上训练Mask-RCNN模型\n' +
//     //         '\n' +
//     //         '本教程将介绍使用**Pet**训练以及测试Mask-RCNN模型进行目标检测的主要步骤，在此我们会指导您如何通过组合**Pet**的提供的各个功能模块来构建Mask-RCNN模型，在此我们仅讲解组件的调用，部分实现细节请查阅系统组件的相应部分。在阅读本教程的之前我们强烈建议您阅读原始论文[Fast-RCNN]()和[Mask-RCNN]()以了解更多关于Mask-RCNN的算法原理。\n' +
//     //         '\n' +
//     //         '\n' +
//     //         '\n' +
//     //         '如果您具有丰富的目标检测算法的研究经验，您也可以直接在**Pet**中运行`$Pet/tools/rcnn/train_net.py`脚本利己开始训练您的Mask-RCNN模型.'
//     //     // return (
//     //     //     <p>
//     //     //         {str}
//     //     //     </p>
//     //     // )
//     //     return (
//     //         <Md_content_component data={test_md_data} />
//     //     )
//     //
//     // }
//
// }