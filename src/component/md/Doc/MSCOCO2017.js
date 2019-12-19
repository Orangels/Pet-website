// import React, { Component } from 'react';
// import '../style/MSCOCO2017.less'
// import Help from "./Help";
// import { Collapse, Icon } from 'antd';
// import { Table, Button } from 'antd/lib/index';
// const Panel = Collapse.Panel;
//
//
// const style = {
//     wrap:{
//         width:1000,
//         marginLeft:50,
//         display:'flex',
//         flexDirection:'column',
//     },
//     code_part:{
//         borderWidth:2,
//         borderLeftStyle:"solid",
//         borderColor:"#F78E3D"
//     },
//     segmentation_part:{
//         borderWidth:2,
//         borderBottomStyle:"solid",
//         borderColor:"#E4E4E4"
//     },
//     part:{
//         marginTop:30,
//         marginBottom:20,
//         backgroundColor:'#FCE8D5',
//         // height:60,
//         height:'auto',
//         display:'flex',
//         flexDirection:'column',
//         justifyContent:'center',
//     },
//     row_center_part:{
//         marginTop:5,
//         marginBottom:20,
//         height:5,
//         display:'flex',
//         flexDirection:'row',
//         justifyContent:'center',
//         alignItems:'center'
//     },
//     img:{
//         width:600
//     },
//     singal_img:{
//         width:500
//     },
//     customPanelStyle:{
//         background: '#E4E4E4',
//         borderRadius: 4,
//         // marginBottom: 24,
//         border: 0,
//         overflow: 'hidden',
//     },
//     Collapse:{
//         width:800,
//         alignSelf:'center',
//         borderTopStyle:"solid",
//         borderTopColor:"#FFA500",
//         borderTopWidth:2
//     }
// };
//
// const table_1 = [{
//     title: 'Method',
//     dataIndex: 'Method',
//     key: 'Method',
//     render: (text, record, index) => {
//         return (<p style={{fontSize:15,fontWeight:500}}>{text}</p>)
//     },
// }, {
//     title: 'Backbone',
//     dataIndex: 'Backbone',
//     key: 'Backbone',
//     render: (text, record, index) => {
//         return (<p style={{fontSize:12,fontWeight:500}}>{text}</p>)
//     },
// },{
//     title: 'boxAP',
//     dataIndex: 'boxAP',
//     key: 'boxAP',
//     render: (text, record, index) => {
//         return (<p style={{fontSize:12,fontWeight:500}}>{text}</p>)
//     },
// },{
//     title: 'maskAP',
//     dataIndex: 'maskAP',
//     key: 'maskAP',
//     render: (text, record, index) => {
//         return (<p style={{fontSize:12,fontWeight:500}}>{text}</p>)
//     },
// }];
//
// const table_2 = [{
//     title: 'Method',
//     dataIndex: 'Method',
//     key: 'Method',
//     render: (text, record, index) => {
//         return (<p style={{fontSize:15,fontWeight:500}}>{text}</p>)
//     },
// }, {
//     title: 'boxAP(person)',
//     dataIndex: 'boxAP',
//     key: 'boxAP',
//     render: (text, record, index) => {
//         return (<p style={{fontSize:12,fontWeight:500}}>{text}</p>)
//     },
// },{
//     title: 'maskAP(person)',
//     dataIndex: 'maskAP',
//     key: 'maskAP',
//     render: (text, record, index) => {
//         return (<p style={{fontSize:12,fontWeight:500}}>{text}</p>)
//     },
// },{
//     title: 'kpAP(person)',
//     dataIndex: 'kpAP',
//     key: 'kpAP',
//     render: (text, record, index) => {
//         return (<p style={{fontSize:12,fontWeight:500}}>{text}</p>)
//     }
// }];
//
// let table_1_data = [
//     {
//         key: '1',
//         Method: 'Faster RCNN',
//         Backbone:'R-50-FPN',
//         boxAP: 36.4,
//         maskAP:'----',
//     },{
//         key: '2',
//         Method: 'Mask RCNN',
//         Backbone:'R-50-FPN',
//         boxAP: 37.4,
//         maskAP:34.2,
//     },];
//
//
// let table_2_data = [
//     {
//         key: '1',
//         Method: 'Faster RCNN',
//         boxAP: 52.5,
//         maskAP:'----',
//         kpAP:'----'
//     },{
//         key: '2',
//         Method: 'Mask RCNN(mask only)',
//         boxAP: 53.6,
//         maskAP:45.8,
//         kpAP:'----'
//     },{
//         key: '3',
//         Method: 'Mask RCNN(keypoiny only)',
//         boxAP: 50.7,
//         maskAP:'----',
//         kpAP:64.2
//     },{
//         key: '4',
//         Method: 'Mask RCNN(keypoiny&mask)',
//         boxAP: 52,
//         maskAP:45.1,
//         kpAP:64.7
//     },];
//
// class MSCOCO2017 extends Component {
//     // 构造
//     constructor(props) {
//         super(props);
//         // 初始状态
//         this.state = {};
//     }
//
//     render() {
//         let slider_left_width = 250
//         let slider_right_width = 150
//         return (
//             <div style={style.wrap}>
//                 <h1 style={{height:45}}>在MSCOCO2017数据集上训练Mask-RCNN模型</h1>
//                 <br/>
//                 <Help markdown_text='**本教程将介绍使用Pet训练以及测试Mask-RCNN模型进行目标检测的主要步骤，在此我们会指导您如何通过组合Pet的提供的各个功能模块来构建Mask-RCNN模型，在此我们仅讲解组件的调用，部分实现细节请查阅系统组件的相应部分。在阅读本教程的之前我们强烈建议您阅读原始论文[Fast-RCNN]()和[Mask-RCNN]()以了解更多关于Mask-RCNN的算法原理。**'/>
//                 <p style={{marginLeft:15}}>
//                     <strong>如果您具有丰富的目标检测算法的研究经验，您也可以直接在Pet中运行</strong>
//                     <code style={{backgroundColor:'#F6F7F8'}}>Pet/tools/rcnn/train_net.py</code>
//                     <strong>脚本利己开始训练您的Mask-RCNN模型.</strong>
//                 </p>
//                 <h2>用法示例：</h2>
//                 <ul>
//                     <li>在8块GPU上使用`MSCOCO2017_train`训练一个端到端的Mask-RCNN模型，使用两个全连接层作为`RCNN`的功能网络：</li>
//                 </ul>
//                 <Help type='shell' style={style.code_part} markdown_text={'```\n' +
//                 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch tools/rcnn/train_net.py \n' +
//                 '--cfg cfgs/rcnn/mscoco/e2e_mask_rcnn_R-50-FPN_1x.yaml\n' +
//                 '```\n'}/>
//                 <Help markdown_text='* **在8块GPU上使用`MSCOCO2017_val`数据集上测试训练的Mask-RCNN模型：**'/>
//                 <Help type='shell' style={style.code_part} markdown_text={'```\n' +
//                 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/rcnn/test_net.py \n' +
//                 '--cfg cfgs/rcnn/mscoco/e2e_mask_rcnn_R-50-FPN_1x.yaml\n' +
//                 '```\n'} />
//                 <p style={{marginLeft:15 ,borderWidth:2,
//                     borderBottomStyle:"solid",
//                     borderColor:"#E7EBEE",paddingBottom:20}}>
//                     <strong>在进行任何与模型训练和测试有关的操作之前，需要先选择一个指定的</strong>
//                     <code style={{backgroundColor:'#F6F7F8'}}>yaml</code>
//                     <strong>文件，明确在训练时候对数据集、模型结构、优化策略以及其他重要参数的需求与设置，本教程以</strong>
//                     <code style={{backgroundColor:'#F6F7F8'}}>Pet/cfgs/rcnn/mscoco/e2e_mask_rcnn_R-50-FPN_1x.yaml</code>
//                     <strong>为例，讲解训练过程中所需要的关键配置，该套配置将指导此Mask-RCNN模型训练以及测试的全部步骤和细节，全部参数设置请见</strong>
//                     <a>e2e_mask_rcnn_R-50-FPN_1x.yaml</a>
//                 </p>
//                 <div style={style.part}>
//                     <h2 style={{paddingLeft:20, color:'#F78E3D',}}>数据载入</h2>
//                 </div>
//                 <Help markdown_text='**确保MSCOCO2017数据集已经存放在您的硬盘中，接下来我们可以开始加载`coco_2017_train`训练集。**'/>
//                 <Help type='shell' markdown_text={'```Python\n' +
//                 '    # Create data loder\n' +
//                 '    datasets = build_dataset(cfg.TRAIN.DATASETS, is_train=True)\n' +
//                 '    train_loader = make_train_data_loader(\n' +
//                 '        datasets,\n' +
//                 '        is_distributed=args.distributed,\n' +
//                 '        start_iter=scheduler.iteration,\n' +
//                 '    )\n' +
//                 '```\n'} />
//                 <Help  markdown_text='**使用MSCOCO数据集训练Faster-RCNN、Mask-RCNN任务时，将输入图像的短边缩放到800像素，同时保证长边不超过1333像素，这样做的目的是要保证输入图像的长宽比不失真。**'/>
//                 <Collapse
//                     style={style.Collapse}
//                     bordered={false}
//                     defaultActiveKey={['1']}
//                     expandIcon={({ isActive }) => (<div style={{paddingTop:10}}>
//                         <Icon style={{color:'orange', float:'left'}} type="right-circle" rotate={isActive ? 90 : 0} />
//                         <p style={{color:'orange', marginLeft:5, float:'left' ,fontSize:12}}>{isActive ? '收起' : '展开'}</p>
//                     </div>)}
//                 >
//                     <Panel key="1" style={style.customPanelStyle} header={(<div style={style.row_center_part}>
//                         <h3 style={{paddingTop:20,color:'#2F2376',}}>Faster-RCNN和Mask-RCNN的训练尺寸</h3>
//                     </div>)}>
//
//                         <Help border={true} style={{background: '#E4E4E4',}} markdown_text='在最早流行的目标检测模型的训练中，数据集内的图像在进入网络之前，其尺寸被按照短边600像素，长边不超过1000像素进行缩放，目的也是为了保证图像中的视觉目标长宽比不失真，这种做法在很长时间内被用于在PASCAL VOC以及MSCOCO数据集上训练Faster-RCNN模型。PASCAL VOC数据集的图片尺寸平均大小为384像素x500像素，且图像中视觉目标大多尺寸较大，而MSCOCO数据集图像中的目标数量大幅增加，同时MSCOCO数据集中大多数的目标的像素数不足图片像素数1%，这使得在MSCOCO数据集上进行目标检测的难度要远远高于在PASCAL VOC数据集上进行目标检测。'/>
//                         <div style={{marginTop:5,
//                             marginBottom:20,
//                             display:'flex',
//                             flexDirection:'row',
//                             justifyContent:'center',
//                             alignItems:'center',
//                             background: '#E4E4E4'}}>
//                             <img src={require('../../asset/md_img/voc&coco_image.png')} style={style.img}/>
//                         </div>
//                         <Help border={true} style={{background: '#E4E4E4',}} markdown_text={'随着目标检测算法以及卷积神经网络的发展，目标检测模型的精度越来越高，对于不同尺度，尤其是小物体的检测效果越来越受到重视，因此MSCOCO数据集被更加普遍的用来评估模型精度，在[FPN]()中提到在MSCOCO数据集上训练目标检测模型时增大输入图像的尺寸可以提升小目标的检测效果。原理很简单，在目前流行的Faster-RCNN、FPN等anchor-based检测算法中，需要通过主干网络对输入图像不断进行下采样，在经过16倍下采样之后，原始图像中某些尺寸很小的目标所保留的视觉信息已经所剩无几，适当得提升训练图像尺寸可以在一定程度上在下采样过程中保留小目标，因此从FPN开始，在MSCOCO数据集上训练目标检测模型的输入图像尺寸被按照短边800像素，长边不超过1333像素进行缩放。'}/>
//                     </Panel>
//                 </Collapse>
//                 <Help markdown_text='**Mask-RCNN还对训练数据进行了随机水平翻转来进行数据增广，提升模型的泛化性，经过变换的图像以及其标注的可视化结果如下图：**'/>
//                 <div style={{
//                     display:'flex',
//                     flexDirection:'row',
//                     justifyContent:'center',
//                     alignItems:'center'}}>
//                     <img src={require('../../asset/md_img/mask_all/206_ori.png')} style={style.singal_img}/>
//                     <img src={require('../../asset/md_img/mask_all/206.png')} style={style.singal_img}/>
//                 </div>
//                 <Help markdown_text='**数据载入组件不只是完成了图像数据以及标注信息的读取，还在采集每个批次的数据的同时生成了RPN网络的训练标签，数据载入组件输出的每个批次的数据中包含图片数据、图片中物体的类别，物体的包围框、以及与物体数量相同的分割掩模（每个掩模只包含一个目标的前景掩码）。**' />
//                 <Help markdown_text={'```\n' +
//                 'data: (1, 3, 800, 1196)\n' +
//                 'label: (1, 6)\n' +
//                 'box: (1, 6, 4)\n' +
//                 'mask: (1, 6, 800, 1196)\n' +
//                 '```'} text='Out' type='code'/>
//                 <div style={style.part}>
//                     <h2 style={{paddingLeft:20,color:'#F78E3D',}}>Mask-RCNN 网络结构</h2>
//                 </div>
//                 <Help markdown_text='**在Pet中，Mask-RCNN网络使用`Generalized_RCNN`模型构建器来搭建，`Generalized——RCNN`与`Generalized_CNN`、`Generalized_SSD`一样用来搭建完整的计算机视觉算法网络结构，并遵循Pet的模块化构建网络的规则。我们在`yaml`文件中设置如下参数来使用配置系统构建Mask-RCNN网络：**'/>
//                 <Help type='yaml' markdown_text={'```\n' +
//                 'MODEL:\n' +
//                 '  CONV_BODY: "resnet50_1x64d"\n' +
//                 '  MASK_ON: True\n' +
//                 'FPN:\n' +
//                 '  FPN_ON: True\n' +
//                 '```'}/>
//                 <Help type='shell' markdown_text={'```Python\n' +
//                 'class Generalized_RCNN(nn.Module):\n' +
//                 '    def __init__(self):\n' +
//                 '        super().__init__()\n' +
//                 '\n' +
//                 '        # Backbone for feature extraction\n' +
//                 '        conv_body = registry.BACKBONES[cfg.MODEL.CONV_BODY]\n' +
//                 '        self.Conv_Body = conv_body()\n' +
//                 '        self.dim_in = self.Conv_Body.dim_out\n' +
//                 '        self.spatial_scale = self.Conv_Body.spatial_scale\n' +
//                 '\n' +
//                 '        # Feature Pyramid Networks\n' +
//                 '        if cfg.FPN.FPN_ON:\n' +
//                 '            self.Conv_Body_FPN = FPN.fpn(self.dim_in, self.spatial_scale)\n' +
//                 '            self.dim_in = self.Conv_Body_FPN.dim_out\n' +
//                 '            self.spatial_scale = self.Conv_Body_FPN.spatial_scale\n' +
//                 '        else:\n' +
//                 '            self.dim_in = self.dim_in[-1]\n' +
//                 '            self.spatial_scale = [self.spatial_scale[-1]]\n' +
//                 '\n' +
//                 '        # Region Proposal Network\n' +
//                 '        self.RPN = build_rpn(self.dim_in)\n' +
//                 '\n' +
//                 '        if not cfg.MODEL.RETINANET_ON:\n' +
//                 '            self.Fast_RCNN = FastRCNN(self.dim_in, self.spatial_scale)\n' +
//                 '\n' +
//                 '            if cfg.MODEL.MASK_ON:\n' +
//                 '                self.Mask_RCNN = MaskRCNN(self.dim_in, self.spatial_scale)\n' +
//                 '```'}/>
//                 <Help markdown_text='**与`Generalized_CNN`略有不同，`Generalized_RCNN`对Mask-RCNN网络结构的划分主要有以下几点不同：**'/>
//                 {/*<p style={{marginLeft:24, fontWeight:500}}>与`Generalized_CNN`略有不同，`Generalized_RCNN`对Mask-RCNN网络结构的划分主要有以下几点不同：</p>*/}
//                 <Help markdown_text='* **除了特征提取网络、功能网络中、输出分支等三个网络模块之外，Mask-RCNN网络在`Generalized_RCNN`中还包括区域建议网络（RPN）；**'/>
//                 <Collapse
//                     style={style.Collapse}
//                     bordered={false}
//                     defaultActiveKey={['1']}
//                     expandIcon={({ isActive }) => (<div style={{paddingTop:10}}>
//                         <Icon style={{color:'orange', float:'left'}} type="right-circle" rotate={isActive ? 90 : 0} />
//                         <p style={{color:'orange', marginLeft:5, float:'left',fontSize:12}}>{isActive ? '收起' : '展开'}</p>
//                     </div>)}
//                 >
//                     <Panel key="1" style={style.customPanelStyle} header={(<div style={style.row_center_part}>
//                         <h3 style={{paddingTop:20,color:'#2F2376',}}>区域建议网络(RPN)</h3>
//                     </div>)}>
//                         <Help border={true} style={{background: '#E4E4E4',}} markdown_text='RPN将特征图上可能出现前景物体的区域提取出来作为候选框，其输入为特征图，输出为一系列矩形候选框。RPN的实现原理如下图，根据特征图（HxW）的下采样倍   率预设N个不同尺寸、不同长宽比的矩形窗口，以输入的特征图上的每一个像素位置为中心铺设N个窗口，这样在特征图上得到HxWxN个候选框，RPN网络进行区域生成的   本质是滑窗方法。在生成大量的候选框之后，根据每一个候选框与物体标注框之间的交并比来区分前景和背景候选框，生成RPN训练标签，包括每个前景候选框的类别和   坐标修正值。'/>
//                         <div style={{marginTop:5,
//                             marginBottom:20,
//                             display:'flex',
//                             flexDirection:'row',
//                             justifyContent:'center',
//                             alignItems:'center',
//                             background: '#E4E4E4'}}>
//                             <img src={require('../../asset/md_img/RPN.png')} style={style.img}/>
//                         </div>
//                         <Help border={true} style={{background: '#E4E4E4',}} markdown_text='RPN对输入特征图上的每一个像素点分别预测其上铺设候选框的两个类别以及四个坐标修正值，与数据载入时计算出来的RPN标签做偏差，进行RPN的训练。被修正位   置的前景和背景候选框被到`RCNN`网络进行RoIAlign，进一步地分类以及回归。'/>
//                     </Panel>
//                 </Collapse>
//                 <Help markdown_text='* **FPN结构（如果需要）被归纳于特征提取网络模块中，在基础特征提取网络构建之后，FPN结构被构建于特征提取网络之上，被称为`Conv_Body_FPN`；**'/>
//                 <Collapse
//                     style={style.Collapse}
//                     bordered={false}
//                     defaultActiveKey={['1']}
//                     expandIcon={({ isActive }) => (<div style={{paddingTop:10}}>
//                         <Icon style={{color:'orange', float:'left'}} type="right-circle" rotate={isActive ? 90 : 0} />
//                         <p style={{color:'orange', marginLeft:5, float:'left',fontSize:12}}>{isActive ? '收起' : '展开'}</p>
//                     </div>)}
//                 >
//                     <Panel key="1" style={style.customPanelStyle} header={(<div style={style.row_center_part}>
//                         <h3 style={{paddingTop:20,color:'#2F2376',}}>特征金字塔网络（FPN)</h3>
//                     </div>)}>
//                         <Help border={true} style={{background: '#E4E4E4',}} markdown_text='目标检测模型需要对图像中不同尺寸的目标进行定位与分类，但是在FPN之前的以Faster-RCNN为主的两阶段目标检测器大多在16倍下采样之后的特征图上铺设锚框   进行目标检测，经过多次下采样之后，小目标的信息已经所剩无几，因此Faster-RCNN对于小目标的检测效果一直有待提高。faster-RCNN、R-FCN等一系列方法普   遍采用多尺度测试和训练策略，即图像金字塔，虽然能够一定程度上提升对于小目标的检测效果，但是随之而来的计算开销也是巨大的。'/>
//                         <Help border={true} style={{background: '#E4E4E4',}} markdown_text='在单阶段检测器中，SSD为了检测不同尺寸的目标，同时不增加计算开销，在特征提取网络中的多级特征图上铺设预选框，利用了特征金字塔进行目标检测取得了一定   的效果，但是以SSD为代表的单阶段检测器由于底层特征的语义信息的缺乏，检测效果始终逊与两阶段检测器。特征金字塔在两阶段检测器中的使用受阻主要原因在于，   特征金字塔和RPN网络结构的工程实现难度较大，直到FPN的出现才解决这一问题。'/>
//                         <div style={{marginTop:5,
//                             marginBottom:20,
//                             display:'flex',
//                             flexDirection:'row',
//                             justifyContent:'center',
//                             alignItems:'center',
//                             background: '#E4E4E4'}}>
//                             <img src={require('../../asset/md_img/FPN.png')} style={style.img}/>
//                         </div>
//                         <Help border={true} style={{background: '#E4E4E4',}} markdown_text='FPN在两阶段检测器中利用了特征金字塔，同时在进行区域建议之前对不同分辨率的特征进行了融合，补强了底层特征图的语义信息，使用FPN结构的两阶段检测器在   多个分辨率特征图上进行区域建议，对所有RPN提取的候选框根据其大小重新将之分配在特定层级的特征图上进行RoIAlign。使用FPN结构可以使两阶段检测器的目标   检测效果稳定提升1%，在FPN出现之后，单阶段检测器也普遍使用了FPN结构来提升检测精度，具有代表性的工作如[RetinaNet]()。'/>
//                     </Panel>
//                 </Collapse>
//                 <Help markdown_text='* **功能网络中模块包含检测分支（FastRCNN）与实例分割分支（MaskRCNN），损失函数被构建在相应的功能网络中。**'/>
//                 <Collapse
//                     style={style.Collapse}
//                     bordered={false}
//                     defaultActiveKey={['1']}
//                     expandIcon={({ isActive }) => (<div style={{paddingTop:10}}>
//                         <Icon style={{color:'orange', float:'left'}} type="right-circle" rotate={isActive ? 90 : 0} />
//                         <p style={{color:'orange', marginLeft:5, float:'left',fontSize:12}}>{isActive ? '收起' : '展开'}</p>
//                     </div>)}
//                 >
//                     <Panel key="1" style={style.customPanelStyle} header={(<div style={style.row_center_part}>
//                         <h3 style={{paddingTop:20, color:'#2F2376',}}>基于区域的多任务学习</h3>
//                     </div>)}>
//                         <Help border={true} style={{background: '#E4E4E4',}} markdown_text='Mask-RCNN在Faster-RCNN的基础上添加了实例分割分支，同时进行两个实例分析任务的学习，下表中是Pet下Faster-RCNN与Mask-RCNN在MSCOCO2017_train训练得到的模型在MSCOCO2017_val上进行评估所得精度对比，可以看出在加入实例分割任务之后，目标检测任务的精度也得到了可观的提升。'/>
//                         <Table columns={table_1} dataSource={table_1_data} rowClassName={'table_1'} pagination={false}/>
//                         <Help border={true} style={{background: '#E4E4E4',}} markdown_text='在[Mask-RCNN]()论文中也展示了不同任务之间相互影响的退化实验。如下表所示，使用ResNet50作为backbone在MSCOCO2017_train训练得到的模型在MSCOCO2017_val集上进行评估，在`person`这一类别上进行对比，`FastRCNN`、`MaskRCNN`与`KeypointRCNN`三个任务在训练过程中的损失函数拥有同样的权重。可知将`MaskRCNN`分支添加到Faster-RCNN或KeyPoint-RCNN上一致地提升了模型在这些任务上的精度，但添加`KeypointRCNN`在Faster-RCNN或Mask-RCNN上会略微降低`boxAP`和`maskAP`，这表明虽然关键点检测受益于多任务训练，但它不会反过来帮助其他任务。更多的实验表明不同的实例分析任务之间是有联系的，某些任务之间会相互促进，有些任务共同训练则会产生负面的影响。'/>
//                         <Table style={{marginBottom:20}} columns={table_2} dataSource={table_2_data} rowClassName={'table_1'} pagination={false} />
//                     </Panel>
//                 </Collapse>
//                 <div style={style.part}>
//                     <h2 style={{paddingLeft:20,color:'#F78E3D',}}>训练</h2>
//                 </div>
//                 <Help border={true} markdown_text='**完成了数据载入以及模型构建之后，我们需要在开始训练之前选择训练Mask-RCNN模型的优化策略，在批次大小为16的情况下，设置初始学习率为0.02，训练900000次迭代，组合使用了学习率预热与阶段下降策略，分别在60000与80000次迭代时将学习率减小十倍。**'/>
//                 <Help type='shell' border={true} markdown_text={'```Python\n' +
//                 'def train(model, loader, optimizer, scheduler, checkpointer, logger):\n' +
//                 '    # switch to train mode\n' +
//                 '    model.train()\n' +
//                 '    device = torch.device(\'cuda\')\n' +
//                 '\n' +
//                 '    # main loop\n' +
//                 '    cur_iter = scheduler.iteration\n' +
//                 '    for iteration, (images, targets, _) in enumerate(loader, cur_iter):\n' +
//                 '        logger.iter_tic()\n' +
//                 '        logger.data_tic()\n' +
//                 '\n' +
//                 '        scheduler.step()    # adjust learning rate\n' +
//                 '        optimizer.zero_grad()\n' +
//                 '\n' +
//                 '        images = images.to(device)\n' +
//                 '        targets = [target.to(device) for target in targets]\n' +
//                 '        logger.data_toc()\n' +
//                 '\n' +
//                 '        outputs = model(images, targets)\n' +
//                 '\n' +
//                 '        logger.update_stats(outputs, args.distributed, args.world_size)\n' +
//                 '        loss = outputs[\'total_loss\']\n' +
//                 '        loss.backward()\n' +
//                 '        optimizer.step()\n' +
//                 '\n' +
//                 '        if args.local_rank == 0:\n' +
//                 '            logger.log_stats(scheduler.iteration, scheduler.new_lr)\n' +
//                 '\n' +
//                 '            # Save model\n' +
//                 '            if cfg.SOLVER.SNAPSHOT_ITERS > 0 and (iteration + 1) % cfg.SOLVER.SNAPSHOT_ITERS == 0:\n' +
//                 '                checkpointer.save(model, optimizer, scheduler, copy_latest=True, infix=\'iter\')\n' +
//                 '        logger.iter_toc()\n' +
//                 '    return None\n' +
//                 '    \n' +
//                 '    # Train\n' +
//                 '    logging_rank(\'Training starts !\', distributed=args.distributed, local_rank=args.local_rank)\n' +
//                 '    train(model, train_loader, optimizer, scheduler, checkpointer, logger)\n' +
//                 '    logging_rank(\'Training done !\', distributed=args.distributed, local_rank=args.local_rank)\n' +
//                 '```'}/>
//                 <p style={{marginLeft:15}}>
//                     <strong>在训练过程中，日志记录仪会在每若干次迭代后记录当前网络训练的迭代数、各项偏差数值等训练信息，检查点组件会定期保存网络模型到配置系统中</strong>
//                     <code style={{backgroundColor:'#F6F7F8'}}>cfg.CKPT</code>
//                     <strong>所设置的路径下。</strong>
//                 </p>
//                 <p style={{marginLeft:15}}>
//                     <strong>根据</strong>
//                     <code style={{backgroundColor:'#F6F7F8'}}>cfg.DISPLAY_ITER</code>
//                     <strong>设置的日志记录间隔，在训练过程中每经过20次迭代，日志记录仪会在终端中记录模型的训练状态。</strong>
//                 </p>
//                 <Help border={true} markdown_text={'```\n' +
//                 '[Training][e2e_mask_rcnn_R-50-FPN_1x.yaml][iter: 80/90000][lr: 0.004400][eta: 22:07:32]\n' +
//                 '          total_loss: 1.517374 (1.552808), iter_time: 0.7195 (0.8858), data_time: 0.2417 (0.2413)\n' +
//                 '          loss_mask: 0.357312 (0.375371), loss_objectness: 0.352190 (0.361728), loss_classifier: 0.366364 (0.368482), loss_rpn_box_reg: 0.236925 (0.257432), loss_box_reg: 0.191814 (0.203634)\n' +
//                 '[Training][e2e_mask_rcnn_R-50-FPN_1x.yaml][iter: 100/90000][lr: 0.004667][eta: 21:25:25]\n' +
//                 '          total_loss: 1.509785 (1.562251), iter_time: 0.8045 (0.8579), data_time: 0.2629 (0.2414)\n' +
//                 '          loss_mask: 0.314586 (0.326509), loss_objectness: 0.343614 (0.357139), loss_classifier: 0.369052 (0.367820), loss_rpn_box_reg: 0.215749 (0.234119), loss_box_reg: 0.189691 (0.193587)\n' +
//                 '[Training][e2e_mask_rcnn_R-50-FPN_1x.yaml][iter: 120/90000][lr: 0.004933][eta: 21:04:43]\n' +
//                 '          total_loss: 1.571844 (1.582153), iter_time: 0.7302 (0.8443), data_time: 0.2380 (0.2422)\n' +
//                 '          loss_mask: 0.333583 (0.353402), loss_objectness: 0.342298 (0.350190), loss_classifier: 0.347794 (0.357265), loss_rpn_box_reg: 0.239373 (0.256294), loss_box_reg: 0.207887 (0.215416)\n' +
//                 '```'} text='Out' />
//                 <div style={style.part}>
//                     <h2 style={{paddingLeft:20,color:'#F78E3D',}}>测试</h2>
//                 </div>
//                 <p style={{marginLeft:15}}>
//                     <strong>在完成Mask-RCNN模型的训练之后，我们使用</strong>
//                     <code style={{backgroundColor:'#F6F7F8'}}>Pet/tools/rcnn/test_net.py</code>
//                     <strong>在MSCOCO2017_val评估模型的精度。同样需需要使用</strong>
//                     <code style={{backgroundColor:'#F6F7F8'}}>Dataloader</code>
//                     <strong>来加载测试数据集，并对图像做同样尺度的缩放。</strong>
//                 </p>
//                 <p style={{marginLeft:15}}>
//                     <strong>通过加载训练最大迭代数之后的模型</strong>
//                     <code style={{backgroundColor:'#F6F7F8'}}>Pet/ckpts/rcnn/mscoco/e2e_mask_rcnn_R-50-FPN-600_0.5x/model_latest.pth</code>
//                     <strong>，执行下面的命令进行Mask-RCNN模型的测试，Mask-RCNN的测试日志同样会被</strong>
//                     <code style={{backgroundColor:'#F6F7F8'}}>Logger</code>
//                     <strong>所记录。</strong>
//                 </p>
//                     <Help type='shell' style={style.code_part} markdown_text={'```\n' +
//                     'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch tools/rcnn/test_net.py \n' +
//                     '--cfg cfgs/rcnn/mscoco/e2e_mask_rcnn_R-50-FPN_1x.yaml\n' +
//                     '```\n'}/>
//                     <h2>用法示例：</h2>
//                     <Help markdown_text={'**在Pet中Mask-RCNN返回每一个目标的类别ID、置信度分数，边界框坐标和分割掩码。将MSCOCO2017_val中的图片的推理结果进行可视化如下图。**'}/>
//                 <div style={{
//                     display:'flex',
//                     flexDirection:'row',
//                     justifyContent:'center',
//                     alignItems:'center'}}>
//                     <img src={require('../../asset/md_img/mask_all/417.png')} style={style.singal_img}/>
//                     <img src={require('../../asset/md_img/mask_all/423.png')} style={style.singal_img}/>
//                 </div>
//             </div>
//
//         )
//
//     }
// }
//
// export default MSCOCO2017