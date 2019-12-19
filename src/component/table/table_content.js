import React from "react";
import background from '../../asset/HOME_icon/Model_Zoo/bj.jpg'
import {screen_scale_width} from "../../common/parameter/parameters";
import {model_content_data} from '../../view/Model/model_content_data'


import './table.css'

const style = {
    mid:{
        fontSize:15,
        fontWeight:500,
        color:'#FFFFFF'
    },
    big:{
        fontSize:20,
        fontWeight:600,
        color:'#FFFFFF'
    },
    bj:{
        background:`url(${background}) no-repeat `,
        width: '100%',
        height:210*screen_scale_width,
        backgroundSize: '100% 100%',
        display:'flex',
        flexDirection:'column',
        alignItems:'center',
        justifyContent:'center',
        marginTop:50*screen_scale_width,
        marginBottom:50*screen_scale_width
    },
    text:{
        fontSize:27*screen_scale_width,
        color: '#FFFFFF',
        letterspacing:'0.12px'
    }
}

let Content_block_detail = ({data}) => {
    data = data.map((item, i)=>{
        return (
            <span style={{fontSize:22*screen_scale_width, color: '#3765A0', letterSpacing:'0.08px',
            fontFamily:'freight-sans, sans-serif'}}>
                {item}
            </span>
        )
    })

    return (
        <div style={{display:'flex', flexDirection:"column", marginLeft:20*screen_scale_width, lineHeight:`${38*screen_scale_width}px`}}>
            {data}
        </div>
    )
}

let Content_block = ({data}) => {

    data = data.map((item,i)=>{
        return (
            <Content_block_detail data={item} />
        )
    })

    return (
        <div style={{display:'flex', height:478*screen_scale_width, width:738*screen_scale_width,
            padding:'47px 45px 41px 39px', boxShadow:'0 0 4px 5px rgba(159,159,159,0.14)'}}>
            {data}
        </div>
    )
}

let Content_block_wrap = ({data}) => {

    data = data.map((item,i)=>{
        return (
            <Content_block data={item} />
        )
            })

    return (
        <div style={{display:'flex', justifyContent: 'space-between'}}>
            {data}
        </div>
    )
}


export let Classification_cifar_content = ({type}) => {
    switch (type) {
        case 0:
            return (<div style={{backgroundColor:"#E4F4F6", marginTop:10, marginBottom:10, textAlign:'left', padding:15, }}>
                <b style={{fontSize:20, fontWeight:600}}>Model attributes are coded in their names. For instance, ssd_300_vgg16_atrous_voc consists of four parts</b>
                <b style={{fontSize:20, fontWeight:600}}>Hardware :</b>
                <ul type="circle">
                    <li style={{fontSize:15, fontWeight:500}} type="disc">1 NVIDIA GTX Titan V GPU *</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc">Intel Xeon 4114 CPU @ 2.20GHz</li>
                </ul>
                <b style={{fontSize:20, fontWeight:600}}>Software environment</b>
                <ul type="circle">
                    <li style={{fontSize:15, fontWeight:500}} type="disc">Python 3.5</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc">Pet 1.0</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc">CUDA 9.0.176</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc">CUDNN 7.0.4</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc">NCCL 2.1.15</li>
                </ul>
                <b style={{fontSize:20, fontWeight:600}}>训练阶段:详细参数相见对应的yaml。
                </b>
            </div>)
        case 1:
            return (<div style={{backgroundColor:"#E4F4F6", marginTop:10, marginBottom:10, textAlign:'left', padding:15, }}>
                <b style={{fontSize:20, fontWeight:600}}>Model attributes are coded in their names. For instance, ssd_300_vgg16_atrous_voc consists of four parts:</b>
                <ul type="circle" >
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>ssd</strong> indicate the algorithm is “Single Shot Multibox Object Detection” </li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>300</strong> is the training image size, which means training images are resized to 300x300 and all anchor boxes are designed to match this shape. This may not apply to some models.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>vgg16_atrous</strong> is the type of base feature extractor network.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>voc</strong> is the training dataset. You can choose voc or coco, etc.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>(320x320)</strong> indicate that the model was evaluated with resolution 320x320. If not otherwise specified, all detection models in GluonCV can take various input shapes for prediction. Some models are trained with various input data shapes, e.g., Faster-RCNN and YOLO models.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>ssd_300_vgg16_atrous_voc_int8</strong> is a quantized model calibrated on Pascal VOC dataset for <strong>00000</strong>strong>.</li>
                </ul>
            </div>)
    }
} ;

export let Classification_image_content = ({type}) => {
    switch (type) {
        case 0:
            return (<div style={{backgroundColor:"#E4F4F6", marginTop:10, marginBottom:10, textAlign:'left', padding:15, }}>
                <b style={{fontSize:20, fontWeight:600}}>Model attributes are coded in their names. For instance, ssd_300_vgg16_atrous_voc consists of four parts:</b>
                <ul type="circle" >
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>ssd</strong> indicate the algorithm is “Single Shot Multibox Object Detection” </li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>300</strong> is the training image size, which means training images are resized to 300x300 and all anchor boxes are designed to match this shape. This may not apply to some models.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>vgg16_atrous</strong> is the type of base feature extractor network.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>voc</strong> is the training dataset. You can choose voc or coco, etc.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>(320x320)</strong> indicate that the model was evaluated with resolution 320x320. If not otherwise specified, all detection models in GluonCV can take various input shapes for prediction. Some models are trained with various input data shapes, e.g., Faster-RCNN and YOLO models.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>ssd_300_vgg16_atrous_voc_int8</strong> is a quantized model calibrated on Pascal VOC dataset for <strong>11111</strong>strong>.</li>
                </ul>
            </div>)
        case 1:
            return (<div style={{backgroundColor:"#E4F4F6", marginTop:10, marginBottom:10, textAlign:'left', padding:15, }}>
                <b style={{fontSize:20, fontWeight:600}}>Model attributes are coded in their names. For instance, ssd_300_vgg16_atrous_voc consists of four parts:</b>
                <ul type="circle" >
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>ssd</strong> indicate the algorithm is “Single Shot Multibox Object Detection” </li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>300</strong> is the training image size, which means training images are resized to 300x300 and all anchor boxes are designed to match this shape. This may not apply to some models.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>vgg16_atrous</strong> is the type of base feature extractor network.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>voc</strong> is the training dataset. You can choose voc or coco, etc.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>(320x320)</strong> indicate that the model was evaluated with resolution 320x320. If not otherwise specified, all detection models in GluonCV can take various input shapes for prediction. Some models are trained with various input data shapes, e.g., Faster-RCNN and YOLO models.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>ssd_300_vgg16_atrous_voc_int8</strong> is a quantized model calibrated on Pascal VOC dataset for <strong>111111</strong>strong>.</li>
                </ul>
            </div>)
    }
} ;

export let Detection_coco_content = ({type, en}) => {
    switch (type) {
        case 0:
            return ( en == 0 ?
                    <div style={{display:"flex", flexDirection:'column',}}>
                    </div> :
                    <div style={{display:"flex", flexDirection:'column',}}>
                    </div>
            );
        case 1:
            return (
                en == 0 ?
                    <div style={{display:"flex", flexDirection:'column',}}>
                    <span style={{
                        // fontSize:27*screen_scale_width,
                        fontSize:16,
                        color: '#484B4D',letterSpacing:'0.12px',marginBottom:34*screen_scale_width}}>
                        To facilitate collation and search, we name  each model by the configuration of model:
                    </span>
                    <div style={{marginBottom:41*screen_scale_width}}>
                        <Content_block_wrap data={model_content_data.detection.coco.detail_content}/>
                    </div>
                    </div> :
                    <div style={{display:"flex", flexDirection:'column',}}>
                    <span style={{fontSize:22*screen_scale_width,color: '#484B4D',letterSpacing:'0.12px',marginBottom:34*screen_scale_width}}>
                        为了便于整理和查找，我们采用通用的模型命名方式来定义每个模型的配置，下面是使用的缩写词的解释:
                    </span>
                        <div style={{marginBottom:41*screen_scale_width}}>
                            <Content_block_wrap data={model_content_data.classification.cifar.detail_content}/>
                        </div>
                    </div>
            )
    }

} ;


export let Detection_voc_content = ({type}) => {
    switch (type) {
        case 0:
            return (<div style={{backgroundColor:"#E4F4F6", marginTop:10, marginBottom:10, textAlign:'left', padding:15, }}>
                <b style={{fontSize:20, fontWeight:600}}>Model attributes are coded in their names. For instance, ssd_300_vgg16_atrous_voc consists of four parts:</b>
                <ul type="circle" >
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>ssd</strong> indicate the algorithm is “Single Shot Multibox Object Detection” </li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>300</strong> is the training image size, which means training images are resized to 300x300 and all anchor boxes are designed to match this shape. This may not apply to some models.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>vgg16_atrous</strong> is the type of base feature extractor network.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>voc</strong> is the training dataset. You can choose voc or coco, etc.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>(320x320)</strong> indicate that the model was evaluated with resolution 320x320. If not otherwise specified, all detection models in GluonCV can take various input shapes for prediction. Some models are trained with various input data shapes, e.g., Faster-RCNN and YOLO models.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>ssd_300_vgg16_atrous_voc_int8</strong> is a quantized model calibrated on Pascal VOC dataset for <strong>333333</strong>strong>.</li>
                </ul>
            </div>)
        case 1:
            return (<div style={{backgroundColor:"#E4F4F6", marginTop:10, marginBottom:10, textAlign:'left', padding:15, }}>
                <b style={{fontSize:20, fontWeight:600}}>Model attributes are coded in their names. For instance, ssd_300_vgg16_atrous_voc consists of four parts:</b>
                <ul type="circle" >
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>ssd</strong> indicate the algorithm is “Single Shot Multibox Object Detection” </li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>300</strong> is the training image size, which means training images are resized to 300x300 and all anchor boxes are designed to match this shape. This may not apply to some models.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>vgg16_atrous</strong> is the type of base feature extractor network.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>voc</strong> is the training dataset. You can choose voc or coco, etc.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>(320x320)</strong> indicate that the model was evaluated with resolution 320x320. If not otherwise specified, all detection models in GluonCV can take various input shapes for prediction. Some models are trained with various input data shapes, e.g., Faster-RCNN and YOLO models.</li>
                    <li style={{fontSize:15, fontWeight:500}} type="disc"><strong>ssd_300_vgg16_atrous_voc_int8</strong> is a quantized model calibrated on Pascal VOC dataset for <strong>333333</strong>strong>.</li>
                </ul>
            </div>)
    }
} ;

//E4F4F6
