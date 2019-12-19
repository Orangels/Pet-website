let a = 'pet.{task}.ops接口\n' +
    ' \n' +
    '通过pet.{task}.ops接口可以实例化一些不同视觉任务中所使用的特殊操作符，可以通过Python的import功能直接引入。\n' +
    ' \n' +
    'pet.rcnn.ops接口\n' +
    '=====================================================\n' +
    '\n' +
    '> class pet.rcnn.ops.DeformRoIPooling()\n' +
    '\n' +
    '> class pet.rcnn.ops.DeformRoIPoolingPack()\n' +
    '\n' +
    '> class pet.rcnn.ops.ModulatedDeformRoIPoolingPack()\n' +
    '\n' +
    '> class pet.rcnn.ops.ROIAlign(output_size, spatial_scale, sampling_ratio, aligned)  [源代码]()\n' +
    '\n' +
    '父类：torch.nn.modules.module.Module\n' +
    '\n' +
    '> __init__(output_size, spatial_scale, sampling_ratio, aligned)  [源代码]()\n' +
    '\n' +
    '参数：\n' +
    '\n' +
    '* **output_size**（int）- ROIAlign输出的ROI特征图的尺寸。\n' +
    '* **spatial_scale**（int）- 当前特征图的下采样倍率。\n' +
    '* **sampling_ratio**（int）- ROIAlign操作中每一个bin中所划分的grid的数量。\n' +
    '* **aligned**（bool）- 是否在采样之前对ROI的坐标进行对齐，参考[detectron2]()的实现。\n' +
    '\n' +
    '>> forward(input, rois)\n' +
    '\n' +
    '进行前向计算。\n' +
    '\n' +
    '输入：\n' +
    '\n' +
    '* **input**（tensor）- 特性图，维度(N, C, H, W)。\n' +
    '* **rois**（tensor) - ROI，维度(n, 5)。\n' +
    '\n' +
    '> class pet.rcnn.ops.ROIPool(output_size, spatial_scale)  [源代码]()\n' +
    '\n' +
    '父类：torch.nn.modules.module.Module\n' +
    '\n' +
    '> __init__(output_size, spatial_scale)  [源代码]()\n' +
    '\n' +
    '参数：\n' +
    '\n' +
    '* **output_size**（int）- ROIAlign输出的ROI特征图的尺寸。\n' +
    '* **spatial_scale**（int）- 当前特征图的下采样倍率。\n' +
    '\n' +
    '>> forward(input, rois)\n' +
    '\n' +
    '进行前向计算，输入与ROIAlign相同。\n' +
    '\n' +
    '\n' +
    'pet.instance.ops接口\n' +
    '=====================================================\n' +
    '\n' +
    '> class pet.instance.ops.JointsMSELoss()\n' +
    '\n' +
    '父类：torch.nn.modules.module.Module\n' +
    '\n' +
    '继承 `torch.nn.MSELoss`，针对人体关键点改进的关键点热力图的损失函数。\n' +
    '\n' +
    '> __init__()  [源代码]()\n' +
    '\n' +
    '> forward(output, target, target_weight=None)  [源代码]()\n' +
    '\n' +
    '累计人体关键点的偏差。\n' +
    '\n' +
    '输入：\n' +
    '\n' +
    '* **output**（tensor）- keypoint_head的输出热图。\n' +
    '* **target**（tensro）- 关键点的真值分布热力图。\n' +
    '* **target_weight**（tensor）- 根据可见性给每个关键点分配的权重。\n' +
    '\n' +
    '输出：\n' +
    '\n' +
    '根据关键数量平均后的损失函数。\n' +
    '\n' +
    '\n' +
    'pet.ssd.ops接口\n' +
    '=====================================================\n' +
    '\n' +
    '> class pet.ssd.ops.MultiBoxLoss(num_classes, overlap_thresh, neg_pos, fp16=False)  [源代码]()\n' +
    '\n' +
    '父类：torch.nn.modules.module.Module\n' +
    '\n' +
    'SSD算法计算框分类与位置回归的损失函数。通过将SSD的prior box和真值的iou匹配完成了监督信息的计算，包括分类标签和回归的偏移量，进一步计算损失函数，并且实现了在线困难样本挖掘。\n' +
    '\n' +
    '>> __init__(num_classes, overlap_thresh, neg_pos, fp16=False)  [源代码]()\n' +
    '\n' +
    '参数：\n' +
    '\n' +
    '* **num_classes**（int）- 框分类的类别数。\n' +
    '* **overlap_thresh**（float）- 进行框的匹配时的iou阈值。\n' +
    '* **negpos_ratio**（float）- 在线困难样本挖掘时采样的困难样本比例。\n' +
    '* **fp16**（bool）- 是否使用半精度浮点运算，默认：False。\n' +
    '\n' +
    '>> forward(predictions, priors, targets)  [源代码]()\n' +
    '\n' +
    '输入：\n' +
    '\n' +
    '* **predictions**（tensor）- SSD网络输出的分类与回归特征图。\n' +
    '* **priors**（tensor）- SSD在各级特征图上生成的预选框。\n' +
    '* **targets**（tensor）- 真值目标框。\n' +
    '\n' +
    '> class pet.ssd.ops.GiouLoss(pred_mode=\'Center\', size_sum=True, variances=None)   [源代码]()\n' +
    '\n' +
    '父类：torch.nn.modules.module.Module\n' +
    '\n' +
    '实现了优化检测框IOU的位置回归损失函数——GIOU loss。\n' +
    '\n' +
    '+++++++++++++++++++++++++++++++++++++++\n' +
    'NOTE\n' +
    '最终的loss是由每个minibatch的loss相加而得。\n' +
    '+++++++++++++++++++++++++++++++++++++++\n' +
    '\n' +
    '>> __init__(pred_mode=\'Center\', size_sum=True, variances=None)   [源代码]()\n' +
    '\n' +
    '参数：\n' +
    '\n' +
    '* **pred_mode**（str）- 预测框的坐标形制，默认：\'Center\'（[x, y, w, h]），\'Corner\'对应（[x1, y1, x2, y2]）。\n' +
    '* **size_sum**（bool）- 不同minibatch之上loss的收集方式，默认：True（相加），否则为求均值。\n' +
    '* **variances**（list）- 将预测的位移偏差加到预选框上时的方差。\n' +
    '\n' +
    '>> forward(loc_p, loc_t, prior_data)   [源代码]()\n' +
    '\n' +
    '输入：\n' +
    '\n' +
    '* **loc_p**（tensor）- 预测的位置偏差。\n' +
    '* **loc_t**（tensor）- 目标的真值框。\n' +
    '* **prior_data**（tensor）- SSD算法中设置的预选框。\n' +
    '\n' +
    '> class pet.ssd.ops.PriorBox(prior, scale=None)   [源代码]()\n' +
    '\n' +
    '父类：Python object\n' +
    '\n' +
    'SSD算法中，此类以（[x, y, w, h]）的形式计算了每一个输入特征图上的预选框。\n' +
    '\n' +
    '+++++++++++++++++++++++++++++++++++++++\n' +
    'NOTE\n' +
    '预选框的生成方式跟随SSD论文的版本发生变化，PriorBox保留了两个版本，V2是被使用最多、也是最新的版本。\n' +
    '+++++++++++++++++++++++++++++++++++++++\n' +
    '\n' +
    '>> __init__(prior, scale=None)\n' +
    '\n' +
    '参数：   \n' +
    '\n' +
    '* **prior**（AttrDict[源代码]()）- 用于生成预选框的配置参数字典。\n' +
    '* **scale**（list）- 长度为2的list，分别代表用于训练SSD的输入图像的长和宽（须相等）。\n' +
    '\n' +
    '>> forward()\n' +
    '\n' +
    '计算预选框。\n' +
    '\n' +
    '输出:\n' +
    '\n' +
    '* 在所有分辨率的特征图上计算得到的预选框，维度（[N, 4]）。\n' +
    '\n' +
    '> class pet.ssd.ops.Detect(num_classes, bkg_label, variance, object_score=0)   [源代码]()\n' +
    '\n' +
    '父类：torch.autograd.Function\n' +
    '\n' +
    '本类在测试阶段将SSD的网络输出进行解码，得到最终的检测结果。 \n' +
    '\n' +
    '>> __init__(num_classes, bkg_label, variance, object_score=0)   [源代码]()\n' +
    '\n' +
    '参数：\n' +
    '\n' +
    '* **num_classes**（int）- 分类类别数量。\n' +
    '* **bkg_label**（int）- 背景类别标签，默认：0。\n' +
    '* **variance**（list of float）- 解码输出框是的方差。\n' +
    '* **object_score**（）- 区分没有可能存在物体位置的得分阈值，默认：0。\n' +
    '\n' +
    '>> forward(predictions, prior, arm_data=None)   [源代码]()\n' +
    '\n' +
    '解码SSD网络输出，计算得到最终的检测结果，支持[RefineDet]()的相关操作\n' +
    '\n' +
    '输入：\n' +
    '\n' +
    '* **predictions**（list of tensor）- 将预测的类别特征图和位置特征图打包成的列表。\n' +
    '* **prior**（tensor）- 由[PriorBox]()计算得到的预选框。\n' +
    '\n' +
    '输出：\n' +
    '\n' +
    '* **boxes、scores**（tensor）- 解码网络输出得到的预测框和得分，维度分别为（[batch, N, 4]）、（[batch, N, num_cls]）。\n' +
    '\n' +
    '\n' +
    'pet.face.ops接口\n' +
    '=====================================================\n' +
    '\n' +
    '> class pet.face.ops.FocalLoss()   [源代码]()\n' +
    '\n' +
    '父类：torch.nn.modules.module.Module\n'