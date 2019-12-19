pet.{task}.ops接口
 
通过pet.{task}.ops接口可以实例化一些不同视觉任务中所使用的特殊操作符，可以通过Python的import功能直接引入。
 
pet.rcnn.ops接口
=====================================================

> class pet.rcnn.ops.DeformRoIPooling()

> class pet.rcnn.ops.DeformRoIPoolingPack()

> class pet.rcnn.ops.ModulatedDeformRoIPoolingPack()

> class pet.rcnn.ops.ROIAlign(output_size, spatial_scale, sampling_ratio, aligned)  [源代码]()

父类：torch.nn.modules.module.Module

> __init__(output_size, spatial_scale, sampling_ratio, aligned)  [源代码]()

参数：

* **output_size**（int）- ROIAlign输出的ROI特征图的尺寸。
* **spatial_scale**（int）- 当前特征图的下采样倍率。
* **sampling_ratio**（int）- ROIAlign操作中每一个bin中所划分的grid的数量。
* **aligned**（bool）- 是否在采样之前对ROI的坐标进行对齐，参考[detectron2]()的实现。

>> forward(input, rois)

进行前向计算。

输入：

* **input**（tensor）- 特性图，维度(N, C, H, W)。
* **rois**（tensor) - ROI，维度(n, 5)。

> class pet.rcnn.ops.ROIPool(output_size, spatial_scale)  [源代码]()

父类：torch.nn.modules.module.Module

> __init__(output_size, spatial_scale)  [源代码]()

参数：

* **output_size**（int）- ROIAlign输出的ROI特征图的尺寸。
* **spatial_scale**（int）- 当前特征图的下采样倍率。

>> forward(input, rois)

进行前向计算，输入与ROIAlign相同。


pet.instance.ops接口
=====================================================

> class pet.instance.ops.JointsMSELoss()

父类：torch.nn.modules.module.Module

继承 `torch.nn.MSELoss`，针对人体关键点改进的关键点热力图的损失函数。

> __init__()  [源代码]()

> forward(output, target, target_weight=None)  [源代码]()

累计人体关键点的偏差。

输入：

* **output**（tensor）- keypoint_head的输出热图。
* **target**（tensro）- 关键点的真值分布热力图。
* **target_weight**（tensor）- 根据可见性给每个关键点分配的权重。

输出：

根据关键数量平均后的损失函数。


pet.ssd.ops接口
=====================================================

> class pet.ssd.ops.MultiBoxLoss(num_classes, overlap_thresh, neg_pos, fp16=False)  [源代码]()

父类：torch.nn.modules.module.Module

SSD算法计算框分类与位置回归的损失函数。通过将SSD的prior box和真值的iou匹配完成了监督信息的计算，包括分类标签和回归的偏移量，进一步计算损失函数，并且实现了在线困难样本挖掘。

>> __init__(num_classes, overlap_thresh, neg_pos, fp16=False)  [源代码]()

参数：

* **num_classes**（int）- 框分类的类别数。
* **overlap_thresh**（float）- 进行框的匹配时的iou阈值。
* **negpos_ratio**（float）- 在线困难样本挖掘时采样的困难样本比例。
* **fp16**（bool）- 是否使用半精度浮点运算，默认：False。

>> forward(predictions, priors, targets)  [源代码]()

输入：

* **predictions**（tensor）- SSD网络输出的分类与回归特征图。
* **priors**（tensor）- SSD在各级特征图上生成的预选框。
* **targets**（tensor）- 真值目标框。

> class pet.ssd.ops.GiouLoss(pred_mode='Center', size_sum=True, variances=None)   [源代码]()

父类：torch.nn.modules.module.Module

实现了优化检测框IOU的位置回归损失函数——GIOU loss。

+++++++++++++++++++++++++++++++++++++++
NOTE
最终的loss是由每个minibatch的loss相加而得。
+++++++++++++++++++++++++++++++++++++++

>> __init__(pred_mode='Center', size_sum=True, variances=None)   [源代码]()

参数：

* **pred_mode**（str）- 预测框的坐标形制，默认：'Center'（[x, y, w, h]），'Corner'对应（[x1, y1, x2, y2]）。
* **size_sum**（bool）- 不同minibatch之上loss的收集方式，默认：True（相加），否则为求均值。
* **variances**（list）- 将预测的位移偏差加到预选框上时的方差。

>> forward(loc_p, loc_t, prior_data)   [源代码]()

输入：

* **loc_p**（tensor）- 预测的位置偏差。
* **loc_t**（tensor）- 目标的真值框。
* **prior_data**（tensor）- SSD算法中设置的预选框。

> class pet.ssd.ops.PriorBox(prior, scale=None)   [源代码]()

父类：Python object

SSD算法中，此类以（[x, y, w, h]）的形式计算了每一个输入特征图上的预选框。

+++++++++++++++++++++++++++++++++++++++
NOTE
预选框的生成方式跟随SSD论文的版本发生变化，PriorBox保留了两个版本，V2是被使用最多、也是最新的版本。
+++++++++++++++++++++++++++++++++++++++

>> __init__(prior, scale=None)

参数：   

* **prior**（AttrDict[源代码]()）- 用于生成预选框的配置参数字典。
* **scale**（list）- 长度为2的list，分别代表用于训练SSD的输入图像的长和宽（须相等）。

>> forward()

计算预选框。

输出:

* 在所有分辨率的特征图上计算得到的预选框，维度（[N, 4]）。

> class pet.ssd.ops.Detect(num_classes, bkg_label, variance, object_score=0)   [源代码]()

父类：torch.autograd.Function

本类在测试阶段将SSD的网络输出进行解码，得到最终的检测结果。 

>> \_\_init\_\_(num_classes, bkg_label, variance, object_score=0)   [源代码]()

参数：

* **num_classes**（int）- 分类类别数量。
* **bkg_label**（int）- 背景类别标签，默认：0。
* **variance**（list of float）- 解码输出框是的方差。
* **object_score**（）- 区分没有可能存在物体位置的得分阈值，默认：0。

>> forward(predictions, prior, arm_data=None)   [源代码]()

解码SSD网络输出，计算得到最终的检测结果，支持[RefineDet]()的相关操作

输入：

* **predictions**（list of tensor）- 将预测的类别特征图和位置特征图打包成的列表。
* **prior**（tensor）- 由[PriorBox]()计算得到的预选框。

输出：

* **boxes、scores**（tensor）- 解码网络输出得到的预测框和得分，维度分别为（[batch, N, 4]）、（[batch, N, num_cls]）。


pet.face.ops接口
=====================================================

> class pet.face.ops.FocalLoss()   [源代码]()

父类：torch.nn.modules.module.Module
