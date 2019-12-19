export let MSCOCO_Mask_RCNN = {
    key: 'MSCOCO_Mask_RCNN',
    dataSource: [
        {title:'在MSCOCO2017数据集上训练Mask R-CNN模型'},
        {text:'本教程将介绍使用Pet训练以及测试Mask R-CNN模型进行目标检测的主要步骤，在此我们会指导您如何通过组合Pet的提供的各个功能模块来构建Mask R-CNN模型，在此我们仅讲解组件的调用，部分实现细节请查阅系统组件的相应部分。在阅读本教程的之前我们强烈建议您阅读原始论文[Faster R-CNN](https://arxiv.org/abs/1506.01497v3)\\[1\\]、[FPN](https://arxiv.org/abs/1612.03144v2)\\[2\\]和[Mask R-CNN](https://arxiv.org/abs/1703.06870v3)\\[3\\]以了解更多关于Mask R-CNN的算法原理。\n'},
        {
            note:[
                {text:'首先参阅[CIHP数据准备](https://github.com/BUPT-PRIV/Pet-DOC/blob/master/%E6%95%99%E7%A8%8B/%E5%88%9D%E7%BA%A7%E6%95%99%E7%A8%8B/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87.md#cihp%E6%95%B0%E6%8D%AE%E9%9B%86)教程并在硬盘上准备好CIHP数据集。\n'}
            ]
        },
        {text:'如果您具有丰富的目标检测算法的研究经验，您也可以直接在Pet中运行`$Pet/tools/rcnn/train_net.py`脚本利己开始训练您的Mask R-CNN模型.\n' +
                '\n' +
                '用法示例：\n'},
        {ul:'在8块GPU上使用`coco_2017_train`训练一个端到端的Mask R-CNN模型，使用两个全连接层作为`RCNN`的功能网络：'},
        {shell:'```\n' +
                '    cd $Pet\n' +
                '\n' +
                '    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch tools/rcnn/train_net.py --cfg cfgs/rcnn/mscoco/e2e_mask_rcnn_R-50-FPN_1x.yaml\n' +
                '```\n'},
        {ul:'在8块GPU上使用`coco_2017_val`数据集上测试训练的Mask R-CNN模型：'},
        {shell:'```\n' +
                '    cd $Pet\n' +
                '\n' +
                '    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/rcnn/test_net.py --cfg cfgs/rcnn/mscoco/e2e_mask_rcnn_R-50-FPN_1x.yaml\n' +
                '```\n'},
        {text:'在进行任何与模型训练和测试有关的操作之前，需要先选择一个指定的`yaml`文件，明确在训练时候对数据集、模型结构、优化策略以及其他重要参数的需求与设置，本教程以Pet/cfgs/rcnn/mscoco/e2e_mask_rcnn_R-50-FPN_1x.yaml为例，讲解训练过程中所需要的关键配置，该套配置将指导此Mask R-CNN模型训练以及测试的全部步骤和细节，全部参数设置请见[$Pet/cfgs/rcnn/mscoco/e2e_mask_rcnn_R-50-FPN_1x.yaml]()。\n'},
        {part_title:'数据载入'},
        {text:'确保MSCOCO2017数据集已经存放在您的硬盘中，并按照[数据制备]()中的文件结构整理好MSCOCO数据集的文件结构，接下来我们可以开始加载`coco_2017_train`训练集。\n' +
                '\n' +
                '```Python\n' +
                '    # Create data loder\n' +
                '    datasets = build_dataset(cfg.TRAIN.DATASETS, is_train=True)\n' +
                '    train_loader = make_train_data_loader(\n' +
                '        datasets,\n' +
                '        is_distributed=args.distributed,\n' +
                '        start_iter=scheduler.iteration,\n' +
                '    )\n' +
                '```\n'},
        {ul:'使用MSCOCO数据集训练Faster R-CNN、Mask R-CNN任务时，将输入图像的短边缩放到800像素，同时保证长边不超过1333像素，这样做的目的是要保证输入图像的长宽比不失真'},
        {
            block:{
                title:'Faster R-CNN和Mask R-CNN的训练尺寸',
                children:[
                    {text:'在最早流行的目标检测模型的训练中，数据集内的图像在进入网络之前，其尺寸被按照短边600像素，长边不超过1000像素进行缩放，目的也是为了保证图像\n' +
                            '中的视觉目标长宽比不失真，这种做法在很长时间内被用于在PASCAL VOC以及MSCOCO数据集上训练Faster R-CNN模型。PASCAL VOC数据集的图片尺寸平\n' +
                            '均大小为384像素x500像素，且图像中视觉目标大多尺寸较大，而MSCOCO数据集图像中的目标数量大幅增加，同时MSCOCO数据集中大多数的目标的像素数不\n' +
                            '足图片像素数1%，这使得在MSCOCO数据集上进行目标检测的难度要远远高于在PASCAL VOC数据集上进行目标检测。\n'},
                    {img:'voc_coco_image'},
                    {text:'随着目标检测算法以及卷积神经网络的发展，目标检测模型的精度越来越高，对于不同尺度，尤其是小物体的检测效果越来越受到重视，因此MSCOCO数据集被\n' +
                            '更加普遍的用来评估模型精度，在[FPN]()中提到在MSCOCO数据集上训练目标检测模型时增大输入图像的尺寸可以提升小目标的检测效果。原理很简单，在目\n' +
                            '前流行的Faster R-CNN、FPN等anchor-based检测算法中，需要通过主干网络对输入图像不断进行下采样，在经过16倍下采样之后，原始图像中某些尺寸很\n' +
                            '小的目标所保留的视觉信息已经所剩无几，适当得提升训练图像尺寸可以在一定程度上在下采样过程中保留小目标，因此从FPN开始，在MSCOCO数据集上训练\n' +
                            '目标检测模型的输入图像尺寸被按照短边800像素，长边不超过1333像素进行缩放。\n'},
                ]
            }
        },
        {ul:'Mask R-CNN还对训练数据进行了随机水平翻转来进行数据增广，提升模型的泛化性，经过变换的图像以及其标注的可视化结果如下图：'},
        {img:'mask_aug'},
        {ul:'数据载入组件不只是完成了图像数据以及标注信息的读取，还在采集每个批次的数据的同时生成了RPN网络的训练标签，数据载入组件输出的每个批次的数据中包含图片数据、图片中物体的类别，物体的包围框、以及与物体数量相同的分割掩模（每个掩模只包含一个目标的前景掩码）。'},
        {shell:'```\n' +
                '    data: (1, 3, 800, 1196)\n' +
                '    label: (1, 6)\n' +
                '    box: (1, 6, 4)\n' +
                '    mask: (1, 6, 800, 1196)\n' +
                '```\n'},
        {part_title: 'Mask R-CNN网络结构'},
        {text:'在Pet中，Mask R-CNN网络使用`Generalized_RCNN`模型构建器来搭建，`Generalized_RCNN`与`Generalized_CNN`、`Generalized_SSD`一样用来搭建完整的计算机视觉算法网络结构，并遵循Pet的模块化构建网络的规则。我们在`yaml`文件中设置如下参数来使用配置系统构建Mask R-CNN网络：\n'},
        {yaml:'```\n' +
                '    MODEL:\n' +
                '      FPN_ON: True\n' +
                '      MASK_ON: True\n' +
                '      NUM_CLASSES: 81\n' +
                '      CONV1_RGB2BGR: False  # caffe style\n' +
                '    BACKBONE:\n' +
                '      CONV_BODY: "resnet"\n' +
                '      RESNET:  # caffe style\n' +
                '        LAYERS: (3, 4, 6, 3)\n' +
                '```\n'},
        {text:'```Python\n' +
                '    class Generalized_RCNN(nn.Module):\n' +
                '        def __init__(self):\n' +
                '            super().__init__()\n' +
                '\n' +
                '            # Backbone for feature extraction\n' +
                '            conv_body = registry.BACKBONES[cfg.MODEL.CONV_BODY]\n' +
                '            self.Conv_Body = conv_body()\n' +
                '            self.dim_in = self.Conv_Body.dim_out\n' +
                '            self.spatial_scale = self.Conv_Body.spatial_scale\n' +
                '\n' +
                '            # Feature Pyramid Networks\n' +
                '            if cfg.FPN.FPN_ON:\n' +
                '                self.Conv_Body_FPN = FPN.fpn(self.dim_in, self.spatial_scale)\n' +
                '                self.dim_in = self.Conv_Body_FPN.dim_out\n' +
                '                self.spatial_scale = self.Conv_Body_FPN.spatial_scale\n' +
                '            else:\n' +
                '                self.dim_in = self.dim_in[-1]\n' +
                '                self.spatial_scale = [self.spatial_scale[-1]]\n' +
                '\n' +
                '            # Region Proposal Network\n' +
                '            self.RPN = build_rpn(self.dim_in)\n' +
                '\n' +
                '            if not cfg.MODEL.RETINANET_ON:\n' +
                '                self.Fast_RCNN = FastRCNN(self.dim_in, self.spatial_scale)\n' +
                '\n' +
                '                if cfg.MODEL.MASK_ON:\n' +
                '                    self.Mask_RCNN = MaskRCNN(self.dim_in, self.spatial_scale)\n' +
                '```\n'},
        {text:'与`Generalized_CNN`略有不同，`Generalized_RCNN`对Mask R-CNN网络结构的划分主要有以下几点不同：\n'},
        {ul:'除了特征提取网络、功能网络中、输出分支等三个网络模块之外，Mask R-CNN网络在`Generalized_RCNN`中还包括**区域建议网络**（RPN）；'},
        {
            block:{
                title:'RPN',
                children:[
                    {text:'RPN将特征图上可能出现前景物体的区域提取出来作为候选框，其输入为特征图，输出为一系列矩形候选框。RPN的实现原理如下图，根据特征图（HxW）的下\n' +
                            '采样倍率预设N个不同尺寸、不同长宽比的矩形窗口，以输入的特征图上的每一个像素位置为中心铺设N个窗口，这样在特征图上得到HxWxN个候选框，RPN网\n' +
                            '络进行区域生成的本质是滑窗方法。在生成大量的候选框之后，根据每一个候选框与物体标注框之间的交并比来区分前景和背景候选框，生成RPN训练标签，包\n' +
                            '括每个前景候选框的类别和坐标修正值。\n'},
                    {img:'RPN'},
                    {text:'RPN对输入特征图上的每一个像素点分别预测其上铺设候选框的两个类别以及四个坐标修正值，与数据载入时计算出来的RPN标签做偏差，进行RPN的训练。被\n' +
                            '修正位置的前景和背景候选框被到`RCNN`网络进行RoIAlign，进一步地分类以及回归。\n'},
                ]
            }
        },
        {ul:'FPN结构（如果需要）被归纳于特征提取网络模块中，在基础特征提取网络构建之后，FPN结构被构建于特征提取网络之上，被称为`Conv_Body_FPN`；'},
        {
            block:{
                title:'特征金字塔网络（FPN）',
                children:[
                    {text:'目标检测模型需要对图像中不同尺寸的目标进行定位与分类，但是以Faster R-CNN为主的两阶段目标检测器大多在16倍下采样之后的特征图上铺设锚框进行目\n' +
                            '标检测，经过多次下采样之后，小目标的信息已经所剩无几，因此Faster R-CNN对于小目标的检测效果一直有待提高。Faster R-CNN、R-FCN等一系列方法\n' +
                            '普遍采用多尺度测试和训练策略，即图像金字塔，虽然能够一定程度上提升对于小目标的检测效果，但是随之而来的计算开销也是巨大的。\n'},
                    {text:'在单阶段检测器中，SSD为了检测不同尺寸的目标，同时不增加计算开销，在特征提取网络中的多级特征图上铺设预选框，利用了特征金字塔进行目标检测取得\n' +
                            '了一定的效果，但是以SSD为代表的单阶段检测器由于底层特征的语义信息的缺乏，检测效果始终逊与两阶段检测器。特征金字塔在两阶段检测器中的使用受阻\n' +
                            '主要原因在于，特征金字塔和RPN网络结构的工程实现难度较大，直到FPN的出现才解决这一问题。\n'},
                    {img:'FPN'},
                    {text:'FPN在两阶段检测器中利用了特征金字塔，同时在进行区域建议之前对不同分辨率的特征进行了融合，补强了底层特征图的语义信息，使用FPN结构的两阶段检\n' +
                            '测器在多个分辨率特征图上进行区域建议，对所有RPN提取的候选框根据其大小重新将之分配在特定层级的特征图上进行RoIAlign。使用FPN结构可以使两阶\n' +
                            '段检测器的目标检测效果稳定提升1%，在FPN出现之后，单阶段检测器也普遍使用了FPN结构来提升检测精度，具有代表性的工作如[RetinaNet](https://arxiv.org/abs/1708.02002v2)\\[4\\]。\n'},
                ]
            }
        },
        {ul:'功能网络中模块包含**检测分支**（FastRCNN）与**实例分割分支**（MaskRCNN），`RoIAlign`、不同的子卷积神经网络以及相应的损失函数均被构建在相应的功能网络中。'},
        {
            block:{
                title:'RoI Align',
                children:[
                    {text:'Faster R-CNN由于对RCNN进行了大量的计算共享，因此在整个网络前向计算过程中需要一个GPU操作将候选区域从特征图上截取出来生成区域特征图，至此\n' +
                            'RoI Pooling操作被Faster R-CNN提出。RoI Pooling操作将每个RoI在特征图上的区域分为N x N个方格，N为RoI Pooling后生成区域特征图的尺寸。\n' +
                            '在每个方格中取最大值为区域特征图中该方格的像素值。\n'},
                    {text:'在这个过程中会有两次量化操作。对于一个region proposal，首先从原图经过全卷积网络到特征图，得到的候选框位置可能存在浮点数，进行取整操作从\n' +
                            '而出现第一次量化；其次，在ROI Pooling求取每个小网格的位置时也同样存在浮点数取整的情况。这两次量化的结果都使得候选框的位置会出现偏差，在\n' +
                            '论文里这种现象被总结为RoI Pooling的“像素不匹配问题”。如下图所示，假设图片经过主干网络提取特征后，特征图缩放步长（stride）为32，则该层\n' +
                            '特征图上0.1个像素的偏差，缩放到原图就是3.2个像素。\n'},
                    {text:'RoI Pooling中的量化操作对于小目标的定位精度有着很大的影响，为了解决这一问题，Mask R-CNN提出了ROI Align这一改进的方法。ROI Align的思\n' +
                            '路很简单：取消量化操作，使用双线性内插的方法获得坐标为浮点数的像素点上的图像数值,从而将整个特征聚集过程转化为一个连续的操作。值得注意的是，\n' +
                            '在具体的算法操作上，ROI Align并不是简单地补充出候选区域边界上的坐标点，然后将这些坐标点进行池化，而是遍历每一个候选区域，保持浮点数边界不\n' +
                            '做量化。将候选区域分割成N x N个单元，每个单元的边界也不做量化。在每个单元中计算固定四个坐标位置，用双线性内插的方法计算出这四个位置的值，\n' +
                            '然后进行最大池化操作。\n'},
                    {text:'如下表所示，在ResNet50的C5 block上使用RoI Align操作代替RoI Pooling，可以使Mask-rcnn的`maskAP`与`boxAP`均有很明显的提升。\n'},
                    {
                        table:{
                            titles:['Method','maskAP','maskAP50','maskAP75','boxAP','boxAP50','boxAP75'],
                            data:[
                                ['RoI Pooling',23.6,46.5,21.6,28.2,52.7,26.9],
                                ['RoI Align','30.9(+7.3)','52.8(+5.3)','32.1(+10.5)','34.0(+5.8)','55.3(+2.6)','36.4(+9.5)']
                            ]
                        }
                    }
                ]
            }
        },
        {
            block:{
                title:'基于区域的多任务学习',
                children:[
                    {text:'Mask R-CNN在Faster R-CNN的基础上添加了实例分割分支，同时进行两个实例分析任务的学习，下表中是Pet下Faster R-CNN与Mask R-CNN在\n' +
                            'MSCOCO2017_train上训练得到的模型在MSCOCO2017_val上进行评估所得精度对比，可以看出在加入实例分割任务之后，目标检测任务的精度也得到了可\n' +
                            '观的提升。\n'},
                    {
                        table:{
                            titles:['Method','Backbone','boxAP','maskAP'],
                            data:[
                                ['Faster R-CNN','R-50-FPN',36.4,'-'],
                                ['Mask R-CNN','R-50-FPN',37.4,34.2]
                            ]
                        }
                    },
                    {text:'在[Mask R-CNN]()论文中也展示了不同任务之间相互影响的退化实验。如下表所示，使用ResNet50作为backbone在MSCOCO2017_train训练得到的模型在       MSCOCO2017_val集上进行评估，在`person`这一类别上进行对比，`FastRCNN`、`MaskRCNN`与`KeypointRCNN`三个任务在训练过程中的损失函数拥有       同样的权重。可知将`MaskRCNN`分支添加到Faster R-CNN或KeyPoint-RCNN上一致地提升了模型在这些任务上的精度，但添加`KeypointRCNN`在\n' +
                            'Faster R-CNN或Mask R-CNN上会略微降低`boxAP`和`maskAP`，这表明虽然关键点检测受益于多任务训练，但它不会反过来帮助其他任务。更多的实验表明       不同的实例分析任务之间是有联系的，某些任务之间会相互促进，有些任务共同训练则会产生负面的影响。\n'},
                    {
                        table:{
                            titles:['Method','boxAP(person)','maskAP(person)','kpAP(person)'],
                            data:[
                                ['Faster R-CNN',52.5,'-','-'],
                                ['Mask R-CNN,mask-only',53.6,45.8,'-'],
                                ['Mask R-CNN, keypoint-only',50.7,'-',64.2],
                                ['Mask R-CNN, keypoint & mask',52.0,45.1,64.7]
                            ]
                        }
                    }

                ]
            }
        },
        {part_title:'训练'},
        {text:'完成了数据载入以及模型构建之后，我们需要在开始训练之前选择训练Mask R-CNN模型的优化策略，遵循[Mask R-CNN]()的思想，在批次大小为16的情况下，设置初始学习率为0.02，训练900000次迭代，组合使用了学习率预热与阶段下降策略，分别在60000与80000次迭代时将学习率减小十倍。\n'},
        {text:'```Python\n' +
                '    def train(model, loader, optimizer, scheduler, checkpointer, logger):\n' +
                '        # switch to train mode\n' +
                '        model.train()\n' +
                '        device = torch.device(\'cuda\')\n' +
                '\n' +
                '        # main loop\n' +
                '        cur_iter = scheduler.iteration\n' +
                '        for iteration, (images, targets, _) in enumerate(loader, cur_iter):\n' +
                '            logger.iter_tic()\n' +
                '            logger.data_tic()\n' +
                '\n' +
                '            scheduler.step()    # adjust learning rate\n' +
                '            optimizer.zero_grad()\n' +
                '\n' +
                '            images = images.to(device)\n' +
                '            targets = [target.to(device) for target in targets]\n' +
                '            logger.data_toc()\n' +
                '\n' +
                '            outputs = model(images, targets)\n' +
                '\n' +
                '            logger.update_stats(outputs, args.distributed, args.world_size)\n' +
                '            loss = outputs[\'total_loss\']\n' +
                '            loss.backward()\n' +
                '            optimizer.step()\n' +
                '\n' +
                '            if args.local_rank == 0:\n' +
                '                logger.log_stats(scheduler.iteration, scheduler.new_lr)\n' +
                '\n' +
                '                # Save model\n' +
                '                if cfg.SOLVER.SNAPSHOT_ITERS > 0 and (iteration + 1) % cfg.SOLVER.SNAPSHOT_ITERS == 0:\n' +
                '                    checkpointer.save(model, optimizer, scheduler, copy_latest=True, infix=\'iter\')\n' +
                '            logger.iter_toc()\n' +
                '        return None\n' +
                '\n' +
                '        # Train\n' +
                '        logging_rank(\'Training starts !\', distributed=args.distributed, local_rank=args.local_rank)\n' +
                '        train(model, train_loader, optimizer, scheduler, checkpointer, logger)\n' +
                '        logging_rank(\'Training done !\', distributed=args.distributed, local_rank=args.local_rank)\n' +
                '```\n' +
                '\n' +
                '在训练过程中，日志记录仪会在每若干次迭代后记录当前网络训练的迭代数、各项偏差数值等训练信息，检查点组件会定期保存网络模型到配置系统中`cfg.CKPT`所设置的路径下。\n'},
        {text:'根据`cfg.DISPLAY_ITER`设置的日志记录间隔，在训练过程中每经过20次迭代，日志记录仪会在终端中记录模型的训练状态。\n'},
        {shell:'```\n' +
                '[Training][e2e_mask_rcnn_R-50-FPN_1x.yaml][iter: 100/90000][lr: 0.005600][eta: 11:02:46]\n' +
                '\t  total_loss: 1.111744 (1.174495), iter_time: 0.4124 (0.4423), data_time: 0.1492 (0.1291)\n' +
                '\t  loss_box_reg: 0.286152 (0.318465), loss_mask: 0.215813 (0.206653), loss_classifier: 0.426412 (0.401559), loss_rpn_box_reg: 0.123295 (0.088334), loss_objectness: 0.246631 (0.234026)\n' +
                '[Training][e2e_mask_rcnn_R-50-FPN_1x.yaml][iter: 120/90000][lr: 0.006320][eta: 10:58:06]\n' +
                '\t  total_loss: 1.369676 (1.362034), iter_time: 0.4920 (0.4393), data_time: 0.1344 (0.1292)\n' +
                '\t  loss_box_reg: 0.286030 (0.315170), loss_mask: 0.145122 (0.208846), loss_classifier: 0.443223 (0.390104), loss_rpn_box_reg: 0.116009 (0.100111), loss_objectness: 0.249885 (0.248740)\n' +
                '[Training][e2e_mask_rcnn_R-50-FPN_1x.yaml][iter: 140/90000][lr: 0.007040][eta: 11:01:01]\n' +
                '\t  total_loss: 1.671880 (1.377465), iter_time: 0.5166 (0.4414), data_time: 0.1237 (0.1289)\n' +
                '\t  loss_box_reg: 0.352341 (0.323566), loss_mask: 0.217875 (0.215822), loss_classifier: 0.529664 (0.435546), loss_rpn_box_reg: 0.182061 (0.136870), loss_objectness: 0.385769 (0.291985)\n' +
                '```\n'},
        {part_title:'测试'},
        {text:'在完成Mask R-CNN模型的训练之后，我们使用[$Pet/tools/rcnn/test_net.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/tools/rcnn/test_net.py)在MSCOCO2017_val上评估模型的精度。同样需需要使用`Dataloader`来加载测试数据集，并对图像做同样尺度的缩放。\n'},
        {text:'通过加载训练最大迭代数之后的模型`$Pet/ckpts/rcnn/mscoco/e2e_mask_rcnn_R-50-FPN-600_0.5x/model_latest.pth`，执行下面的命令进行Mask R-CNN模型的测试，Mask R-CNN的测试日志同样会被`Logger`所记录。\n'},
        {shell:'```\n' +
                '    cd $Pet\n' +
                '\n' +
                '    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/rcnn/test_net.py --cfg cfgs/rcnn/mscoco/e2e_mask_rcnn_R-50-FPN_1x.yaml\n' +
                '```\n'},
        {part_title:'推理结果可视化'},
        {text:'在Pet中Mask R-CNN返回每一个目标的类别ID、置信度分数，边界框坐标和分割掩码。将MSCOCO2017_val中的一张图片的推理结果进行可视化如下图。\n'},
        {img:'test_mask_rcnn_000000151820'},
        {part_title:'参考文献'},
        {text:'\\[1\\] Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun. Faster {R-CNN}: Towards Real-Time Object Detection with Region Proposal Networks. NIPS 2015.\n'},
        {text:'\\[2\\] Tsung-Yi Lin and Piotr Dollár and Ross Girshick and Kaiming He and Bharath Hariharan and Serge Belongie. Feature Pyramid Networks for Object Detection. CVPR 2017.\n'},
        {text:'\\[3\\] Kaiming He and Georgia Gkioxari and Piotr Dollár and and Ross Girshick. Mask {R-CNN}. ICCV 2017.\n'},
        {text:'\\[4\\] Tsung-Yi Lin and Priya Goyal and Ross Girshick and Kaiming He and Piotr Dollár. Focal loss for dense object detection. CVPR 2018.\n'},

    ]
}