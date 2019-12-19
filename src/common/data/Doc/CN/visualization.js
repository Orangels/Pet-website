export let visualization_data = {
    key: 'Visualization',
    dataSource: [
        {title:'可视化'},
        {text:'可视化可以帮助我们直观的了解数据标注和模型推断结果。Pet根据每个任务不同的数据格式和需求，在`$Pet/pet/utils/vis.py`文件中封装了与不同任务对应的可视化工具。\n'},
        {part_title:'可视化内容'},
        {text:'这里给出Pet对于不同任务支持的可视化功能介绍。\n'},
        {table:{
                titles:['任务','支持可视化的功能'],
                data:[["rcnn","1、对图像中置信度高于阈值的实例上加外接框；\n"+
                "2、在图像指定位置上加类别标签字符；\n" +
                "3、对图像中置信度高于阈值的实例上加分割掩模；\n" +
                "4、对图像中置信度高于阈值的人物实例所有部位上加对应的分割掩模；\n" +
                "5、对图像中置信度高于阈值的人物实例加上人体部位关键点。\n" +
                "6、对图像中置信度高于阈值的人物实例加上密集姿态信息。"],["ssd","1、对图像中置信度高于阈值的实例上加外接框；\n" +
                "2、在图像指定位置上加类别标签字符。"],["pose","1、对图像中置信度高于阈值的人物实例上加外接框；\n" +
                "2、在图像指定位置上加类别标签字符；\n" +
                "3、对图像中置信度高于阈值的人物实例上加人体部位关键点。"],]
            }
            , className:'table_1',type:'start',table_width:530},
        {part_title: 'vis_one_image_opencv'},
        {text:'Pet使用`vis_one_image_opencv`函数来实现单张图像上所有置信度高于阈值实例的可视化，这里调用到了`get_instance_parsing_colormap`来获取图像的实例掩模颜色和人体各部位的掩模颜色集，再通过`vis_bbox`、`vis_class`、`vis_mask`、`vis_keypoints`、`vis_parsing`、`vis_uv`等函数来分别可视化图像中单个实例的外接框、类别标签、分割掩模、人物实例关键点、各部位掩模、密集姿态标注信息。\n' +
                '\n' +
                '```Python\n' +
                '    def vis_one_image_opencv(im, boxes, segms=None, keypoints=None,\n' +
                '                             parsing=None, uv=None, dataset=None):\n' +
                '        """Constructs a numpy array with the detections visualized."""\n' +
                '        timers = defaultdict(Timer)\n' +
                '        timers[\'bbox_prproc\'].tic()\n' +
                '\n' +
                '        if isinstance(boxes, list):\n' +
                '            boxes, segms, keypoints, parsing, uv, classes = convert_from_cls_format(\n' +
                '                boxes, segms, keypoints, parsing, uv)\n' +
                '        else:\n' +
                '            return im\n' +
                '\n' +
                '        if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < cfg.VIS.VIS_TH:\n' +
                '            return im\n' +
                '\n' +
                '        if segms is not None and len(segms) > 0:\n' +
                '            masks = mask_util.decode(segms)\n' +
                '\n' +
                '        # get color map\n' +
                '        ins_colormap, parss_colormap = get_instance_parsing_colormap()\n' +
                '\n' +
                '        # Display in largest to smallest order to reduce occlusion\n' +
                '        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])\n' +
                '        sorted_inds = np.argsort(-areas)\n' +
                '        timers[\'bbox_prproc\'].toc()\n' +
                '\n' +
                '        instance_id = 1\n' +
                '        for i in sorted_inds:\n' +
                '            bbox = boxes[i, :4]\n' +
                '            score = boxes[i, -1]\n' +
                '            if score < cfg.VIS.VIS_TH:\n' +
                '                continue\n' +
                '\n' +
                '            # get instance color (box, class_bg)\n' +
                '            if cfg.VIS.SHOW_BOX.COLOR_SCHEME == \'category\':\n' +
                '                ins_color = ins_colormap[classes[i]]\n' +
                '            elif cfg.VIS.SHOW_BOX.COLOR_SCHEME == \'instance\':\n' +
                '                instance_id = instance_id % len(ins_colormap.keys())\n' +
                '                ins_color = ins_colormap[instance_id]\n' +
                '            else:\n' +
                '                ins_color = _GREEN\n' +
                '            instance_id += 1\n' +
                '\n' +
                '            # show box (off by default)\n' +
                '            if cfg.VIS.SHOW_BOX.ENABLED:\n' +
                '                timers[\'show_box\'].tic()\n' +
                '                im = vis_bbox(im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), ins_color)\n' +
                '                timers[\'show_box\'].toc()\n' +
                '\n' +
                '            # show class (off by default)\n' +
                '            if cfg.VIS.SHOW_CLASS.ENABLED:\n' +
                '                timers[\'show_class\'].tic()\n' +
                '                class_str = get_class_string(classes[i], score, dataset)\n' +
                '                im = vis_class(im, (bbox[0], bbox[1] - 2), class_str, ins_color)\n' +
                '                timers[\'show_class\'].toc()\n' +
                '\n' +
                '            show_segms = True if cfg.VIS.SHOW_SEGMS.ENABLED and segms is not None and len(segms) > i else False\n' +
                '            show_kpts = True if cfg.VIS.SHOW_KPS.ENABLED and keypoints is not None and len(keypoints) > i else False\n' +
                '            show_parss = True if cfg.VIS.SHOW_PARSS.ENABLED and parsing is not None and len(parsing) > i else False\n' +
                '            show_uv = True if cfg.VIS.SHOW_UV.ENABLED and uv is not None and len(uv) > i else False\n' +
                '            # show mask\n' +
                '            if show_segms:\n' +
                '                timers[\'show_segms\'].tic()\n' +
                '                color_list = colormap_utils.colormap()\n' +
                '                im = vis_mask(im, masks[..., i], ins_color, show_parss=show_parss)\n' +
                '                timers[\'show_segms\'].toc()\n' +
                '\n' +
                '            # show keypoints\n' +
                '            if show_kpts:\n' +
                '                timers[\'show_kpts\'].tic()\n' +
                '                im = vis_keypoints(im, keypoints[i], show_parss=show_parss)\n' +
                '                timers[\'show_kpts\'].toc()\n' +
                '\n' +
                '            # show parsing\n' +
                '            if show_parss:\n' +
                '                timers[\'show_parss\'].tic()\n' +
                '                im = vis_parsing(im, parsing[i], parss_colormap, show_segms=show_segms)\n' +
                '                timers[\'show_parss\'].toc()\n' +
                '\n' +
                '            # show uv\n' +
                '            if show_uv:\n' +
                '                timers[\'show_uv\'].tic()\n' +
                '                im = vis_uv(im, uv[i], bbox)\n' +
                '                timers[\'show_uv\'].toc()\n' +
                '\n' +
                '        # for k, v in timers.items():\n' +
                '        #     print(\' | {}: {:.3f}s\'.format(k, v.total_time))\n' +
                '\n' +
                '        return im\n' +
                '```\n' +
                '\n' +
                '这里需要传入`im`、`boxes`、`classes`、`segms`、`keypoints`、`parsing`、`uv`、`dataset`等图像信息和标注信息。其中`segms`、`keypoints`、`parsing`、`uv`和`dataset`等变量加了空的预设值用来跳过缺省内容的可视化。另外函数会在开始时对数据外接框数量和置信度进行检测，在外接框数量为零或所有实例置信度都低于阈值的情况下会自动跳过当前图像的可视化，返回输入的原始图像。\n'},
        {table:{
                titles:['输入变量','内容','数据结构'],
                data:[["im","单张原始图像数据",'numpy'],["boxes","图像中所有实例的外接框坐标和得分",'list'],["classes","图像中所有实例的类别索引",'list'],["segms","图像中所有实例掩模的polygen编码",'list'],["keypoints","图像中所有人物实例的人体部位关键点坐标",'list'],["parsing","图像中所有人物实例的人体部位掩模",'list'],["uv","图像中所有人物实例的密集姿态",'list'],['dataset','包含数据集类别信息的类变量','class']]
            }
            , className:'table_2'},
        {h3_title:'get_instance_parsing_colormap'},
        {text:'Pet在[$Pet/pet/utils/colormap.py](https://github.com/BUPT-PRIV/Pet/blob/master/pet/utils/colormap.py)文件中定义了COCO、CIHP、VOC、ADE20K、MHP、CityScape等数据集的字典形式的颜色数据，通过`get_instance_parsing_colormap`函数将cfg参数中设置的数据集字典取出，并作为外接框、类别信息、实例分割和人体部位分割的颜色字典集合。在获取到数据集标准的颜色集合后，`vis_one_image_opencv`会通过cfg参数中的`COLOR_SCHEME`来确定当前实例可视化颜色的选取方式，选取方式可以是按照所属类别、按实例从获取的颜色集合中选取、或统一使用`$Pet/pet/utils/vis.py`中定义的绿色。\n'},
        {text:'```Python\n' +
                '    def get_instance_parsing_colormap(rgb=False):\n' +
                '        instance_colormap = eval(\'colormap_utils.{}\'.format(cfg.VIS.SHOW_BOX.COLORMAP))\n' +
                '        parsing_colormap = eval(\'colormap_utils.{}\'.format(cfg.VIS.SHOW_PARSS.COLORMAP))\n' +
                '        if rgb:\n' +
                '            instance_colormap = colormap_utils.dict_bgr2rgb(instance_colormap)\n' +
                '            parsing_colormap = colormap_utils.dict_bgr2rgb(parsing_colormap)\n' +
                '\n' +
                '        return instance_colormap, parsing_colormap\n' +
                '```\n'},
        {
            note:[
                {text:'在[$Pet/pet/utils/colormap.py](https://github.com/BUPT-PRIV/Pet/blob/master/pet/utils/colormap.py)文件中，Pet将各数据集的官方配色集按照BGR格式，以字典形式进行存放。\n'},
                {text:'颜色集合的名字是以数据集名称加数据集总的类别数来命名的。您在加入新的数据集配色集时，为了方便理解也建议使用这一标准。另外在给出数据集所对应官方颜色集合之前，加入了[Detectron](https://github.com/facebookresearch/Detectron/blob/8170b25b425967f8f1c7d715bea3c5b8d9536cd8/detectron/utils/colormap.py#L26)工程的配色方式的颜色集合`colormap`，使用字典的方式定义了79种颜色，使用corlormap可以为每个实例上加相应颜色的分割掩模。\n'}
            ]
        },
        {h3_title: 'vis_bbox'},
        {text:'参数说明：\n'},
        {text:'用于实现对单个实例外接框的可视化，需要传入`img`、`bbox`、`bbox_color`等参数。\n'},
        {
            ul:['`img`：以numpy数组格式存储的图片像素数据；',
                '`bbox`：当前实例外接框的左上角点的X、Y坐标和宽、高信息，形式是[x、y、w、h]；',
                '`bbox_color`：外接框要显示的颜色，形式是[r, g, b]。']
        },
        {text:'使用案例：\n'},

        {text:'```Python\n' +
                '    # show box (off by default)\n' +
                '    if cfg.VIS.SHOW_BOX.ENABLED:\n' +
                '        im = vis_bbox(im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), ins_color)\n' +
                '```\n'},
        {text:'可视化示例：'},
        {img:'Box_000000183648'},
        {h3_title:'vis_class'},
        {text:'参数说明：'},
        {text:'用于实现在相应位置上加实例的对应类别、分数等字符数据的可视化，需要传入`img`、`pos`、`class_str`、`bg_color`等参数。颜色可以通过`cfg.VIS.SHOW_CLASS.COLOR`参数修改，默认为白色。Pet又定义了`get_class_string`函数来方便用户获取要可视化的实例类别，置信度返回给`class_str`变量。\n'},
        {
            ul:['`img`：以numpy数组格式存储的图片像素数据；',
                '`pos`：两个元素的元组数据，用于表达可视化内容起始点的X、Y坐标，形式是（x、y）；',
                '`class_str`：要进行可视化的字符内容；',
                '`bg_color`：显示的背景色。'
            ]
        },
        {text: '```Python\n' +
                '    # show class (off by default)\n' +
                '    if cfg.VIS.SHOW_CLASS.ENABLED:\n' +
                '        class_str = get_class_string(classes[i], score, dataset)\n' +
                '        im = vis_class(im, (bbox[0], bbox[1] - 2), class_str, ins_color)\n' +
                '```\n'},
        {text:'可视化示例:'},
        {img:'Class_000000183648'},
        {h3_title:'vis_mask'},
        {text:'参数说明：\n'},
        {text:'用于实现对单个实例掩模的可视化，需要传入`img`、`mask`、`bbox_color`、`show_parss`等参数。\n'},
        {
            ul:['`img`：以numpy数组格式存储的图片像素数据；',
                '`mask`：和img具有相同长宽的单通道numpy；',
                '`bbox_color`：外接框要显示的颜色，形式是[r, g, b]；',
                '`show_parss`：标识变量，在人体各部位掩模执行可视化时跳过对整体实例的掩模可视化。'
            ]
        },
        {text:'```Python\n' +
                '    def vis_mask(img, mask, bbox_color, show_parss=False):\n' +
                '        """Visualizes a single binary mask."""\n' +
                '        img = img.astype(np.float32)\n' +
                '        idx = np.nonzero(mask)\n' +
                '\n' +
                '        border_color = cfg.VIS.SHOW_SEGMS.BORDER_COLOR\n' +
                '        border_thick = cfg.VIS.SHOW_SEGMS.BORDER_THICK\n' +
                '\n' +
                '        mask_color = bbox_color if cfg.VIS.SHOW_SEGMS.MASK_COLOR_FOLLOW_BOX else _WHITE\n' +
                '        mask_color = np.asarray(mask_color)\n' +
                '        mask_alpha = cfg.VIS.SHOW_SEGMS.MASK_ALPHA\n' +
                '\n' +
                '        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)\n' +
                '        if cfg.VIS.SHOW_SEGMS.SHOW_BORDER:\n' +
                '            cv2.drawContours(img, contours, -1, border_color, border_thick, cv2.LINE_AA)\n' +
                '\n' +
                '        if cfg.VIS.SHOW_SEGMS.SHOW_MASK and not show_parss:\n' +
                '            img[idx[0], idx[1], :] *= 1.0 - mask_alpha\n' +
                '            img[idx[0], idx[1], :] += mask_alpha * mask_color\n' +
                '\n' +
                '        return img.astype(np.uint8)\n' +
                '```\n'},
        {text:'函数介绍：'},
        {text:'传入变量后，我们通过获取`mask`中非零的前景掩模索引，然后将对应位置的像素和输入的颜色像素进行一定比例的加权求和，并覆盖原始像素得到掩模可视化结果，比例因子通过`cfg.VIS.SHOW_SEGMS.MASK_ALPHA`参数控制。\n'},
        {text:'使用案例：'},
        {text:'```Python\n' +
                '    # show mask\n' +
                '    if cfg.VIS.SHOW_SEGMS.ENABLED:\n' +
                '        color_list = colormap_utils.colormap()\n' +
                '        im = vis_mask(im, masks[..., i], ins_color, show_parss=show_parss)\n' +
                '\n' +
                '```\n'},
        {text:'可视化示例：'},
        {img:'000000183648_Mask'},
        {h3_title:'vis_keypoints'},
        {text:'参数说明：\n'},
        {text:'用于实现对单个人物实例关键点的可视化，需要传入`img`、`kps`、`show_parss`等参数。\n'},
        {
            ul:['`img`：以numpy数组格式存储的图片像素数据；',
                '`kps`：人体关键点标注，形式是[[x1, x2, ..., x17], [y1, y2, ..., y17], [c1, c2, ..., c17]]；',
                '`show_parss`：跳过可视化操作的标识变量，在人体各部分执行可视化时跳过对实例关键点的可视化。'
            ]
        },
        {text:'```Python\n' +
                '    def vis_keypoints(img, kps, show_parss=False):\n' +
                '        """Visualizes keypoints (adapted from vis_one_image).\n' +
                '        kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).\n' +
                '        """\n' +
                '        dataset_keypoints, _ = keypoint_utils.get_keypoints()\n' +
                '        kp_lines = kp_connections(dataset_keypoints)\n' +
                '\n' +
                '        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.\n' +
                '        cmap = plt.get_cmap(\'rainbow\')\n' +
                '        colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]\n' +
                '        if show_parss:\n' +
                '            colors = [cfg.VIS.SHOW_KPS.KPS_COLOR_WITH_PARSING for c in colors]\n' +
                '        else:\n' +
                '            colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]\n' +
                '\n' +
                '        # Perform the drawing on a copy of the image, to allow for blending.\n' +
                '        kp_mask = np.copy(img)\n' +
                '\n' +
                '        # Draw mid shoulder / mid hip first for better visualization.\n' +
                '        mid_shoulder = (kps[:2, dataset_keypoints.index(\'right_shoulder\')] +\n' +
                '                        kps[:2, dataset_keypoints.index(\'left_shoulder\')]) / 2.0\n' +
                '        sc_mid_shoulder = np.minimum(\n' +
                '            kps[2, dataset_keypoints.index(\'right_shoulder\')],\n' +
                '            kps[2, dataset_keypoints.index(\'left_shoulder\')])\n' +
                '        mid_hip = (kps[:2, dataset_keypoints.index(\'right_hip\')] +\n' +
                '                   kps[:2, dataset_keypoints.index(\'left_hip\')]) / 2.0\n' +
                '        sc_mid_hip = np.minimum(\n' +
                '            kps[2, dataset_keypoints.index(\'right_hip\')],\n' +
                '            kps[2, dataset_keypoints.index(\'left_hip\')])\n' +
                '        nose_idx = dataset_keypoints.index(\'nose\')\n' +
                '        if sc_mid_shoulder > cfg.VIS.SHOW_KPS.KPS_TH and kps[2, nose_idx] > cfg.VIS.SHOW_KPS.KPS_TH:\n' +
                '            cv2.line(kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]), color=colors[len(kp_lines)],\n' +
                '                     thickness=cfg.VIS.SHOW_KPS.LINK_THICK, lineType=cv2.LINE_AA)\n' +
                '        if sc_mid_shoulder > cfg.VIS.SHOW_KPS.KPS_TH and sc_mid_hip > cfg.VIS.SHOW_KPS.KPS_TH:\n' +
                '            cv2.line(kp_mask, tuple(mid_shoulder), tuple(mid_hip), color=colors[len(kp_lines) + 1],\n' +
                '                     thickness=cfg.VIS.SHOW_KPS.LINK_THICK, lineType=cv2.LINE_AA)\n' +
                '\n' +
                '        # Draw the keypoints.\n' +
                '        for l in range(len(kp_lines)):\n' +
                '            i1 = kp_lines[l][0]\n' +
                '            i2 = kp_lines[l][1]\n' +
                '            p1 = kps[0, i1], kps[1, i1]\n' +
                '            p2 = kps[0, i2], kps[1, i2]\n' +
                '            if kps[2, i1] > cfg.VIS.SHOW_KPS.KPS_TH and kps[2, i2] > cfg.VIS.SHOW_KPS.KPS_TH:\n' +
                '                cv2.line(kp_mask, p1, p2, color=colors[l],\n' +
                '                         thickness=cfg.VIS.SHOW_KPS.LINK_THICK, lineType=cv2.LINE_AA)\n' +
                '            if kps[2, i1] > cfg.VIS.SHOW_KPS.KPS_TH:\n' +
                '                cv2.circle(kp_mask, p1, radius=cfg.VIS.SHOW_KPS.CIRCLE_RADIUS, color=colors[l],\n' +
                '                           thickness=cfg.VIS.SHOW_KPS.CIRCLE_THICK, lineType=cv2.LINE_AA)\n' +
                '            if kps[2, i2] > cfg.VIS.SHOW_KPS.KPS_TH:\n' +
                '                cv2.circle(kp_mask, p2, radius=cfg.VIS.SHOW_KPS.CIRCLE_RADIUS, color=colors[l],\n' +
                '                           thickness=cfg.VIS.SHOW_KPS.CIRCLE_THICK, lineType=cv2.LINE_AA)\n' +
                '\n' +
                '        # Blend the keypoints.\n' +
                '        return cv2.addWeighted(img, 1.0 - cfg.VIS.SHOW_KPS.KPS_ALPHA, kp_mask, cfg.VIS.SHOW_KPS.KPS_ALPHA, 0)\n' +
                '```\n'},
        {text:'函数介绍：\n'},
        {text:'`vis_keypoints`函数中，不仅用到了原始的17个人体部位关键点，还对应计算了肩部中间点和胯部中间点来辅助完成人体关键点的连接。\n'},
        {text:'使用案例：\n'},
        {text:'```Python  \n' +
                '    # show keypoints\n' +
                '    if cfg.VIS.SHOW_KPS.ENABLED:\n' +
                '        im = vis_keypoints(im, keypoints[i], show_parss=show_parss)\n' +
                '```\n'},
        {text:'可视化示例：\n'},
        {img:'Keypoints_000000183648'},
        {h3_title:'vis_parsing'},
        {text:'参数说明：\n'},
        {text:'用于实现对单个人物实例所有部位的掩模可视化，也可以用于全景分割的掩模可视化，需要传入`img`、`parsing`、`colormap`、`show_segms`等参数。\n'},
        {
            ul:['`img`：numpy格式的图像；',
                '`parsing`：和img具有相同长宽的单通道numpy数组信息；',
                '`colormap`：字典格式的人体部位对应的颜色集合；',
                '`show_segms`：跳过可视化操作的标识变量，在执行整体实例的掩模可视化时跳过对人体各部位的可视化。'
            ]
        },
        {text:'```Python\n' +
                '    def vis_parsing(img, parsing, colormap, show_segms=True):\n' +
                '        """Visualizes a single binary parsing."""\n' +
                '        img = img.astype(np.float32)\n' +
                '        idx = np.nonzero(parsing)\n' +
                '\n' +
                '        parsing_alpha = cfg.VIS.SHOW_PARSS.PARSING_ALPHA\n' +
                '        colormap = colormap_utils.dict2array(colormap)\n' +
                '        parsing_color = colormap[parsing.astype(np.int)]\n' +
                '\n' +
                '        border_color = cfg.VIS.SHOW_PARSS.BORDER_COLOR\n' +
                '        border_thick = cfg.VIS.SHOW_PARSS.BORDER_THICK\n' +
                '\n' +
                '        img[idx[0], idx[1], :] *= 1.0 - parsing_alpha\n' +
                '        # img[idx[0], idx[1], :] += alpha * parsing_color\n' +
                '        img += parsing_alpha * parsing_color\n' +
                '\n' +
                '        if cfg.VIS.SHOW_PARSS.SHOW_BORDER and not show_segms:\n' +
                '            _, contours, _ = cv2.findContours(parsing.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)\n' +
                '            cv2.drawContours(img, contours, -1, border_color, border_thick, cv2.LINE_AA)\n' +
                '\n' +
                '        return img.astype(np.uint8)\n' +
                '```\n'},
        {text:'函数介绍：\n'},
        {text:'不同于`vis_mask`中将非零像素作为一个整体，只有一个非零取值范围，`vis_parsing`的掩模参数要包含多个像素值来对应人体的不同部位。因此，传入的颜色集合也要保证键值与`parsing`中各部位像素值对应，函数通过`parsing_color`来完成从掩模数据的像素点到类别对应颜色的映射。\n'},
        {text:'使用案例：\n'},
        {text:'```Python\n' +
                '    # show parsing\n' +
                '    if cfg.VIS.SHOW_PARSS.ENABLED:\n' +
                '        im = vis_parsing(im, parsing[i], parss_colormap, show_segms=show_segms)\n' +
                '```\n'},
        {text:'可视化示例：\n'},
        {img:'Parsing_000000183648'},
        {h3_title:'vis_uv'},
        {text:'参数说明：\n'},
        {text:'用于实现对单个人物实例密集姿态的可视化，需要传入`img`、`uv`、`bbox`等参数。\n'},
        {
            ul:['`img`：以numpy数组格式存储的图片像素数据；',
                '`uv`：当前实例的I、U、V信息，形式是尺寸为(3,56,56)的numpy数组；',
                '`bbox`：当前实例外接框的左上角X、Y坐标和宽、高信息，形式是[x1、y1、x2、y2]。']
        },
        {text:'```Python\n' +
                '    def vis_uv(img, uv, bbox):\n' +
                '        border_thick = cfg.VIS.SHOW_UV.BORDER_THICK\n' +
                '        grid_thick = cfg.VIS.SHOW_UV.GRID_THICK\n' +
                '        lines_num = cfg.VIS.SHOW_UV.LINES_NUM\n' +
                '\n' +
                '        uv = np.transpose(uv, (1, 2, 0))\n' +
                '        uv = cv2.resize(uv, (int(bbox[2] - bbox[0] + 1), int(bbox[3] - bbox[1] + 1)), interpolation=cv2.INTER_LINEAR)\n' +
                '        roi_img = img[int(bbox[1]):int(bbox[3] + 1), int(bbox[0]):int(bbox[2] + 1), :]\n' +
                '\n' +
                '        roi_img_resize = cv2.resize(roi_img, (2 * roi_img.shape[1], 2 * roi_img.shape[0]), interpolation=cv2.INTER_LINEAR)\n' +
                '\n' +
                '        I = uv[:, :, 0]\n' +
                '        for i in range(1, 25):\n' +
                '            if (len(I[I == i]) == 0):\n' +
                '                continue\n' +
                '\n' +
                '            u = np.zeros_like(I)\n' +
                '            v = np.zeros_like(I)\n' +
                '            u[I == i] = uv[:, :, 1][I == i]\n' +
                '            v[I == i] = uv[:, :, 2][I == i]\n' +
                '\n' +
                '            for ind in range(1, lines_num):\n' +
                '                thred = 1.0 * ind / lines_num\n' +
                '                _, thresh = cv2.threshold(u, u.min() + thred * (u.max() - u.min()), 255, 0)\n' +
                '                dist_transform = cv2.distanceTransform(np.uint8(thresh), cv2.DIST_L2, 0)\n' +
                '                dist_transform = np.uint8(dist_transform)\n' +
                '\n' +
                '                _, contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)\n' +
                '                contours = [(col * 2) for col in contours]\n' +
                '                cv2.drawContours(roi_img_resize, contours, -1, ((1 - thred) * 255, thred * 255, thred * 200), grid_thick)\n' +
                '\n' +
                '                _, thresh = cv2.threshold(v, v.min() + thred * (v.max() - v.min()), 255, 0)\n' +
                '                dist_transform = cv2.distanceTransform(np.uint8(thresh), cv2.DIST_L2, 0)\n' +
                '                dist_transform = np.uint8(dist_transform)\n' +
                '\n' +
                '                _, contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)\n' +
                '                contours = [(col * 2) for col in contours]\n' +
                '                cv2.drawContours(roi_img_resize, contours, -1, (thred * 255, (1 - thred) * 255, thred * 200), grid_thick)\n' +
                '\n' +
                '        _, thresh = cv2.threshold(I, 0.5, 255, 0)\n' +
                '        dist_transform = cv2.distanceTransform(np.uint8(thresh), cv2.DIST_L2, 0)\n' +
                '        dist_transform = np.uint8(dist_transform)\n' +
                '        _, contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)\n' +
                '        contours = [(col * 2) for col in contours]\n' +
                '        cv2.drawContours(roi_img_resize, contours, -1, (70, 150, 0), border_thick)\n' +
                '\n' +
                '        roi_img[:] = cv2.resize(roi_img_resize, (roi_img.shape[1], roi_img.shape[0]), interpolation=cv2.INTER_LINEAR)[:]\n' +
                '\n' +
                '        return img\n' +
                '```\n' },
        {text:'函数介绍：\n'},
        {text:'函数在进行人体密集姿态可视化时，会通过传入的`bbox`参数取出人物实例的区域来进行可视化操作。在获取到表示人物实例各部位掩模信息的`uv`第一通道信息后，先画每个部位的等值线和掩模轮廓，最后再画实例整体的掩模轮廓，并覆盖到根据`bbox`取出的可视化区域。\n'},
        {text:'使用案例：\n'},
        {text:'```Python\n' +
                '    # show uv\n' +
                '    if cfg.VIS.SHOW_UV.ENABLED:\n' +
                '        im = vis_uv(im, uv[i], bbox)\n' +
                '```\n'},
        {text:'可视化示例：\n'},
        {img:'UV_000000183648'},
        {part_title:'便利与规范'},
        {text:'基于Pet的代码实现标准，我们对可视化系统进行了封装，在测试阶段可视化网络输出结果时，您可以获得以下便利：\n'},
        {
            ul:['将多种功能的可视化和图像保存便捷高效的封装起来，加快工程开发周期。',
                '在功能实现的同时，加入了对控制各功能是否执行的开关标志变量，增加了操作的灵活性。']
        },
        {text:'如果您准备对我们的可视化系统进行拓展，您需要完全遵循Pet的代码实现标准来进行修改和添加，欢迎您将有价值的代码和意见提交到我们的github，我们十分感谢任何对于发展Pet有益的贡献。\n'}
    ],
    dataNav:[
        '可视化内容',
        {
            'vis_one_image_opencv':[
                'get_instance_parsing_colormap','vis_bbox','vis_class','vis_mask','vis_keypoints','vis_parsing','vis_uv'
            ]
        },
        '便利与规范'
    ]
}