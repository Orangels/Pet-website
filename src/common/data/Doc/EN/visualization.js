export let visualization_data = {
    key: 'Visualization',
    dataSource: [
        {title:'Visualization'},
        {text:'Visualization can help us intuitively understand data annotation and model inference results. Under the `$Pet/pet` path, according to the different data format and requirements of each task, Pet packages the corresponding visualization tools in the `$Pet/pet/utils/vis.py` file of the corresponding task.\n'},
        {part_title:'Visual Content'},
        {text:'Here is an introduction to the visualization capabilities of Pet for different tasks.\n'},
        {table:{
                titles:['task','supports visualization'],
                data:[["cls","--"],["rcnn","1. Add an external frame to the instance of each mark in the image;\n"+
                "2. Add category label characters at the designated position of the image;\n" +
                "3. Add mask to each instance of the image;\n" +
                "4. Add a corresponding mask to each part of the instance in the image;\n" +
                " 5. Add dense attitude effect to each instance of the image."],["ssd","1. Add an external frame to each instance in the image;\n" +
                "2. Add category label characters at the designated position of the image. "],["pose","1. Add external frame to each instance in the image;\n" +
                "2. Add category label characters at the designated position of the image;\n" +
                "3. Add key points of body parts to each instance of the image."],]
            }
            , className:'table_1',type:'start',table_width:430},
        {part_title: 'vis_one_image_opencv'},
        {text:'Pet uses a total visualization function `vis_one_image_opencv` to visualize all the visualization functions on a single image, `get_instance_parsing_colormap` is called here to get the instance mask color of the image and the mask color set of various parts of the body, then `vis_bbox`, `vis_class`, `vis_mask`, `vis_keypoints`, `vis_parsing`, `vis_uv` and other functions are used to analyze bounding box, instance category, mask, keypoints of body instance, mask of each part, and intensive gesture labeling information.\n' +
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
                'Here requires passing image and annotating information, such as im, boxes, segms, keypoints, parsing, uv, dataset, etc. The variables segms, keypoints, parsing, uv, and dataset have an empty default value that is used to skip visualization. In addition, the function initially detects the data format and automatically skips the visualization of the current instance in case of formatting errors.\n'},
        {table:{
                titles:['input variable','content','data structure'],
                data:[["im","Single original image data",'numpy'],["boxes","External frame coordinates and scores of all instances in a single original image data image",'list'],["classes","Category index of all instances in an image",'list'],["segms","Masking of all instances in an image",'list'],["keypoints","Coordinates of key points in human body parts of all pedestrian instances in the image",'list'],["parsing","Human body mask for all pedestrian instances in the image",'list'],["uv","Dense Posture of All Pedestrian Instances in Image",'list'],['dataset','Class Variables Containing Category Information of Data Sets','class']]
            }
            , className:'table_2'},
        {h3_title:'get_instance_parsing_colormap'},
        {text:'In the file [$Pet/pet/utils/colormap.py](https://github.com/BUPT-PRIV/Pet-dev/blob/master/pet/utils/colormap.py), color data in the form of dictionaries of data sets such as COCO, CIHP, VOC, ADE20K, MHP, CityScape and so on are defined. Through `get_instance_parsing_colormap` function, the dictionary set of data set set in cfg parameter is taken out and used as the color dictionary set of external frame and body part segmentation. After obtaining the standard color set of the dataset, `vis_one_image_opencv` will use `COLOR_SCHEME` in the parameter cfg to choose whether the visual color of an instance is added by category or by instance.\n'},
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
                {text:'In `$Pet/pet/utils/colormap.py` file, Pet store all the datasets\' official color set according to the BGR format, in the form of a dictionary.\n'},
                {text:'The name of the color set is named after the data set name with the total number of categories in the dataset. When you add a new set of dataset to the color set, it is also recommended to use this standard. In addition, before difining the official color set corresponding to the dataset, the color set `color map` is added based on the color matching method of [Detectron](https://github.com/facebookresearch/Detectron) project, 79 kinds of colors are defined by dictionary, and the color specific segmentation mask can be added to each instance via corlormap.\n'}
            ]
        },
        {h3_title: 'vis_bbox'},
        {text:'Parameter description:\n'},
        {text:'To realize the visualization of the external box of a single instance, `img`, `bbox`, `bbox_color` and other parameters need to be passed in.\n'},
        {
            ul:['`img`: image pixel data stored in the format of numpy array;',
                '`bbox`: the X, Y coordinates and width and height information in the upper left corner of the current instance, in the form [x, y, w, h];',
                '`bbox_color`: the color to be displayed by the peripheral box, in the form of [r, g, b].']
        },
        {text:'Use case:：\n'},

        {text:'```Python\n' +
                '    # show box (off by default)\n' +
                '    if cfg.VIS.SHOW_BOX.ENABLED:\n' +
                '        im = vis_bbox(im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), ins_color)\n' +
                '```\n'},
        {text:'Visualization results:：'},
        {img:'Box_000000183648'},
        {h3_title:'vis_class'},
        {text:'Parameter description:'},
        {text:'In order to realize the visualization of the character data such as the corresponding category and score with the instance added on the corresponding position, `img`, `pos`, `class_str`, `bg_color` and other parameters need to be passed in. The display content COLOR can be modified with the cfg.vis.show_class.color parameter, which defaults to white. Pet also defines the get_class_string function to make it easy for the user to get the instance category to visualize, and the confidence content is returned to the class_str variable.\n'},
        {
            ul:['`img`: image pixel data stored in numpy array format;',
                '`pos`: tuple data of two elements, used to express the X and Y coordinates of the starting point of the visual content, in the form of (x and y);',
                '`class_str`: the character content to visualize;',
                '`bg_color`: the background color for displaying content.'
            ]
        },
        {text: '```Python\n' +
                '    # show class (off by default)\n' +
                '    if cfg.VIS.SHOW_CLASS.ENABLED:\n' +
                '        class_str = get_class_string(classes[i], score, dataset)\n' +
                '        im = vis_class(im, (bbox[0], bbox[1] - 2), class_str, ins_color)\n' +
                '```\n'},
        {text:'Visualization results::'},
        {img:'Class_000000183648'},
        {h3_title:'vis_mask'},
        {text:'Parameter description:\n'},
        {text:'To realize the visualization of a single instance mask, parameters such as img, mask, bbox_color, show_parss and so on are required to be passed in.\n'},
        {
            ul:['`img`: image pixel data stored in numpy array format;',
                '`mask`: single channel numpy with the same length and width as img;',
                '`bbox_color`: the color to be displayed by the peripheral box, in the form of [r, g, b];',
                '`show_parss`: identifies variables and skips the mask visualization of the whole instance when the mask visualization of various parts of the body is performed.'
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
        {text:'Function introduction:'},
        {text:'After the variable is passed in, we obtain the foreground mask index of non-zero in the mask, and then weighted sum the corresponding pixels and input color pixels in a certain proportion, and cover the original pixels to get the mask visualization result. The scale factor is controlled by the parameter cfg.vis.show_segms.MASK_ALPHA.\n'},
        {text:'Use case:'},
        {text:'```Python\n' +
                '    # show mask\n' +
                '    if cfg.VIS.SHOW_SEGMS.ENABLED:\n' +
                '        color_list = colormap_utils.colormap()\n' +
                '        im = vis_mask(im, masks[..., i], ins_color, show_parss=show_parss)\n' +
                '\n' +
                '```\n'},
        {text:'Visualization results:'},
        {img:'000000183648_Mask'},
        {h3_title:'vis_keypoints'},
        {text:'Parameter description:\n'},
        {text:'img, kps, show_parss and other parameters need to be passed to realize the visualization of key points of a single human body instance.\n'},
        {
            ul:['`img`: image pixel data stored in numpy array format;',
                '`kps`: Human point mark, form is the [[x1, x2,..., x17], [y1, y2,..., y17], [c1 and c2,..., c17]].',
                '`show_parss`: skips the identification variable of the visualization operation, skips the visualization of the key points of the instance when each part of the body performs the visualization.'
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
        {text:'Function introduction:\n'},
        {text:'In vis_keypoints function, not only the original 17 keypoints of human body parts are used, but also the middle points of shoulders and hips are calculated correspondingly to assist the connection of keypoints of human body.\n'},
        {text:'Use case:\n'},
        {text:'```Python  \n' +
                '    # show keypoints\n' +
                '    if cfg.VIS.SHOW_KPS.ENABLED:\n' +
                '        im = vis_keypoints(im, keypoints[i], show_parss=show_parss)\n' +
                '```\n'},
        {text:'Visualization results:\n'},
        {img:'Keypoints_000000183648'},
        {h3_title:'vis_parsing'},
        {text:'Parameter description:\n'},
        {text:'It is used to make the mask visualization of all parts of a single body instance, and it can also be used to make the mask visualization of panoramic segmentation, requiring passing in parameters such as img, parsing, colormap, show_segms, etc.\n'},
        {
            ul:['`img`: image in numpy format;',
                '`parsing`: single channel numpy array information with the same width and length as img;',
                '`colormap`: the set of colors corresponding to human body parts in dictionary format;',
                '`show_segms`: skip the identification variable of the visualization operation, and skip the visualization of various parts of the human body when performing the mask visualization of the overall instance.'
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
        {text:'Function introduction:\n'},
        {text:'Unlike vis_mask, which focuses on non-zero pixels as a whole with only one non-zero value range, the mask parameter of vis_parsing includes multiple pixel values to correspond to different parts of the body. Therefore, the incoming color set must make sure that the key value corresponds to the pixel value in each part of the parsing, and the parsing_color is used to map the pixel point of the mask data to the corresponding color of the category.\n'},
        {text:'Use case:\n'},
        {text:'```Python\n' +
                '    # show parsing\n' +
                '    if cfg.VIS.SHOW_PARSS.ENABLED:\n' +
                '        im = vis_parsing(im, parsing[i], parss_colormap, show_segms=show_segms)\n' +
                '```\n'},
        {text:'Visualization results:\n'},
        {img:'Parsing_000000183648'},
        {h3_title:'vis_uv'},
        {text:'Parameter description:\n'},
        {text:'For realizing the visualization of the dense posture of a single human body instance, `img`、`uv`、`bbox` and other parameters need to be passed in.\n'},
        {
            ul:['`img`: image pixel data stored in numpy array format;',
                '`uv`: IUV information of the current instance;',
                '`bbox`: the X, Y coordinates and width and height information in the upper left corner of the current instance\'s outer box, in the form [X, Y, w, h].']
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
        {text:'Function introduction:\n'},
        {text:'When the function is used to visualize the dense posture of human body, the area of human body instance will be taken out through the passed bbox parameter for visualization. After obtaining the uv channel information representing the mask of human body parts, contour lines and mask contour of each part are drawn first, and then the overall mask contour of the instance is drawn, and the visual area taken out according to bbox is covered.\n'},
        {text:'Use case:\n'},
        {text:'```Python\n' +
                '    # show uv\n' +
                '    if cfg.VIS.SHOW_UV.ENABLED:\n' +
                '        im = vis_uv(im, uv[i], bbox)\n' +
                '```\n'},
        {text:'Visualization results:\n'},
        {img:'UV_000000183648'},
        {part_title:'Convenience and Specification'},
        {text:'Based on the implementation standard of Pet code, we have packaged the visualization system, and you can get the following convenience when processing data and testing network results:\n'},
        {
            ul:['The visualization of various functions and image storage are conveniently and efficiently packaged to speed up the engineering development cycle.',
                'At the same time of function realization, the switch flag variable is added to control whether each function is executed or not, which increases the flexibility of operation.']
        },
        {text:'If you are ready to expand our visualization system, you need to completely follow the code implementation standards of Pet to modify and add. You are welcome to submit your valuable code and comments to our github. We appreciate any useful contributions to the development of Pet.\n'}
    ],
    dataNav:[
        'Visual Content',
        {
            'vis_one_image_opencv':[
                'get_instance_parsing_colormap','vis_bbox','vis_class','vis_mask','vis_keypoints','vis_parsing','vis_uv'
            ]
        },
        'Convenience and Specification'
    ]
}