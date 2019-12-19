export let config_system_data={
    key:'config_system',
    dataSource:[
        {title:'配置系统'},
        {part_title:'Configure简介'},
        {text:'配置系统是Pet在训练和测试过程中，用户对各个环节进行配置的控制台。配置系统使得pet运行过程中的设置变得灵活、高效和容易实现，同时配置系统还为pet提供了可扩展的配置，使得Pet用户能够自定义配置，不仅如此，配置系统还包括以下优点。\n'},
        {
            ul:[
                '配置设置易读性：由于配置系统是基于YAML文件类型，YAML文件具备可读性高、和脚本语言交互性好以及扩展性高等优势，所有训练测试过程中的配置信息都存放在YAML文本文件中，可以使用文本编辑器直接修改和设置相应配置细节。',
                '更新的便捷与即时性：由于配置系统具有参数配置高度集成的特点，在配置系统中某些配置被更改后，无需再逐一修改训练与测试中的脚本文件，大大提升了用户在优化迭代及测试过程中的效率。',
                '可扩展性：配置系统具有很强的扩展性，用户可根据自身需求在配置系统中添加所需参数，同时利用配置系统，用户能够自定义配置细节。\n'
            ]
        },
        {text:'我们使用YAML文件将训练和测试过程中的参数以键值对的形式进行保存，同时针对Pet的不同组件以及各任务中的具体操作参数使用层级关系进行对应。在配置文件书写过程中需注意的是YAML对大小写敏感，需严格注意Pet提供参数的大小写；对于每个对象的格式表达为key: value，同时冒号后面要加一个空格；对参数层级之间的缩进只能使用空格，不能使用TAB进行。配置系统所采用的格式实例如下所示：\n'},
        {yaml:'```\n' +
                '  Key1: \n' +
                '    child-key1: value1\n' +
                '    child-key2: value2\n' +
                '    …\n' +
                '  Key2:\n' +
                '    child-key1: value1\n' +
                '    child-key2: value2\n' +
                '    …\n' +
                '```\n'},
        {part_title: '配置系统的结构与内容'},
        {h3_title:'通用配置'},
        {text:'作为Pet用户对训练测试过程中各环节进行配置的控制台，配置系统按照Pet组件提供了一系列任务通用的参数设置，包括训练测试过程、模型搭建、迭代优化及可视化等参数设定。下表所示为Pet的配置系统针对每个组件所提供的参数以及各参数的用途：\n'},
        {h3_title: '各任务特有配置'},
        {text:'根据Pet对多任务支持的优势，配置系统同时提供了针对不同任务的参数配置，满足用户对pet所支持的各项视觉任务的自定义参数配置。\n'},
        {
            ul:[
                'Pose：针对姿态估计任务，Pet提供了差异化的图像预处理方式，姿态估计head网络结构选择及参数设置与最终推断结果可视化等配置，提供的所有参数如下表所示：',
                'SSD：针对单阶段检测，Pet为用户提供多种单阶段检测网络结构及损失函数，用户可自行选择调整单阶段检测模型的参数，Pet提供的SSD特有参数如下表所示：',
                'RCNN：作为Pet中支持网络结构最丰富的任务模块，RCNN提供的参数选择也相对较多，涵盖了两阶段目标检测过程中RPN的参数设置，同时针对不同的head结构，如mask-rcnn，faster-rcnn，RoI等参数也提供了可选配置， Pet提供的RCNN特有参数如下表所示：'
            ]
        },
        {part_title:'配置系统的使用'},
        {h3_title:'已有参数的使用'},
        {text:'Pet中参数配置模块配置系统采用三级调用的方式适配用户所需的参数配置。用户在训练与测试过程中，可在训练或测试时在命令行中提供YAML文件的相对路径及所需配置的更改，代码如下所示：\n' +
                '```Python\n' +
                '  # Parse arguments\n' +
                '  parser = argparse.ArgumentParser(description=\'Pet Model Training\')\n' +
                '  parser.add_argument(\'--cfg\', dest=\'cfg_file\',\n' +
                '                      help=\'optional config file\',\n' +
                '                      default=\'./cfgs/pose/mscoco/simple_R-50-1x64d-D3K4C256_192x256_1x.yaml\',\n' +
                '                      type=str)\n' +
                '  parser.add_argument(\'opts\', help=\'See $Pet/pose/core/config.py for all options\',\n' +
                '                      default=None,\n' +
                '                      nargs=argparse.REMAINDER)\n' +
                '  args = parser.parse_args()\n' +
                '  if args.cfg_file is not None:\n' +
                '      merge_cfg_from_file(args.cfg_file)\n' +
                '  if args.opts is not None:\n' +
                '  merge_cfg_from_list(args.opts)\n' +
                '```\n' +
                '配置系统采用三级调用的方式，其中第一级为命令行提供的参数配置。用户在训练测试过程中可选择在命令行中以列表\\[param1, value1, param2, value2, ……\\]的形式给出所需更改的参数。第二级为用户提供的YAML配置文件，Pet根据用户提供的配置文件路径寻找配置文件，当用户需要修改Pet已在`config.py`中定义好的参数时，可以在对应的YAML文件中按照格式要求对参数进行赋值。第三级为Pet在每个任务的`config.py`中定义完毕的整套默认参数，如果这些参数在前两级中没有被提及或是修改，Pet在运行过程中将调用定义的默认参数。第一级与第二级调用分别采用`config.py`中的`merge_cfg_from_list`和`merge_cfg_from_file`两个函数完成对默认参数的覆盖：\n' +
                '```Python\n' +
                '  def merge_cfg_from_file(filename):\n' +
                '      """Load a config file and merge it into the default options."""\n' +
                '      with open(filename, \'r\') as f:\n' +
                '          yaml_cfg = AttrDict(yaml.load(f))\n' +
                '      _merge_a_into_b(yaml_cfg, __C)\n' +
                '\n' +
                '\n' +
                '  def merge_cfg_from_list(cfg_list):\n' +
                '      """Merge config keys, values in a list (e.g., from command line) into the\n' +
                '      global config. For example, `cfg_list = [\'TEST.NMS\', 0.5]`.\n' +
                '      """\n' +
                '      assert len(cfg_list) % 2 == 0\n' +
                '      for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):\n' +
                '          if _key_is_deprecated(full_key):\n' +
                '              continue\n' +
                '          if _key_is_renamed(full_key):\n' +
                '              _raise_key_rename_error(full_key)\n' +
                '          key_list = full_key.split(\'.\')\n' +
                '          d = __C\n' +
                '          for subkey in key_list[:-1]:\n' +
                '              assert subkey in d, \'Non-existent key: {}\'.format(full_key)\n' +
                '              d = d[subkey]\n' +
                '          subkey = key_list[-1]\n' +
                '          assert subkey in d, \'Non-existent key: {}\'.format(full_key)\n' +
                '          value = _decode_cfg_value(v)\n' +
                '          value = _check_and_coerce_cfg_value_type(\n' +
                '            value, d[subkey], subkey, full_key\n' +
                '          )\n' +
                '          d[subkey] = value\n' +
                '```\n'},
        {h3_title:'添加新的参数'},
        {text:'当用户需要在训练与测试过程中添加新的参数时，配置系统提供快速便捷的添加渠道，仅需在对应任务的对应组件下定义参数，并在对应实现下进行参数调用即可。\n'}

    ]
}