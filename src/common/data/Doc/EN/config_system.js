export let config_system_data={
    key:'config_system',
    dataSource:[
        {title:'Config system'},
        {part_title:'Introduction to Configure'},
        {text:'Configure is the console that pet configures for each link during training and testing. Configure makes the configuration of the pet run flexible, efficient and easy to implement, while configure also provides pet with a scalable configuration, allowing pet users to customize the configuration, not only that, configure also includes the following advantages.\n'},
        {
            ul:[
                'Configuration settings legibility: Since Configure is based on YAML file types, YAML files have the advantages of high readability, good interactivity with scripting languages, and high scalability. All configuration information during training and testing is stored in YAML text files. You can use a text editor to directly modify and set the appropriate configuration details.',
                'The convenience and immediacy of the update: Since configure has a highly integrated parameter configuration, after some configurations in the Configure are changed, there is no need to modify the script files in the training and test one by one, which greatly improves the user\'s optimization iteration and testing process. Efficiency in the middle.',
                'Scalability: Configure is very scalable, users can add the required parameters to configure according to their needs, and with Configure, users can customize the configuration details.Application configuration using Configure greatly enhances the extensibility and flexibility of the application, and changes to the configuration file can be immediately applied to the PET application.\n'
            ]
        },
        {text:'We use YAML files to save the parameters of the training test in the form of key-value pairs, and use hierarchical relationships for different components of pet and specific operational parameters in each task. In the process of writing the configuration file, it should be noted that YAML is case sensitive. It is necessary to strictly pay attention to the case of the parameters provided by pet. For each object, the format is expressed as key: value, and a space is added after the colon; Indentation can only use spaces, not TAB. The format examples used by Configure are as follows:\n'},
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
        {part_title: 'Configure structure and content'},
        {h3_title:'General configuration'},
        {text:'As a console for pet users to configure each step in the training and testing process, configure provides a series of parameters common to the pet component, including training and testing process, model building, iterative optimization and visualization.The following table shows the parameters that pet\'s configure provides for each component and what they are used for:\n'},
        {h3_title: 'Task-specific configuration'},
        {text:'According to pet\'s advantage of multi-tasking support, configure also provides parameter configuration for different tasks, which satisfies the user\'s custom parameter configuration for each visual task supported by pet.\n'},
        {
            ul:[
                'Pose: For pose estimation tasks, pet provides differentiated image preprocessing methods, pose estimation head network structure selection and parameter settings and final inference results visualization. All the parameters provided are as follows:',
                'SSD: For single-stage detection, pet provides users with a variety of single-stage detection network structure and loss function. Users can choose to adjust the parameters of the single-stage detection model. The SSD-specific parameters provided by pet are shown in the following table:',
                'RCNN: As the most abundant task module supporting pet network structure in pet, RCNN provides relatively more parameter selection, covering the parameter setting of RPN in the two-stage target detection process, and for different head structures, such as mask-rcnn, Parameters such as faster-rcnn, RoI, etc. are also available. The RCNN-specific parameters provided by pet are shown in the following table:'
            ]
        },
        {part_title:'Use of Configuraion'},
        {h3_title:'Use of existing parameters'},
        {text:'The parameter configuration module configure in Pet adapts the parameter configuration required by the user in a three-level call. During the training and testing process, the user can provide the relative path of the YAML file and the required configuration changes on the command line during training or testing. The code is as follows:\n' +
                '```Python\n' +
                '  # Parse arguments\n' +
                '  parser = argparse.ArgumentParser(description=\'Pet Model Training\')\n' +
                '  parser.add_argument(\'--cfg\', dest=\'cfg_file\',\n' +
                '                      help=\'optional config file\',\n' +
                '                      default=\'./cfgs/pose/mscoco/simple_R-50-1x64d-D3K4C256_192x256_1x.yaml\',\n' +
                '                      type=str)\n' +
                '  parser.add_argument(\'opts\', help=\'See $Pet/pet/pose/core/config.py for all options\',\n' +
                '                      default=None,\n' +
                '                      nargs=argparse.REMAINDER)\n' +
                '  args = parser.parse_args()\n' +
                '  if args.cfg_file is not None:\n' +
                '      merge_cfg_from_file(args.cfg_file)\n' +
                '  if args.opts is not None:\n' +
                '  merge_cfg_from_list(args.opts)\n' +
                '```\n' +
                'Configure uses a three-level call, where the first level is the parameter configuration provided by the command line. During the training test, the user can choose to give the parameters of the required changes in the command line in the form of a list [param1, value1, param2, value2, ...]. The second level is the YAML configuration file provided by the user. Pet searches for the configuration file according to the configuration file path provided by the user. When the user needs to modify the parameters that pet has defined in ``config.py``, the corresponding YAML file can be used. The parameters are assigned according to the format requirements. The third level is the set of default parameters defined by pet in ``config.py``. If these parameters are not mentioned or modified in the first two levels, pet will call the defined default parameters during the running process. The first level and the second level call respectively use the ``merge_cfg_from_list`` and ``merge_cfg_from_file`` functions in ``config.py`` to complete the override of the default parameters:\n' +
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
        {h3_title:'Add new parameters'},
        {text:'When the user needs to add new parameters during the training and testing process, configure provides a quick and convenient way to add channels, just define the parameters under the corresponding components of the corresponding task, and call the parameters under the corresponding implementation.\n'}

    ]
}