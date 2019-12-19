export let overview = {
    key:'overview',
    dataSource:[
        {title:'Overview'},
        {text:'Building a high-performance deep learning toolbox for Computer Vision requires many system-level design decisions to ensure efficient training and testing. An excellent deep learning model training and testing platform includes several components like data preparation, data loading, model building, optimization, and model loading and saving, visualization and logging. \n'},
        {text:'In order to ensure the comprehensiveness, flexibility and efficiency when using Pet. We carefully design the above components, greatly enriching the functions of each component in CNN training and testing process. In the Pet\'s architecture description, we share the rules and basic principles that are followed when designing Pet. We hope that these insights can help deep learning learners and practitioners. '},
        {part_title:'System Implementation'},
        {text:'Pet is designed with high conciseness and efficiency, the system structure is clear and the code style is logical. On the premise of comprehensive work involving visual tasks and its efficiency, Pet also has considerable flexibility and extensibility, which facilitate users to expand and add new visual task solutions in the existing architecture.'},
        {img:'sys_arch_eng'},
        {part_title:'System Components'},
        {text:'Following the system architecture of Pet, the roles components in the training and testing CNN models are as follows:\n'},
        {h3_title:'Config System'},
        {text:'All visual tasks based on Pet are modularized by configuration system, using the configuration system as the control interface can direct the training and testing of different visual tasks. The training and testing process can be completed only with training, testing scripts and configuration system.\n'},
        {h3_title:'Data Preparation'},
        {text:'For different visual tasks, Pet supports the training and testing of models on multiple data sets, with specified file structure and label format.\n'},
        {h3_title:'Data Loading'},
        {text:'Pet implements a set of standard data loading interface, and provides a variety of online data augmentation methods such as scaling, rotating, flipping, etc. to make neural network more robust on other fields.\n'},
        {h3_title:'Model Building'},
        {text:'Pet divide the CNN in all vision tasks into four modules: feature extraction network, function network, task output network and loss function network. Through the combination of network modules, we can flexibly construct networks of different structure. This rule has high extensibility, you can build a new network and develop the experiments only by defining a new convbody or a new head, so as to improve the efficiency of configuration modification in series of experiments.\n'},
        {h3_title:'Optimization'},
        {text:'Pet not only provides comprehensive optimization of hyperparameter Settings and optimizers, but also provides rich optimization solutions for the training of convolutional neural networks. According to different tasks, Pet give you some  best training configuration for reference, shortens research period. At the same time Pet supports distributed data parallel training and mixed precision computing and other efficient computing technologies.\n'},
        {h3_title:'Model Loading and Saving'},
        {text:'For various subtasks, Pet provides thousands of pre-trained models in [MODEL_ZOO]() for training and testing. At the same time, checkpoints during training are saved in an efficient way to prevent time loss caused by unexpected training terminate or crash, and the training process can be resumed from the checkpoint without modifying any training parameters.\n'},
        {h3_title:'Visualization'},
        {text:'Pet provides a unified model output visualization scheme. Users can choose the corresponding visualization tools according to tasks, and visually present the visualization results of model prediction on images.\n'},
        {h3_title:'logging'},
        {text:'Pet records the learning rate, error, precision and other model training status in real time through logger, Logger also draws the accuracy and error curve through every checkpoint.\n'}
    ]
}