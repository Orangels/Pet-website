export let home_use_pet_data = [
    {
        title:'Using Pet',
        text:[
            'Pet provides efficient tools and libraries for supporting ' ,
            'development in Computer Vision. Here are the simple codes ',
            'for training a Faster RCNN model.'
        ],
        code:'python -m torch.distributed.launch --nproc_per_node = 8 \\\n' +
            'tools/rnn/train_net.py \\\n' +
            '--cfg cfgs/rnn/mscoco/e2e_faster_rcnn_R-50-FPN_1x.yaml'
    },
    {
        title:'使用 Pet',
        text:['Pet为支持计算机视觉的开发提供了有效的工具和库。下面是训练Faster RCNN',
        '模型的代码示例。']
    }
]