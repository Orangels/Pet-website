import React from "react";

export let Install_codePart_data_EN = {
    'Install basic dependencies':[
        'apt-get install python3.5-dev',
        'apt-get install python3.5-tk',
        'pip install six --ignore-installed',
        'sudo pip3 install numpy scipy scikit-image matplotlib',
        'sudo pip3 install opencv-python',
    ],
    'Install Pytorch':[
        'sudo pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp35-cp35m-linux_x86_64.whl'
    ],
    'Install torchvision':[
        'sudo pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp35-cp35m-linux_x86_64.whl'
    ],
    'apex':[
        '# Download NVIDIA apex',
        'cd $INSTALL_DIR',
        'git clone https://github.com/NVIDIA/apex.git',
        ' ',
        '# Inatsll NVIDIA apex',
        'cd apex',
        'sudo python setup.py install --cuda_ext --cpp_ext'
    ],
    'pycocotools':[
        '# Download pycocotools',
        'cd $INSTALL_DIR',
        'git clone https://github.com/cocodataset/cocoapi.git',
        ' ',
        '# Inatsll pycocotools',
        'cd cocoapi/PythonAPI',
        'python setup.py build_ext install',
    ],
    'Clone Pet':[
        '# Clone Pet into current path:',
        'git clone https://github.com/BUPT-PRIV/Pet.git',
        ' ',
        '# Install the dependencies：',
        'sudo pip3 install -r Pet/requirements.txt',
        ' ',
        '# Compile Pet:',
        'cd Pet/pet',
        'sh ./make.sh',

    ],
    'Install dependencies':[
        'sudo pip3 install -r Pet/requirements.txt'
    ],
    'Make MaskRCNN':[
        'cd Pet/pet',
        'sh ./make.sh'
    ],
    'ps':[
        'export CXXFLAGS="-std=c++11"',
        'export CFLAGS="-std=c99"',
        '# repeat run make',
        'sh ./make.sh',
    ],
    'complited':[
        '/usr/local/cuda/bin/nvcc -DWITH_CUDA -I/home/user/Downloads/Pet/pet/models/ops/csrc -I/usr/local/lib/python3.5/dist-packages/torch/include -I/usr/local/lib/python3.5/dist-packages/torch/include/torch/csrc/api/include -I/usr/local/lib/python3.5/dist-packages/torch/include/TH -I/usr/local/lib/python3.5/dist-packages/torch/include/THC -I/usr/local/cuda/include -I/usr/include/python3.5m -c /home/user/Downloads/Pet/pet/models/ops/csrc/cuda/deform_pool_cuda.cu -o build/temp.linux-x86_64-3.5/home/user/Downloads/Pet/pet/models/ops/csrc/cuda/deform_pool_cuda.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --compiler-options \'-fPIC\' -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11',
        '/usr/local/lib/python3.5/dist-packages/torch/include/c10/util/ArrayRef.h:277:48: warning: ‘deprecated’ attribute directive ignored [-Wattributes]',
        'using IntList C10_DEPRECATED_USING = ArrayRef<int64_t>;',
        '                                                                                                                                                ^',
        'creating build/lib.linux-x86_64-3.5',
        'x86_64-linux-gnu-g++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 build/temp.linux-x86_64-3.5/home/user/Downloads/Pet/pet/models/ops/csrc/vision.o build/temp.linux-x86_64-3.5/home/user/Downloads/Pet/pet/models/ops/csrc/cpu/nms_cpu.o build/temp.linux-x86_64-3.5/home/user/Downloads/Pet/pet/models/ops/csrc/cpu/ROIAlign_cpu.o build/temp.linux-x86_64-3.5/home/user/Downloads/Pet/pet/models/ops/csrc/cuda/ROIAlign_cuda.o build/temp.linux-x86_64-3.5/home/user/Downloads/Pet/pet/models/ops/csrc/cuda/ROIPool_cuda.o build/temp.linux-x86_64-3.5/home/user/Downloads/Pet/pet/models/ops/csrc/cuda/deform_conv_cuda.o build/temp.linux-x86_64-3.5/home/user/Downloads/Pet/pet/models/ops/csrc/cuda/nms.o build/temp.linux-x86_64-3.5/home/user/Downloads/Pet/pet/models/ops/csrc/cuda/deform_conv_kernel_cuda.o build/temp.linux-x86_64-3.5/home/user/Downloads/Pet/pet/models/ops/csrc/cuda/SigmoidFocalLoss_cuda.o build/temp.linux-x86_64-3.5/home/user/Downloads/Pet/pet/models/ops/csrc/cuda/deform_pool_kernel_cuda.o build/temp.linux-x86_64-3.5/home/user/Downloads/Pet/pet/models/ops/csrc/cuda/deform_pool_cuda.o -L/usr/local/cuda/lib64 -lcudart -o build/lib.linux-x86_64-3.5/_C.cpython-35m-x86_64-linux-gnu.so',
        'copying build/lib.linux-x86_64-3.5/_C.cpython-35m-x86_64-linux-gnu.so ->',
    ],
    'Revise':[
        'sudo vim /usr/local/lib/python3.5/dist-packages/pycocotools/coco.py',
    ],
    'Change':[
        'if type(resFile) == str or type(resFile) == unicode:'
    ],
    'to':[
        'if type(resFile) == str:'
    ]




}