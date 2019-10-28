# Linux系统下利用anaconda创建虚拟环境并安装pytorch0.4.1和torchvision
https://blog.csdn.net/amateur_hy/article/details/90716411

+ conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

利用以上命令，可同时安装PyTorch和torchvision，但PyTorch的版本为1.1，而所需PyTorch版本为0.4.1，因此不能采用以上命令进行配置，而是要分别安装PyTorch 0.4.1和torchvision。

pytorch官网提供了previous versions of PyTorch安装选项，可根据需求进行安装。

·安装pytorch 0.4.1

+ conda install pytorch=0.4.1 cuda90 -c pytorch

以上是cuda9.0版本的安装命令。

·安装torchvision

- conda install torchvision

