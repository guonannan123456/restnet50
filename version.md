# Linux系统下利用anaconda创建虚拟环境并安装pytorch0.4.1和torchvision
https://blog.csdn.net/amateur_hy/article/details/90716411

+ conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

    1

利用以上命令，可同时安装PyTorch和torchvision，但PyTorch的版本为1.1，而所需PyTorch版本为0.4.1，因此不能采用以上命令进行配置，而是要分别安装PyTorch 0.4.1和torchvision。

pytorch官网提供了previous versions of PyTorch安装选项，可根据需求进行安装。

·安装pytorch 0.4.1

conda install pytorch=0.4.1 cuda90 -c pytorch

    1

以上是cuda9.0版本的安装命令。

·安装torchvision

conda install torchvision

    1

其他需要的包，均可以通过pip进行安装，不赘述。
至此完成了环境的配置。
————————————————
版权声明：本文为CSDN博主「amateur_hy」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/amateur_hy/article/details/90716411

