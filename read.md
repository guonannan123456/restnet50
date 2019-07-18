## 解决loss值不下降问题（转）
https://blog.ailemon.me/2019/02/26/solution-to-loss-doesnt-drop-in-nn-train/

 

1.模型结构和特征工程存在问题

 

2.权重初始化方案有问题

 

3.正则化过度

 

4.选择合适的激活函数、损失函数

 

5.选择合适的优化器和学习速率

 

6.训练时间不足

 

7.模型训练遇到瓶颈

 

8.batch size过大

 

9.数据集未打乱

 

10.数据集有问题

 

11.未进行归一化

 

12.特征工程中对数据特征的选取有问题

https://morvanzhou.github.io/tutorials/machine-learning/torch/2-02-variable/

https://blog.csdn.net/caicai2526/article/details/79984950

https://space.bilibili.com/45151802?spm_id_from=333.788.b_765f7570696e666f.2

https://zhuanlan.zhihu.com/p/52007005

https://zhuanlan.zhihu.com/p/31426458

https://www.coursera.org/courses?query=keras&indices%5Bprod_all_products_custom_ranking_revenuelast28d%5D%5Bpage%5D=4&indices%5Bprod_all_products_custom_ranking_revenuelast28d%5D%5Bconfigure%5D%5BclickAnalytics%5D=true&indices%5Bprod_all_products_custom_ranking_revenuelast28d%5D%5Bconfigure%5D%5BhitsPerPage%5D=10&configure%5BclickAnalytics%5D=true

https://paperswithcode.com

https://blog.csdn.net/nuanxin_520/article/details/72818330

http://www.arxiv-sanity.com/search?q=face+detection

## 卷积神经网络反向求导时对池化层怎么处理

无论max pooling还是mean pooling，都没有需要学习的参数。因此，在卷积神经网络的训练中，Pooling层需要做的仅仅是将误差项传递到上一层，而没有梯度的计算。
+ （1）max pooling层：对于max pooling，下一层的误差项的值会原封不动的传递到上一层对应区块中的最大值所对应的神经元，而其他神经元的误差项的值都是0；
+ （2）mean pooling层：对于mean pooling，下一层的误差项的值会平均分配到上一层对应区块中的所有神经元
## 神经网络权值为什么不能初始化为零（1）
+ https://www.cnblogs.com/lky-learning/p/10830223.html
