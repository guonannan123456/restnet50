# CVPR 2019
- Prelimary Technical Program https://docs.google.com/spreadsheets/d/1RU2y-iuzwtAR_hn4V9yz1qpZSiElm3iaCpUoDJ-vfvQ/htmlview?sle=true#
- COIN https://coin-dataset.github.io/

## Classification
- 检测 29
- 分割 36
- 分类、识别 15
- 跟踪 19
- 人脸 8
- 人体姿态估计、位姿估计 18
- 行为/动作识别、手势识别 10
- 时序动作检测、视频相关 18
- Related to Networks 34
- GAN、图像文本生成 21
- 图像/视频处理、超分辨 18
- 点云、三维重建 25
- VQA、视觉语言导航 10
- OCR、文本检测 8
- 自动驾驶、SLAM 12
- 人群计数 3
- 数据集 6
- 行人重识别、行人检测 8
- 其他 254

### 检测 29

1、Stereo R-CNN based 3D Object Detection for Autonomous Driving
作者：Peiliang Li, Xiaozhi Chen, Shaojie Shen
论文链接：https://arxiv.org/abs/1902.09738

2、Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression
作者：Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, Silvio Savarese
论文链接：https://arxiv.org/abs/1902.09630
论文解读：https://mp.weixin.qq.com/s/6QsyYtEVjavoLfU_lQF1pw

3、ROI-10D: Monocular Lifting of 2D Detection to 6D Pose and Metric Shape 作者：Fabian Manhardt, Wadim Kehl, Adrien Gaidon
论文链接：https://arxiv.org/abs/1812.02781

4、Bi-Directional Cascade Network for Perceptual Edge Detection
作者：Jianzhong He, Shiliang Zhang, Ming Yang, Yanhu Shan, Tiejun Huang
论文链接：https://arxiv.org/abs/1902.10903
Github源码：https://github.com/pkuCactus/BDCN

5、RepMet: Representative-based metric learning for classification and one-shot object detection
作者：Leonid Karlinsky, Joseph Shtok, Sivan Harary, Eli Schwartz, Amit Aides, Rogerio Feris, Raja Giryes, Alex M. Bronstein
论文链接：https://arxiv.org/abs/1806.04728

6、Region Proposal by Guided Anchoring
作者：Jiaqi Wang, Kai Chen, Shuo Yang, Chen Change Loy, Dahua Lin
论文链接：https://arxiv.org/abs/1901.03278
论文解读：https://mp.weixin.qq.com/s/Sl958JkcJjy-HW9_c-SH4g
Github链接：https://github.com/open-mmlab/mmdetection

7、Less is More: Learning Highlight Detection from Video Duration
作者：Bo Xiong, Yannis Kalantidis, Deepti Ghadiyaram, Kristen Grauman
论文链接：https://arxiv.org/abs/1903.00859

8、AIRD: Adversarial Learning Framework for Image Repurposing Detection
作者：Ayush Jaiswal, Yue Wu, Wael AbdAlmageed, Iacopo Masi, Premkumar Natarajan
论文链接：https://arxiv.org/abs/1903.00788

9、Feature Selective Anchor-Free Module for Single-Shot Object Detection
作者：Chenchen Zhu, Yihui He, Marios Savvides
论文链接：https://arxiv.org/abs/1903.00621
论文解读：CVPR2019 | FSAF：来自CMU的Single-Shot目标检测算法
一作直播：CVPR2019 专题直播 | CMU 诸宸辰:基于 Anchor-free 特征选择模块的单阶目标检测

10、Learning Attraction Field Representation for Robust Line Segment Detection
作者：Nan Xue, Song Bai, Fudong Wang, Gui-Song Xia, Tianfu Wu, Liangpei Zhang
论文链接：https://arxiv.org/abs/1812.02122
代码链接：https://github.com/cherubicXN/afm_cvpr2019

11、Latent Space Autoregression for Novelty Detection
作者：Davide Abati, Angelo Porrello, Simone Calderara, Rita Cucchiara
论文链接：https://arxiv.org/abs/1807.01653
代码链接: https://github.com/aimagelab/novelty-detection

12、Strong-Weak Distribution Alignment for Adaptive Object Detection
作者：Kuniaki Saito, Yoshitaka Ushiku, Tatsuya Harada, Kate Saenko
论文链接：https://arxiv.org/abs/1812.04798

13、Few-shot Adaptive Faster R-CNN
作者：Tao Wang, Xiaopeng Zhang, Li Yuan, Jiashi Feng
论文链接：https://arxiv.org/abs/1903.09372

14、Attention Based Glaucoma Detection: A Large-scale Database and CNN Model
作者：Liu Li, Mai Xu, Xiaofei Wang, Lai Jiang, Hanruo Liu
论文链接：https://arxiv.org/abs/1903.10831

15、Bounding Box Regression with Uncertainty for Accurate Object Detection（目标检测边界框回归损失算法）
作者：Yihui He, Chenchen Zhu, Jianren Wang, Marios Savvides, Xiangyu Zhang
论文链接：https://arxiv.org/abs/1809.08545
代码链接：https://github.com/yihui-he/KL-Loss

16、Precise Detection in Densely Packed Scenes
作者：Eran Goldman , Roei Herzig, Aviv Eisenschtat, Jacob Goldberger, Tal Hassner
论文链接：https://arxiv.org/abs/1904.00853


17、Activity Driven Weakly Supervised Object Detection
作者：Zhenheng Yang, Dhruv Mahajan, Deepti Ghadiyaram, Ram Nevatia, Vignesh Ramanathan
论文链接：https://arxiv.org/pdf/1904.01665.pdf


18、Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction
作者：Jason Ku, Alex D. Pon, Steven L. Waslander
论文链接：https://arxiv.org/pdf/1904.01690.pdf


19、Libra R-CNN: Towards Balanced Learning for Object Detection(目标检测)
作者：Jiangmiao Pang, Kai Chen, Jianping Shi, Huajun Feng, Wanli Ouyang, Dahua Lin
论文链接：https://arxiv.org/abs/1904.02701


20、Moving Object Detection under Discontinuous Change in Illumination Using Tensor Low-Rank and Invariant Sparse Decomposition
作者：Moein Shakeri, Hong Zhang
论文链接：https://arxiv.org/abs/1904.03175


21、Towards Universal Object Detection by Domain Attention
作者：Xudong Wang, Zhaowei Cai, Dashan Gao, Nuno Vasconcelos
论文链接：https://arxiv.org/abs/1904.04402
项目链接：http://www.svcl.ucsd.edu/projects/universal-detection/


22、NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection
作者：Golnaz Ghiasi, Tsung-Yi Lin, Ruoming Pang, Quoc V. Le
论文链接：https://arxiv.org/abs/1904.07392


23、Deep Anomaly Detection for Generalized Face Anti-Spoofing
作者：Daniel Pérez-Cabo, David Jiménez-Cabello, Artur Costa-Pazo, Roberto J. López-Sastre
论文链接：https://arxiv.org/abs/1904.08241


24、Cascaded Partial Decoder for Fast and Accurate Salient Object Detection
作者：Zhe Wu, Li Su, Qingming Huang
论文链接：https://arxiv.org/abs/1904.08739



25、A Simple Pooling-Based Design for Real-Time Salient Object Detection
作者：Jiang-Jiang Liu, Qibin Hou, Ming-Ming Cheng, Jiashi Feng, Jianmin Jiang
论文链接：https://arxiv.org/abs/1904.09569
源码链接：http://mmcheng.net/poolnet/


26、CapSal: Leveraging Captioning to Boost Semantics for Salient Object Detection
作者：Lu Zhang; Huchuan Lu ; Zhe Lin ; Jianming Zhang; You He
论文链接：https://drive.google.com/open?id=1JcZMHBXEX-7AR1P010OXg_wCCC5HukeZ （需要申请）
源码链接：https://github.com/zhangludl/code-and-dataset-for-CapSal


27、Deep Fitting Degree Scoring Network for Monocular 3D Object Detection
作者：Lijie Liu1, Jiwen Lu, Chunjing Xu, Qi Tian, Jie Zhou
论文链接：https://arxiv.org/pdf/1904.12681.pdf


28、A Mutual Learning Method for Salient Object Detection with intertwined Multi-Supervision
作者：Runmin Wu, Mengyang Feng, Wenlong Guan, Dong Wang, Huchuan Lu, Errui Ding
论文链接：待定
源码链接：https://github.com/JosephineRabbit/MLMSNet


29、ScratchDet:Exploring to Train Single-Shot Object Detectors from Scratch(Oral)
作者：Rui Zhu, Shifeng Zhang, Xiaobo Wang, Longyin Wen, Hailin Shi, Liefeng Bo, Tao Mei
论文链接：https://arxiv.org/abs/1810.08425v3
源码链接：https://github.com/KimSoybean/ScratchDet
论文解读：CVPR 2019 Oral | 京东AI研究院提出 ScratchDet：随机初始化训练SSD目标检测器

### 分割 36
1、Attention-guided Unified Network for Panoptic Segmentation
作者：Yanwei Li, Xinze Chen, Zheng Zhu, Lingxi Xie, Guan Huang, Dalong Du, Xingang Wang
论文链接：https://arxiv.org/abs/1812.03904
论文解读：https://mp.weixin.qq.com/s/1tohID6SM3weS476XU5okw


2、FEELVOS: Fast End-to-End Embedding Learning for Video Object Segmentation
作者：Paul Voigtlaender, Yuning Chai, Florian Schroff, Hartwig Adam, Bastian Leibe, Liang-Chieh Chen
论文链接：https://arxiv.org/abs/1902.09513


3、Associatively Segmenting Instances and Semantics in Point Clouds
作者：Xinlong Wang, Shu Liu, Xiaoyong Shen, Chunhua Shen, Jiaya Jia
论文链接：https://arxiv.org/abs/1902.09852
代码链接：https://github.com/WXinlong/ASIS


4、3D-SIS: 3D Semantic Instance Segmentation of RGB-D Scans
作者：Ji Hou Angela Dai Matthias Nießner
论文链接：https://niessnerlab.org/projects/hou20183dsis.html
YouTube视频：https://youtu.be/IH9rNLD1-JE


5、Data augmentation using learned transforms for one-shot medical image segmentation
作者：Amy Zhao, Guha Balakrishnan, Frédo Durand, John V. Guttag, Adrian V. Dalca
论文链接：https://arxiv.org/abs/1902.09383


6、FickleNet: Weakly and Semi-supervised Semantic Image Segmentation using Stochastic Inference
作者：Jungbeom Lee, Eunji Kim, Sungmin Lee, Jangho Lee, Sungroh Yoon
论文链接：https://arxiv.org/abs/1902.10421



7、Dual Attention Network for Scene Segmentation
作者：Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang, Hanqing Lu
论文链接：https://arxiv.org/abs/1809.02983
Github源码：https://github.com/junfu1115/DANet


8、Mask Scoring R-CNN
作者：Zhaojin Huang, Lichao Huang, Yongchao Gong, Chang Huang, Xinggang Wang
论文链接：https://arxiv.org/abs/1903.00241
Github链接：https://github.com/zjhuang22/maskscoring_rcnn
论文解读：https://mp.weixin.qq.com/s/aP7O7AF6WoynWK_FFHkOTw


9、Hybrid Task Cascade for Instance Segmentation（实例分割）
作者：Kai Chen, Jiangmiao Pang, Jiaqi Wang, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jianping Shi, Wanli Ouyang, Chen Change Loy, Dahua Lin
论文链接：https://arxiv.org/abs/1901.07518
论文解读：https://mp.weixin.qq.com/s/xug0xKfc9RgJEUci1a_xog
Github链接：https://github.com/open-mmlab/mmdetection


10、Object Counting and Instance Segmentation with Image-level Supervision
作者：Hisham Cholakkal, Guolei Sun (equal contribution), Fahad Shahbaz Khan, Ling Shao
论文链接：https://arxiv.org/abs/1903.02494


11、MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation
作者：Yazan Abu Farha, Juergen Gall
论文链接：https://arxiv.org/abs/1903.01945


12、Structured Knowledge Distillation for Semantic Segmentation(语义分割）
作者：Yifan Liu, Ke Chen, Chris Liu, Zengchang Qin, Zhenbo Luo, Jingdong Wang
论文链接：https://arxiv.org/abs/1903.04197


13、RVOS: End-to-End Recurrent Network for Video Object Segmentation
作者：Carles Ventura, Miriam Bellver, Andreu Girbau, Amaia Salvador, Ferran Marques, Xavier Giro-i-Nieto
论文链接：https://arxiv.org/abs/1903.05612
项目链接：https://imatge-upc.github.io/rvos/


14、Structured Knowledge Distillation for Semantic Segmentation（语义分割）
作者：Yifan Liu, Ke Chen, Chris Liu, Zengchang Qin, Zhenbo Luo, Jingdong Wang
论文链接：https://arxiv.org/abs/1903.04197


15、Knowledge Adaptation for Efficient Semantic Segmentation（语义分割）
作者：Tong He, Chunhua Shen, Zhi Tian, Dong Gong, Changming Sun, Youliang Yan
论文链接：https://arxiv.org/abs/1903.04688


16、Improving Semantic Segmentation via Video Propagation and Label Relaxation(oral)
作者：Yi Zhu, Karan Sapra, Fitsum A. Reda, Kevin J. Shih, Shawn Newsam, Andrew Tao, Bryan Catanzaro
论文链接：https://arxiv.org/abs/1812.01593


17、In Defense of Pre-trained ImageNet Architectures for Real-time Semantic Segmentation of Road-driving Images
作者：Marin Oršić, Ivan Krešo, Petra Bevandić, Siniša Šegvić
论文链接：https://arxiv.org/abs/1903.08469
代码链接：https://github.com/orsic/swiftnet


18、Large-scale interactive object segmentation with human annotators
作者：Rodrigo Benenson, Stefan Popov, Vittorio Ferrari
论文链接：https://arxiv.org/abs/1903.10830
BubbleNets: Learning to Select the Guidance Frame in Video Object Segmentation by Deep Sorting Frames
作者：Brent A. Griffin, Jason J. Corso
论文链接：https://arxiv.org/abs/1903.11779


19、Pose2Seg: Detection Free Human Instance Segmentation
作者：Song-Hai Zhang, Ruilong Li, Xin Dong, Paul L. Rosin, Zixi Cai, Han Xi, Dingcheng Yang, Hao-Zhi Huang, Shi-Min Hu
论文链接：https://arxiv.org/abs/1803.10683
项目链接：http://www.liruilong.cn/Pose2Seg/index.html
代码链接：https://github.com/liruilong940607/OCHumanApi


20、BubbleNets: Learning to Select the Guidance Frame in Video Object Segmentation by Deep Sorting Frames
作者：Brent A. Griffin, Jason J. Corso
论文链接：https://arxiv.org/abs/1903.11779


21、JSIS3D: Joint Semantic-Instance Segmentation of 3D Point Clouds with Multi-Task Pointwise Networks and Multi-Value Conditional Random Fields（Oral)
作者：Quang-Hieu Pham, Duc Thanh Nguyen, Binh-Son Hua, Gemma Roig, Sai-Kit Yeung
论文链接：https://arxiv.org/abs/1904.00699
项目链接：https://pqhieu.github.io/cvpr19.html


22、Spatiotemporal CNN for Video Object Segmentation
作者：Kai Xu, Longyin Wen, Guorong Li, Liefeng Bo, Qingming Huang
论文链接：https://arxiv.org/abs/1904.02363
代码链接：https://github.com/longyin880815/STCNN


23、Data augmentation using learned transformsfor one-shot medical image segmentation
作者：Amy Zhao, Guha Balakrishnan, Frédo Durand, John V. Guttag, Adrian V. Dalca
论文链接：https://arxiv.org/pdf/1902.09383.pdf
源码链接：https://github.com/xamyzhao/brainstorm


24、DeepCO3: Deep Instance Co-segmentation by Co-peak Search and Co-saliency (Oral )
作者：Kuang-Jui Hsu, Yen-Yu Lin, Yung-Yu Chuang
论文链接：http://cvlab.citi.sinica.edu.tw/images/paper/cvpr-hsu19.pdf
源码链接：https://github.com/KuangJuiHsu/DeepCO3


25、Cross-Modal Self-Attention Network for Referring Image Segmentation
作者：Linwei Ye, Mrigank Rochan, Zhi Liu, Yang Wang
论文链接：https://arxiv.org/abs/1904.04745


26、Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations(Oral)
作者：Jiwoon Ahn, Sunghyun Cho, Suha Kwak
论文链接：https://arxiv.org/abs/1904.05044


27、Adaptive Weighting Multi-Field-of-View CNN for Semantic Segmentation in Pathology
作者：Hiroki Tokunaga, Yuki Teramoto, Akihiko Yoshizawa, Ryoma Bise
论文链接：https://arxiv.org/abs/1904.06040


28、A Relation-Augmented Fully Convolutional Network for Semantic Segmentationin Aerial Scenes
作者：Lichao Mou, Yuansheng Hua, Xiao Xiang Zhu
论文链接：https://arxiv.org/abs/1904.05730


29、DFANet：Deep Feature Aggregation for Real-Time Semantic Segmentation（旷视）
作者：Hanchao Li, Pengfei Xiong,Haoqiang Fan,Jian Sun
论文链接：https://share.weiyun.com/5NgHbWH


30、Exploiting Computation Power of Blockchain for Biomedical Image Segmentation
作者：Boyang Li, Changhao Chenli, Xiaowei Xu, Taeho Jung, Yiyu Shi
论文链接：https://arxiv.org/abs/1904.07349


31、MHP-VOS: Multiple Hypotheses Propagation for Video Object Segmentation(Oral)
作者：Shuangjie Xu, Daizong Liu, Linchao Bao, Wei Liu, Pan Zhou
论文链接：https://arxiv.org/abs/1904.08141


32、Machine Vision Guided 3D Medical Image Compression for Efficient Transmission and Accurate Segmentation in the Clouds
作者：Zihao Liu, Xiaowei Xu, Tao Liu, Qi Liu, Yanzhi Wang, Yiyu Shi, Wujie Wen, Meiping Huang, Haiyun Yuan, Jian Zhuang
论文链接：https://arxiv.org/abs/1904.08487


33、Fast User-Guided Video Object Segmentation by Interaction-and-Propagation Networks
作者：Seoung Wug Oh, Joon-Young Lee, Ning Xu, Seon Joo Kim
论文链接：https://arxiv.org/abs/1904.09791


34、Box-driven Class-wise Region Masking and Filling Rate Guided Loss for Weakly Supervised Semantic Segmentation
作者：Chunfeng Song, Yan Huang, Wanli Ouyang, Liang Wang
论文链接：https://arxiv.org/abs/1904.11693


35、Bidirectional Learning for Domain Adaptation of Semantic Segmentation
作者：Yunsheng Li, Lu Yuan, Nuno Vasconcelos
论文链接：https://arxiv.org/abs/1904.10620
源码链接：https://github.com/liyunsheng13/BDL


36、Learning Unsupervised Video Primary Object Segmentation through Visual Attention
作者：Wenguan Wang, Hongmei Song, Shuyang Zhao, Jianbing Shen, Sanyuan Zhao, Steven Chu Hong Hoi, and Haibin Ling
论文链接：http://www.dabi.temple.edu/~hbling/publication/UVOS-cvpr19.pdf
源码链接：https://github.com/wenguanwang/AGS

### 分类、识别 17
1、Learning a Deep ConvNet for Multi-label Classification with Partial Labels(分类)
作者：Thibaut Durand, Nazanin Mehrasa, Greg Mori
论文链接：https://arxiv.org/abs/1902.09720


2、Efficient Video Classification Using Fewer Frames
作者：Shweta Bhardwaj, Mukundhan Srinivasan, Mitesh M. Khapra
论文链接：https://arxiv.org/abs/1902.10640


3、Weakly Supervised Complementary Parts Models for Fine-Grained Image Classification from the Bottom Up
作者：Weifeng Ge, Xiangru Lin, Yizhou Yu
论文链接：https://arxiv.org/abs/1903.02827


4、All You Need is a Few Shifts: Designing Efficient Convolutional Neural Networks for Image Classification（分类）
作者：Weijie Chen, Di Xie, Yuan Zhang, Shiliang Pu
论文链接：https://arxiv.org/abs/1903.05285


5、Bag of Tricks for Image Classification with Convolutional Neural Networks
作者：Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li
论文链接：https://arxiv.org/abs/1812.01187
源码链接：https://github.com/dmlc/gluon-cv
论文解读：图像分类技巧：Bag of Tricks for Image Classification with Convolutional Neural Networks


6、Direct Object Recognition Without Line-of-Sight Using Optical Coherence(目标识别）
作者：Xin Lei, Liangyu He, Yixuan Tan, Ken Xingze Wang, Xinggang Wang, Yihan Du, Shanhui Fan, Zongfu Yu
论文链接：https://arxiv.org/abs/1903.07705


7、Direct Object Recognition Without Line-of-Sight Using Optical Coherence(非视距物体识别技术)
作者：Xin Lei, Liangyu He, Yixuan Tan, Ken Xingze Wang, Xinggang Wang, Yihan Du, Shanhui Fan, Zongfu Yu
论文链接：https://arxiv.org/abs/1903.07705


8、C2AE: Class Conditioned Auto-Encoder for Open-set Recognition(Oral)
作者：Poojan Oza, Vishal M Patel
论文链接：https://arxiv.org/abs/1904.01198


9、Multispectral Imaging for Fine-Grained Recognition of Powders on Complex Backgrounds(Oral)
作者：Tiancheng Zhi, Bernardo R. Pires, Martial Hebert and Srinivasa G. Narasimhan
论文链接：http://www.cs.cmu.edu/~ILIM/projects/IM/MSPowder/files/ZPHN-CVPR19.pdf
代码链接：https://github.com/tiancheng-zhi/ms-powder
项目链接：http://www.cs.cmu.edu/~ILIM/projects/IM/MSPowder/


10、Large-Scale Long-Tailed Recognition in an Open World（Oral)
作者：Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, Stella X. Yu
论文链接：https://github.com/ofsoundof/3D_Appearance_SR/blob/master/code/scripts/3d_appearance_sr.pdf
源码链接：https://github.com/zhmiao/OpenLongTailRecognition-OLTR


11、Multi-Label Image Recognition with Graph Convolutional Networks（多标签图像识别）
作者：Zhao-Min Chen, Xiu-Shen Wei, Peng Wang, Yanwen Guo
论文链接：https://arxiv.org/abs/1904.03582
源码链接：https://github.com/chenzhaomin123/ML_GCN
简介：本工作针对多标记识别的核心问题，即“如何有效建模标记间的协同关系”进行探索，提出基于图卷积（GCN）的端到端系统，通过data-driven方式建立标记间有向图（directed graph）并由GCN将类别标记映射（mapping）为对应类别分类器，以此建模类别关系，同时可提升表示学习能力。此外针对GCN中的关键元素correlation matrix进行了深入分析和重设计，使其更胜任多标记问题。


12、Gait Recognition via Disentangled Representation Learning（Oral 步态识别）
作者：Ziyuan Zhang, Luan Tran, Xi Yin, Yousef Atoum, Xiaoming Liu, Jian Wan, Nanxin Wang
论文链接：https://arxiv.org/abs/1904.04925


13、Adaptively Connected Neural Networks（分类）
作者：Guangrun Wang, Keze Wang, Liang Lin
论文链接：https://arxiv.org/abs/1904.03579
源码链接：https://github.com/wanggrun/Adaptively-Connected-Neural-Networks


14、Aggregation Cross-Entropy for Sequence Recognition
作者：Zecheng Xie, Yaoxiong Huang, Yuanzhi Zhu, Lianwen Jin, Yuliang Liu, Lele Xie
论文链接：https://arxiv.org/abs/1904.08364


15、Meta-learning Convolutional Neural Architectures for Multi-target Concrete Defect Classification with the COncrete DEfect BRidge IMage Dataset
作者：Martin Mundt, Sagnik Majumder, Sreenivas Murali, Panagiotis Panetsos, Visvanathan Ramesh
论文链接：https://arxiv.org/abs/1904.08486


16、Unsupervised Open Domain Recognition by Semantic Discrepancy Minimization
作者：Junbao Zhuo, Shuhui Wang, Shuhao Cui, Qingming Huang
论文链接：https://arxiv.org/abs/1904.08631


17、Translate-to-Recognize Networks for RGB-D Scene Recognition
作者：Dapeng Du, Limin Wang, Huiling Wang, Kai Zhao, Gangshan Wu
论文链接：https://arxiv.org/abs/1904.12254
源码链接：

https://ownstyledu.github.io/Translate-to-Recognize-Networks/


### 跟踪 19
1、Fast Online Object Tracking and Segmentation: A Unifying Approach(SiamMask,目标跟踪）
作者：Qiang Wang, Li Zhang, Luca Bertinetto, Weiming Hu, Philip H.S. Torr
论文链接：https://arxiv.org/abs/1812.05050
Github链接：https://github.com/foolwood/SiamMask
project链接：http://www.robots.ox.ac.uk/~qwang/SiamMask/
论文解读：CVPR2019 | SiamMask：视频跟踪最高精度


2、Deeper and Wider Siamese Networks for Real-Time Visual Tracking(CIR,目标跟踪）
作者：Zhipeng Zhang, Houwen Peng
论文链接：https://arxiv.org/pdf/1901.01660.pdf
Code链接：https://gitlab.com/MSRA_NLPR/deeper_wider_siamese_trackers


3、SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks(目标跟踪）
作者：Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, Junjie Yan
论文链接：https://arxiv.org/pdf/1812.11703.pdf
Project链接：http://bo-li.info/SiamRPN++/
论文解读：https://mp.weixin.qq.com/s/dB5u2No8eakLnrjto0kvyQ


4、Siamese Cascaded Region Proposal Networks for Real-Time Visual Tracking(CRPN,目标跟踪）
作者：Heng Fan, Haibin Ling
论文链接：https://arxiv.org/pdf/1812.06148.pdf


5、LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking(目标跟踪）
作者：Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, Haibin Ling
论文链接：https://arxiv.org/pdf/1809.07845.pdf
project链接：https://cis.temple.edu/lasot/


6、Leveraging Shape Completion for 3D Siamese Tracking
作者：Silvio Giancola, Jesus Zarzar, Bernard Ghanem
论文链接：https://arxiv.org/abs/1903.01784


7、Cross-Classification Clustering: An Efficient Multi-Object Tracking Technique for 3-D Instance Segmentation in Connectomics（多目标跟踪)
作者：Yaron Meirovitch, Lu Mi, Hayk Saribekyan, Alexander Matveev, David Rolnick, Casimir Wierzynski, Nir Shavit
论文链接：https://arxiv.org/abs/1812.01157


8、Multiview 2D/3D Rigid Registration via a Point-Of-Interest Network for Tracking and Triangulation (POINT^2)
作者：Haofu Liao, Wei-An Lin, Jiarui Zhang, Jingdan Zhang, Jiebo Luo, S. Kevin Zhou
论文链接：https://arxiv.org/abs/1903.03896


9、Inverse Path Tracing for Joint Material and Lighting Estimation(Oral)
作者：Jiaxin Cheng, Yue Wu, Wael Abd-Almageed, Premkumar Natarajan
论文链接：https://arxiv.org/abs/1903.07145


10、Inverse Path Tracing for Joint Material and Lighting Estimation(Oral)
作者：Jiaxin Cheng, Yue Wu, Wael Abd-Almageed, Premkumar Natarajan
论文链接：https://arxiv.org/abs/1903.07145


11、Multi-person Articulated Tracking with Spatial and Temporal Embeddings
作者：Sheng Jin, Wentao Liu, Wanli Ouyang, Chen Qian
论文链接：https://arxiv.org/abs/1903.09214


12、CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification
作者：Zheng Tang, Milind Naphade, Ming-Yu Liu, Xiaodong Yang, Stan Birchfield, Shuo Wang, Ratnesh Kumar, David Anastasiu, Jenq-Neng Hwang
论文链接：https://arxiv.org/abs/1903.09254


13、MOTS: Multi-Object Tracking and Segmentation
作者：Paul Voigtlaender, Michael Krause, Aljosa Osep, Jonathon Luiten, Berin Balachandar Gnana Sekar, Andreas Geiger, Bastian Leibe
论文链接：https://arxiv.org/abs/1902.03604


14、Target-Aware Deep Tracking
作者：Xin Li, Chao Ma, Baoyuan Wu, Zhenyu He, Ming-Hsuan Yang
论文链接：https://arxiv.org/pdf/1904.01772.pdf


15、Unsupervised Deep Tracking
作者：Ning Wang, Yibing Song, Chao Ma, Wengang Zhou, Wei Liu, Houqiang Li
论文链接：https://arxiv.org/pdf/1904.01828.pdf


16、Beyond Tracking: Selecting Memory and Refining Poses for Deep Visual Odometry（Oral)
作者：Fei Xue, Xin Wang, Shunkai Li, Qiuyuan Wang, Junqiu Wang, Hongbin Zha
论文链接：https://arxiv.org/abs/1904.01892


17、SPM-Tracker: Series-Parallel Matching for Real-Time Visual Object Tracking（视觉跟踪）
作者：Guangting Wang, Chong Luo, Zhiwei Xiong, Wenjun Zeng
论文链接：https://arxiv.org/abs/1904.04452


18、Graph Convolutional Tracking
作者：Junyu Gao，Tianzhu Zhang，Changsheng Xu
论文链接：http://nlpr-web.ia.ac.cn/mmc/homepage/jygao/gct_cvpr2019.html


19、ATOM: Accurate Tracking by Overlap Maximization(Oral,目标跟踪)
作者：Martin Danelljan, Goutam Bhat, Fahad Shahbaz Khan, Michael Felsberg
论文链接：https://arxiv.org/abs/1811.07628
源码链接：https://github.com/visionml/pytracking

### 人脸 9
1、Disentangled Representation Learning for 3D Face Shape
作者：Baris Gecer, Stylianos Ploumpis, Irene Kotsia, Stefanos Zafeiriou
论文链接：https://arxiv.org/abs/1902.05978


2、Joint Face Detection and Facial Motion Retargeting for Multiple Faces
作者：Bindita Chaudhuri, Noranart Vesdapunt, Baoyuan Wang
论文链接：https://arxiv.org/abs/1902.10744


3、ArcFace: Additive Angular Margin Loss for Deep Face Recognition（人脸识别）
作者：Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou
论文链接：https://arxiv.org/abs/1801.07698
Demo链接：https://github.com/vita-epfl/openpifpafwebdemo


4、Linkage Based Face Clustering via Graph Convolution Network
作者：Zhongdao Wang, Liang Zheng, Yali Li, Shengjin Wang
论文链接：https://arxiv.org/abs/1903.11306


5、Learning to Cluster Faces on an Affinity Graph
作者：Lei Yang, Xiaohang Zhan, Dapeng Chen, Junjie Yan, Chen Change Loy, Dahua Lin
论文链接：https://arxiv.org/abs/1904.02749


6、Deep Tree Learning for Zero-shot Face Anti-Spoofing(Oral)
作者：Yaojie Liu, Joel Stehouwer, Amin Jourabloo, Xiaoming Liu
论文链接：https://arxiv.org/abs/1904.02860


7、Efficient Decision-based Black-box Adversarial Attacks on Face Recognition（人脸识别）
作者：Yinpeng Dong, Hang Su, Baoyuan Wu, Zhifeng Li, Wei Liu, Tong Zhang, Jun Zhu
论文链接：https://arxiv.org/abs/1904.04433


8、Towards High-fidelity Nonlinear 3D Face Morphable Model
作者：Luan Tran, Feng Liu, Xiaoming Liu
论文链接：https://arxiv.org/abs/1904.04933
项目链接：http://cvlab.cse.msu.edu/project-nonlinear-3dmm.html


9、LBVCNN: Local Binary Volume Convolutional Neural Network for Facial Expression Recognition from Image Sequences
作者：Sudhakar Kumawat, Manisha Verma, Shanmuganathan Raman
论文链接：https://arxiv.org/abs/1904.07647

### 人体姿态估计、位姿估计 18
1、Deep High-Resolution Representation Learning for Human Pose Estimation(目前SOTA,已经开源)
作者：Ke Sun, Bin Xiao, Dong Liu, Jingdong Wang
论文链接：https://128.84.21.199/abs/1902.09212
代码链接：https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
论文解读：https://mp.weixin.qq.com/s/ZRCzBTBmlEzQCVo1HLWtbQ


2、DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion
作者：Chen Wang, Danfei Xu, Yuke Zhu, Roberto Martín-Martín, Cewu Lu, Li Fei-Fei, Silvio Savarese
论文链接：https://arxiv.org/abs/1901.04780
论文解读：https://mp.weixin.qq.com/s/wrND2cocWlPPVXPqpq-Glg


3、RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D Human Pose Estimation
作者：Bastian Wandt, Bodo Rosenhahn
论文链接：https://arxiv.org/abs/1902.09868


4、3D Hand Shape and Pose Estimation from a Single RGB Image
作者：Liuhao Ge, Zhou Ren, Yuncheng Li, Zehao Xue, Yingying Wang, Jianfei Cai, Junsong Yuan
论文链接：https://arxiv.org/abs/1903.00812


5、Self-Supervised Learning of 3D Human Pose using Multi-view Geometry
作者：Muhammed Kocabas, Salih Karagoz, Emre Akbas
论文链接：https://arxiv.org/abs/1903.02330
Github链接：https://github.com/mkocabas/EpipolarPose


6、Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views
作者：Junting Dong, Wen Jiang, Qixing Huang, Hujun Bao, Xiaowei Zhou
论文链接：https://arxiv.org/abs/1901.04111
项目链接：https://zju-3dv.github.io/mvpose/
代码链接：https://github.com/zju-3dv/mvpose


7、Extreme Relative Pose Estimation for RGB-D Scans via Scene Completion (Oral)
作者：Zhenpei Yang, Jeffrey Z.Pan, Linjie Luo, Xiaowei Zhou, Kristen Grauman and Qixing Huang
论文链接：https://arxiv.org/pdf/1901.00063.pdf
代码链接: https://github.com/zhenpeiyang/RelativePose


8、PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation
作者：Sida Peng, Yuan Liu, Qixing Huang, Hujun Bao, and Xiaowei Zhou
论文链接：https://arxiv.org/pdf/1812.11788.pdf


9、PoseFix: Model-agnostic General Human Pose Refinement Network
作者：Gyeongsik Moon, Ju Yong Chang, Kyoung Mu Lee
论文链接：https://arxiv.org/abs/1812.03595
源码链接：https://github.com/mks0601/PoseFix_RELEASE


10、Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation(oral)
作者：He Wang, Srinath Sridhar, Jingwei Huang, Julien Valentin, Shuran Song, Leonidas J. Guibas
论文链接：https://arxiv.org/abs/1901.02970


11、PifPaf: Composite Fields for Human Pose Estimation(姿态估计）
作者：Sven Kreiss, Lorenzo Bertoni, Alexandre Alahi
论文链接：https://arxiv.org/abs/1903.06593
Demo链接：https://github.com/vita-epfl/openpifpafwebdemo


12、Weakly-Supervised Discovery of Geometry-Aware Representation for 3D Human Pose Estimation(Oral，3D姿态估计)
作者：Xipeng Chen, Kwan-Yee Lin, Wentao Liu, Chen Qian, Liang Lin
论文链接：https://arxiv.org/abs/1903.08839


13、CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark
作者：Jiefeng Li, Can Wang, Hao Zhu, Yihuan Mao, Hao-Shu Fang, Cewu Lu
论文链接：https://arxiv.org/abs/1812.00324
代码链接：https://github.com/Jeff-sjtu/CrowdPose


14、Dense Intrinsic Appearance Flow for Human Pose Transfer
作者：Yining Li, Chen Huang, Chen Change Loy
论文链接：https://arxiv.org/abs/1903.11326


15、Generating Multiple Hypotheses for 3D Human Pose Estimation with Mixture Density Network
作者：Chen Li, Gim Hee Lee
论文链接：https://arxiv.org/abs/1904.05547


16、FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image
作者：Tsun-Yi Yang, Yi-Ting Chen, Yen-Yu Lin, and Yung-Yu Chuang
论文链接：https://github.com/shamangary/FSA-Net/blob/master/0191.pdf
源码链接：https://github.com/shamangary/FSA-Net


17、Segmentation-driven 6D Object Pose Estimation
作者：Yinlin Hu, Joachim Hugonot, Pascal Fua, Mathieu Salzmann
论文链接：https://arxiv.org/abs/1812.02541
源码链接：https://github.com/cvlab-epfl/segmentation-driven-pose


18、Progressive Pose Attention Transfer for Person Image Generation
作者：Zhen Zhu, Tengteng Huang, Baoguang Shi, Miao Yu, Bofei Wang, Xiang Bai
论文链接：https://arxiv.org/abs/1904.03349
源码链接：https://github.com/tengteng95/Pose-Transfer

### 行为/动作识别、手势识别 12
1、An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition
作者：Chenyang Si, Wentao Chen, Wei Wang, Liang Wang, Tieniu Tan
论文链接：https://arxiv.org/abs/1902.09130


2、Improving the Performance of Unimodal Dynamic Hand-Gesture Recognition with Multimodal Training
作者：Mahdi Abavisani, Hamid Reza Vaezi Joze, Vishal M. Patel
链接：https://arxiv.org/abs/1812.06145


3、Collaborative Spatio-temporal Feature Learning for Video Action Recognition
作者：Chao Li, Qiaoyong Zhong, Di Xie, Shiliang Pu
论文链接：https://arxiv.org/abs/1903.01197


4、Peeking into the Future: Predicting Future Person Activities and Locations in Videos(行为预测）
作者：Junwei Liang, Lu Jiang, Juan Carlos Niebles, Alexander Hauptmann, Li Fei-Fei
论文链接：https://arxiv.org/abs/1902.03748


5、Neural Scene Decomposition for Multi-Person Motion Capture
作者：Helge Rhodin, Victor Constantin, Isinsu Katircioglu, Mathieu Salzmann, Pascal Fua
论文链接：https://arxiv.org/abs/1903.05684


6、Action Recognition from Single Timestamp Supervision in Untrimmed Videos（动作识别）
作者：Davide Moltisanti, Sanja Fidler, Dima Damen
论文链接：https://arxiv.org/abs/1904.04689


7、Pushing the Envelope for RGB-based Dense 3D Hand Pose Estimation via Neural Rendering
作者：Seungryul Baek, Kwang In Kim, Tae-Kyun Kim
论文链接：https://arxiv.org/abs/1904.04196


8、Relational Action Forecasting(oral)
作者：Chen Sun, Abhinav Shrivastava, Carl Vondrick, Rahul Sukthankar, Kevin Murphy, Cordelia Schmid
论文链接：https://arxiv.org/abs/1904.04231


9、H+O: Unified Egocentric Recognition of 3D Hand-Object Poses and Interactions(Oral)
作者：Bugra Tekin, Federica Bogo, Marc Pollefeys
论文链接：https://arxiv.org/abs/1904.05349



10、Out-of-Distribution Detection for Generalized Zero-Shot Action Recognition
作者：Devraj Mandal, Sanath Narayan, Saikumar Dwivedi, Vikram Gupta, Shuaib Ahmed, Fahad Shahbaz Khan, Ling Shao
论文链接：https://arxiv.org/abs/1904.08703


11、Actional-Structural Graph Convolutional Networks for Skeleton-based Action Recognition
作者：Maosen Li, Siheng Chen, Xu Chen, Ya Zhang, Yanfeng Wang, and Qi Tian
论文链接：https://arxiv.org/pdf/1904.12659.pdf


12、A neural network based on SPD manifold learning for skeleton-based hand gesture recognition
作者：Xuan Son Nguyen, Luc Brun, Olivier Lézoray, Sébastien Bougleux
论文链接：https://arxiv.org/abs/1904.12970

### 时序动作检测、视频相关 18
1、Spatio-Temporal Dynamics and Semantic Attribute Enriched Visual Encoding for Video Captioning
作者：Nayyer Aafaq, Naveed Akhtar, Wei Liu, Syed Zulqarnain Gilani, Ajmal Mian
论文链接：https://arxiv.org/abs/1902.10322
来源：https://mp.weixin.qq.com/s/61C-k3Ijy_7ry5B5lRML6Q


2、Single-frame Regularization for Temporally Stable CNNs（视频处理）
作者：Gabriel Eilertsen, Rafał K. Mantiuk, Jonas Unger
论文链接：https://arxiv.org/abs/1902.10424
来源：https://mp.weixin.qq.com/s/61C-k3Ijy_7ry5B5lRML6Q


3、Neural RGB-D Sensing: Depth estimation from a video
作者：Chao Liu, Jinwei Gu, Kihwan Kim, Srinivasa Narasimhan, Jan Kautz
论文链接：https://arxiv.org/pdf/1901.02571.pdf
project链接：https://research.nvidia.com/publication/2019-06_Neural-RGBD



4、Competitive Collaboration: Joint Unsupervised Learning of Depth, CameraMotion, Optical Flow and Motion Segmentation
作者：Anurag Ranjan, Varun Jampani, Kihwan Kim, Deqing Sun, Jonas Wulff, Michael J. Black
论文链接：https://arxiv.org/pdf/1805.09806.pdf



5、Representation Flow for Action Recognition
作者：AJ Piergiovanni, Michael S. Ryoo
论文链接：https://arxiv.org/abs/1810.01455
项目链接：https://piergiaj.github.io/rep-flow-site/
代码链接：https://github.com/piergiaj/representation-flow-cvpr19


6、Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos
作者：Romero Morais, Vuong Le, Truyen Tran, Budhaditya Saha, Moussa Mansour, Svetha Venkatesh
论文链接：https://arxiv.org/abs/1903.03295


7、Video Generation from Single Semantic Label Map
作者：Junting Pan, Chengyu Wang, Xu Jia, Jing Shao, Lu Sheng, Junjie Yan, Xiaogang Wang
论文链接：https://arxiv.org/abs/1903.04480
源码链接：https://github.com/junting/seg2vid/tree/master


8、Inserting Videos into Videos
作者：Donghoon Lee, Tomas Pfister, Ming-Hsuan Yang
论文链接：https://arxiv.org/abs/1903.06571


9、Recurrent Back-Projection Network for Video Super-Resolution
作者：Muhammad Haris, Greg Shakhnarovich, Norimichi Ukita
论文链接：https://alterzero.github.io/projects/rbpn_cvpr2019.pdf
代码链接：https://github.com/alterzero/RBPN-PyTorch
项目链接：https://alterzero.github.io/projects/RBPN.html


10、Depth-Aware Video Frame Interpolation
作者：Wenbo Bao Wei-Sheng Lai, Chao Ma, Xiaoyun Zhang, Zhiyong Gao, and Ming-Hsuan Yang
论文链接：https://sites.google.com/view/wenbobao/dain
代码链接：https://github.com/baowenbo/DAIN



11、Video Relationship Reasoning using Gated Spatio-Temporal Energy Graph
作者：Yao-Hung Hubert Tsai, Santosh Divvala, Louis-Philippe Morency, Ruslan Salakhutdinov, Ali Farhadi
论文链接：https://arxiv.org/abs/1903.10547


12、Dual Encoding for Zero-Example Video Retrieval
作者：Jianfeng Dong, Xirong Li, Chaoxi Xu, Shouling Ji, Yuan He, Gang Yang and Xun Wang
论文链接：https://arxiv.org/abs/1809.06181
代码链接：https://github.com/danieljf24/dual_encoding


13、Rethinking the Evaluation of Video Summaries
作者：Jacques Manderscheid, Amos Sironi, Nicolas Bourdis, Davide Migliore, Vincent Lepetit
论文链接：https://arxiv.org/abs/1903.11328


14、End-to-End Time-Lapse Video Synthesis from a Single Outdoor Image
作者：Seonghyeon Nam, Chongyang Ma, Menglei Chai, William Brendel, Ning Xu, Seon Joo Kim
论文链接：https://arxiv.org/abs/1904.00680


15、GolfDB: A Video Database for Golf Swing Sequencing
作者：William McNally, Kanav Vats, Tyler Pinto, Chris Dulhanty, John McPhee, Alexander Wong
论文链接：https://arxiv.org/abs/1903.06528v1


16、VORNet: Spatio-temporally Consistent Video Inpainting for Object Removal
作者：Ya-Liang Chang, Zhe Yu Liu, Winston Hsu
论文链接：https://arxiv.org/abs/1904.06726


17、STEP: Spatio-Temporal Progressive Learning for Video Action Detection（Oral）
作者：Xitong Yang, Xiaodong Yang, Ming-Yu Liu, Fanyi Xiao, Larry Davis, Jan Kautz
论文链接：https://arxiv.org/abs/1904.09288


18、UnOS: Unified Unsupervised Optical-flow and Stereo-depth Estimation by Watching Videos
作者：Yang Wang, Peng Wang, Zhenheng Yang, Chenxu Luo, Yi Yang, and Wei Xu
论文链接：https://arxiv.org/abs/1810.03654

### Related to Networks 34
1、RePr: Improved Training of Convolutional Filters
作者：Aaditya Prakash, James Storer, Dinei Florencio, Cha Zhang
论文链接：https://arxiv.org/abs/1811.07275


2、Iterative Residual CNNs for Burst Photography Applications
作者：Filippos Kokkinos   Stamatis Lefkimmiatis
论文链接：https://arxiv.org/abs/1811.12197


3、SpherePHD: Applying CNNs on a Spherical PolyHeDron Representation of 360 degree Images
作者：Yeon Kun Lee, Jaeseok Jeong, Jong Seob Yun, Cho Won June, Kuk-Jin Yoon
论文链接：https://arxiv.org/abs/1811.08196


4、On the Continuity of Rotation Representations in Neural Networks
作者：Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li
论文链接：https://arxiv.org/pdf/1812.07035.pdf


5、Binary Ensemble Neural Network: More Bits per Network or More Networks per Bit?
作者：Shilin Zhu, Xin Dong, Hao Su
论文链接：https://arxiv.org/abs/1806.07550
简要：Ensemble of binary neural networks has better stability and robustness, and may perform as well as floating-point networks.


6、A Neurobiological Evaluation Metric for Neural Network Model Search
作者：Nathaniel Blanchard, Jeffery Kinnison, Brandon RichardWebster, Pouya Bashivan, Walter J. Scheirer
论文链接：https://arxiv.org/pdf/1805.10726.pdf


7、MAN: Moment Alignment Network for Natural Language Moment Retrieval via Iterative Graph Adjustment
作者：Da Zhang, Xiyang Dai, Xin Wang, Yuan-Fang Wang, Larry S. Davis
论文链接：https://arxiv.org/pdf/1812.00087.pdf


8、Multi-Step Prediction of Occupancy Grid Maps with Recurrent Neural Networks
作者：Nima Mohajerin, Mohsen Rohani
论文链接：https://arxiv.org/pdf/1812.09395.pdf


9、Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem（oral)
作者：Matthias Hein, Maksym Andriushchenko, Julian Bitterwolf
论文链接：https://arxiv.org/abs/1812.05720
Reading Note:In the paper, we give a theoretical argument of why ReLU activation can lead to models with overconfident predictions. Moreover, we propose a robust optimization training scheme that mitigates this problem.


10、RGBD Based Dimensional Decomposition Residual Network for 3D Semantic Scene Completion
作者：Jie Li, Yu Liu, Dong Gong, Qinfeng Shi, Xia Yuan, Chunxia Zhao, Ian Reid
论文链接：https://arxiv.org/abs/1903.00620


11、PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation
作者：Fenggen Yu, Kun Liu, Yan Zhang, Chenyang Zhu, Kai Xu
论文链接：https://arxiv.org/abs/1903.00709


12、3D Point-Capsule Networks
作者：Yongheng Zhao, Tolga Birdal, Haowen Deng, Federico Tombari
论文链接：https://arxiv.org/abs/1812.10775


13、CANet: Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive Few-Shot Learning
作者：Chi Zhang, Guosheng Lin, Fayao Liu, Rui Yao, Chunhua Shen
论文链接：https://arxiv.org/abs/1903.02351


14、Path-Invariant Map Networks (Oral)
作者：Zaiwei Zhang, Zhenxiao Liang, Lemeng Wu, Xiaowei Zhou and Qixing Huang
论文链接：https://arxiv.org/pdf/1812.11647.pdf
代码链接: https://github.com/zaiweizhang/path_invariance_map_network


15、A Main/Subsidiary Network Framework for Simplifying Binary Neural Network
作者：Yinghao Xu, Xin Dong, Yudian Li, Hao Su
论文链接：https://arxiv.org/abs/1812.04210
简要：A simple learning-based binary neural network pruning scheme.


16、Knowledge-Embedded Routing Network for Scene Graph Generation
作者：Tianshui Chen, Weihao Yu, Riquan Chen, Liang Lin
论文链接：https://arxiv.org/abs/1903.03326


17、Knowledge-Embedded Routing Network for Scene Graph Generation
作者：Tianshui Chen, Weihao Yu, Riquan Chen, Liang Lin
论文链接：https://arxiv.org/abs/1903.03326


18、HetConv: Heterogeneous Kernel-Based Convolutions for Deep CNNs
作者：Pravendra Singh, Vinay Kumar Verma, Piyush Rai, Vinay P. Namboodiri
论文链接：https://arxiv.org/abs/1903.04120



19、Large-scale Distributed Second-order Optimization Using Kronecker-factored Approximate Curvature for Deep Convolutional Neural Networks
作者：Kazuki Osawa, Yohei Tsuji, Yuichiro Ueno, Akira Naruse, Rio Yokota, Satoshi Matsuoka
论文链接：https://arxiv.org/abs/1811.12019


20、ADCrowdNet: An Attention-injective Deformable Convolutional Network for Crowd Understanding
作者：Ning Liu, Yongchao Long, Changqing Zou, Qun Niu, Li Pan, Hefeng Wu
论文链接：https://arxiv.org/abs/1811.11968


21、LaSO: Label-Set Operations networks for multi-label few-shot learning(oral)
作者：Amit Alfassy, Leonid Karlinsky, Amit Aides, Joseph Shtok, Sivan Harary, Rogerio Feris, Raja Giryes, Alex M. Bronstein
论文链接：https://arxiv.org/abs/1902.09811


22、Selective Kernel Networks
作者：Xiang Li, Wenhai Wang, Xiaolin Hu, Jian Yang
论文链接：https://arxiv.org/abs/1903.06586
源码链接：https://github.com/implus/SKNet


23、Self-calibrating Deep Photometric Stereo Networks(Oral)
作者：Guanying Chen, Kai Han, Boxin Shi, Yasuyuki Matsushita, Kwan-Yee K. Wong
论文链接：https://arxiv.org/abs/1903.07366
项目链接：http://gychen.org/SDPS-Net/
代码链接：https://github.com/guanyingc/SDPS-Net


24、Self-calibrating Deep Photometric Stereo Networks(Oral)
作者：Guanying Chen, Kai Han, Boxin Shi, Yasuyuki Matsushita, Kwan-Yee K. Wong
论文链接：https://arxiv.org/abs/1903.07366
项目链接：http://gychen.org/SDPS-Net/
代码链接：https://github.com/guanyingc/SDPS-Net


25、Networks for Joint Affine and Non-parametric Image Registration
作者：Zhengyang Shen, Xu Han, Zhenlin Xu, Marc Niethammer
论文链接：https://arxiv.org/abs/1903.08811


26、Learning for Single-Shot Confidence Calibration in Deep Neural Networks through Stochastic Inferences
作者：Seonguk Seo, Paul Hongsuck Seo, Bohyung Han
论文链接：https://arxiv.org/abs/1810.02358


27、Towards Optimal Structured CNN Pruning via Generative Adversarial Learning
作者：Shaohui Lin, Rongrong Ji, Chenqian Yan, Baochang Zhang, Liujuan Cao, Qixiang Ye, Feiyue Huang, David Doermann
论文链接：https://arxiv.org/abs/1903.09291


28、TIN: Transferable Interactiveness Network
作者：Yong-Lu Li, Siyuan Zhou, Xijie Huang, Liang Xu, Ze Ma, Hao-Shu Fang, Yan-Feng Wang, Cewu Lu
论文链接：https://arxiv.org/abs/1811.08264
代码链接：https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network


29、Convolutional Neural Networks Deceived by Visual Illusions
作者：Alexander Gomez-Villa, Adrián Martín, Javier Vazquez-Corral, Marcelo Bertalmío
论文链接：https://arxiv.org/abs/1811.10565


30、Fully Learnable Group Convolution for Acceleration of Deep Neural Networks
作者：Xijun Wang, Meina Kan, Shiguang Shan, Xilin Chen
论文链接：https://arxiv.org/abs/1904.00346


31、Kervolutional Neural Networks
作者：Chen Wang, Jianfei Yang, Lihua Xie, Junsong Yuan
论文链接：https://arxiv.org/abs/1904.03955


32、Pixel-Adaptive Convolutional Neural Networks
作者：Hang Su, Varun Jampani, Deqing Sun, Orazio Gallo, Erik Learned-Miller, Jan Kautz
论文链接：https://arxiv.org/abs/1904.05373


33、Binary Ensemble Neural Network: More Bits per Network or More Networks per Bit?
作者：Shilin Zhu, Xin Dong, Hao Su
论文链接：https://arxiv.org/abs/1806.07550


34、Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration(Oral)
作者：Yang He, Ping Liu, Ziwei Wang, Zhilan Hu, Yi Yang
论文链接：https://arxiv.org/abs/1811.00250
源码链接：https://github.com/he-y/filter-pruning-geometric-median


35、D2-Net: A Trainable CNN for Joint Detection and Description of Local Features
作者：Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla, Marc Pollefeys, Josef Sivic, Akihiko Torii, Torsten Sattler
论文链接：https://dsmn.ml/publications/d2-net.html
源码链接：https://github.com/mihaidusmanu/d2-net

### GAN、图像文本生成 21
1、Event-based High Dynamic Range Image and Very High Frame Rate Video Generation using Conditional Generative Adversarial Networks
作者：S. Mohammad Mostafavi I., Lin Wang, Yo-Sung Ho, Kuk-Jin Yoon
论文链接：https://arxiv.org/abs/1811.08230


2、Mixture Density Generative Adversarial Networks
作者：Hamid Eghbal-zadeh, Werner Zellinger, Gerhard Widmer
论文链接：https://arxiv.org/abs/1811.00152


3、GANFIT: Generative Adversarial Network Fitting for High Fidelity 3D Face Reconstruction
作者：Baris Gecer, Stylianos Ploumpis, Irene Kotsia, Stefanos Zafeiriou
论文链接：https://arxiv.org/abs/1902.05978
github链接：https://github.com/barisgecer/ganfit


4、Self-Supervised Generative Adversarial Networks
作者：Ting Chen, Xiaohua Zhai, Marvin Ritter, Mario Lucic, Neil Houlsby
论文链接：https://arxiv.org/abs/1811.11212
Github链接：https://github.com/google/compare_gan


5、CollaGAN : Collaborative GAN for Missing Image Data Imputation
作者：Dongwook Lee, Junyoung Kim, Won-Jin Moon, Jong Chul Ye
论文链接：https://arxiv.org/abs/1901.09764


6、Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis
作者：Qi Mao, Hsin-Ying Lee, Hung-Yu Tseng, Siwei Ma, Ming-Hsuan Yang
论文链接：https://arxiv.org/abs/1903.05628
代码链接：https://github.com/HelenMao/MSGAN （待更新）


7、MirrorGAN: Learning Text-to-image Generation by Redescription（图像文本生成）
作者：Tingting Qiao, Jing Zhang, Duanqing Xu, Dacheng Tao
论文链接：https://arxiv.org/abs/1903.05854


8、From Adversarial Training to Generative Adversarial Networks
作者：Xuanqing Liu, Cho-Jui Hsieh
论文链接：https://arxiv.org/pdf/1807.10454.pdf


9、OCGAN: One-class Novelty Detection Using GANs with Constrained Latent Representations
作者：Pramuditha Perera, Ramesh Nallapati, Bing Xiang
论文链接：https://arxiv.org/abs/1903.08550


10、SalGAN: Visual Saliency Prediction with Generative Adversarial Networks（商汤/华为/港中文）
作者：Junting Pan, Cristian Canton Ferrer, Kevin McGuinness, Noel E. O'Connor, Jordi Torres, Elisa Sayrol, Xavier Giro-i-Nieto
论文链接：https://arxiv.org/abs/1701.01081
代码链接：https://github.com/junting/seg2vid


11、StoryGAN: A Sequential Conditional GAN for Story Visualization（图像文本生成）
作者：Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, Lawrence Carin, David Carlson, Jianfeng Gao
论文链接：https://arxiv.org/abs/1812.02784
代码链接：https://github.com/yitong91/StoryGAN


12、Object-driven Text-to-Image Synthesis via Adversarial Training（图像文本生成）
作者：Wenbo Li, Pengchuan Zhang, Lei Zhang, Qiuyuan Huang, Xiaodong He, Siwei Lyu, Jianfeng Gao
论文链接：https://arxiv.org/abs/1902.10740


13、Text2Scene: Generating Compositional Scenes from Textual Descriptions
作者：Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, Lawrence Carin, David Carlson, Jianfeng Gao
论文链接：https://arxiv.org/abs/1809.01110
代码链接：https://github.com/uvavision/Text2Image


14、Image Generation from Layout
作者：Bo Zhao, Lili Meng, Weidong Yin, Leonid Sigal
论文链接：https://arxiv.org/abs/1811.11389


15、DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis
作者：Minfeng Zhu, Pingbo Pan, Wei Chen, Yi Yang
论文链接：https://arxiv.org/abs/1904.01310



16、Semantics Disentangling for Text-to-Image Generation（Oral)
作者：Guojun Yin, Bin Liu, Lu Sheng, Nenghai Yu, Xiaogang Wang, Jing Shao
论文链接：https://arxiv.org/abs/1904.01480


17、Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks（Oral)
作者：Yinpeng Dong, Tianyu Pang, Hang Su, Jun Zhu
论文链接：https://arxiv.org/abs/1904.02884


18、R2GAN: Cross-modal Recipe Retrieval with Generative Adversarial Network
作者：Bin Zhu, Chong-Wah Ngo, Jingjing Chen, and Yanbin Hao
论文链接：http://vireo.cs.cityu.edu.hk/papers/R2GAN.pdf


19、Multi-Channel Attention Selection GAN with Cascaded Semantic Guidance for Cross-View Image Translation（Oral)
作者：Hao Tang, Dan Xu, Nicu Sebe, Yanzhi Wang, Jason J. Corso, Yan Yan
论文链接：https://arxiv.org/abs/1904.06807
源码链接：https://github.com/Ha0Tang/SelectionGAN


20、Text Guided Person Image Synthesis
作者：Xingran Zhou, Siyu Huang, Bin Li, Yingming Li, Jiachen Li, Zhongfei Zhang
论文链接：https://arxiv.org/abs/1904.05118


21、Max-Sliced Wasserstein Distance and its use for GANs
作者：Ishan Deshpande, Yuan-Ting Hu, Ruoyu Sun, Ayis Pyrros, Nasir Siddiqui, Sanmi Koyejo, Zhizhen Zhao, David Forsyth, Alexander Schwing
论文链接：https://arxiv.org/abs/1904.05877


22、Fashion-AttGAN: Attribute-Aware Fashion Editing with Multi-Objective GAN
作者：Qing Ping, Jiangbo Yuan, Bing Wu, Wanying Ding
论文链接：https://arxiv.org/abs/1904.07460


23、Self-Supervised GANs via Auxiliary Rotation Loss
作者：Ting Chen, Xiaohua Zhai, Marvin Ritter, Mario Lucic, Neil Houlsby
论文链接：https://arxiv.org/abs/1811.11212

### 图像/视频处理、超分辨 18
1、Recurrent MVSNet for High-resolution Multi-view Stereo Depth Inference
作者：Yao Yao, Zixin Luo, Shiwei Li, Tianwei Shen, Tian Fang, Long Quan
论文链接：https://arxiv.org/abs/1902.10556
代码链接：https://github.com/YoYo000/MVSNet


2、Unprocessing Images for Learned Raw Denoising (Oral Presentation)
作者：Tim Brooks, Ben Mildenhall, Tianfan Xue, Jiawen Chen, Dillon Sharlet, Jonathan T. Barron
论文链接：https://arxiv.org/abs/1811.11127
project链接：http://timothybrooks.com/tech/unprocessing/
Reading note:We can learn a better denoising model by processing and unprocessing images the same way a camera does.


3、Image Super-Resolution by Neural Texture Transfer
作者：Zhifei Zhang, Zhaowen Wang, Zhe Lin, Hairong Qi
论文链接：https://arxiv.org/pdf/1903.00834.pdf
项目链接：http://web.eecs.utk.edu/~zzhang61/project_page/SRNTT/SRNTT.html
代码链接：https://github.com/ZZUTK/SRNTT


4、Toward Convolutional Blind Denoising of Real Photographs
作者：Shi Guo, Zifei Yan, Kai Zhang, Wangmeng Zuo, Lei Zhang
论文链接：https://arxiv.org/abs/1807.04686
代码链接：https://github.com/GuoShi28/CBDNet


5、Learning Parallax Attention for Stereo Image Super-Resolution(图像超分辨)
作者：Longguang Wang, Yingqian Wang, Zhengfa Liang, Zaiping Lin, Jungang Yang, Wei An, Yulan Guo
论文链接：https://arxiv.org/abs/1903.05784


6、Dual Residual Networks Leveraging the Potential of Paired Operations for Image Restoration
作者：Xing Liu, Masanori Suganuma, Zhun Sun, Takayuki Okatani
论文链接：https://arxiv.org/abs/1903.08817


7、PASSRnet: Parallax Attention Stereo Super-Resolution Network
作者：Longguang Wang, Yingqian Wang, Zhengfa Liang, Zaiping Lin, Jungang Yang, Wei An, Yulan Guo
论文链接：https://arxiv.org/abs/1903.05784
代码链接：https://github.com/LongguangWang/PASSRnet


8、Feedback Network for Image Super-Resolution
作者：Zhen Li, Jinglei Yang, Zheng Liu, Xiaomin Yang, Gwanggil Jeon, Wei Wu
论文链接：https://arxiv.org/abs/1903.09814


9、Meta-SR: A Magnification-Arbitrary Network for Super-Resolution （旷视，超分辨）
作者：Xuecai Hu, Haoyuan Mu, Xiangyu Zhang, Zilei Wang, Tieniu Tan, Jian Sun
论文链接：https://arxiv.org/abs/1903.00875
论文解读：CVPR2019 | 旷视提出Meta-SR：单一模型实现超分辨率任意缩放因子



10、Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels
作者：Kai Zhang, Wangmeng Zuo, Lei Zhang
论文链接：https://arxiv.org/abs/1903.12529
代码链接：https://github.com/cszn/DPSR


11、Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset
作者：Tianyu Wang, Xin Yang, Ke Xu, Shaozhe Chen, Qiang Zhang, Rynson Lau
论文链接：https://arxiv.org/abs/1904.01538
项目链接：https://stevewongv.github.io/derain-project.html



12、DVC: An End-to-end Deep Video Compression Framework（Oral）
作者：Guo Lu, Wanli Ouyang, Dong Xu, Xiaoyun Zhang, Chunlei Cai, Zhiyong Gao
论文链接：https://arxiv.org/abs/1812.00101
代码链接：https://github.com/GuoLusjtu/DVC


13、Blind Visual Motif Removal from a Single Image
作者：Amir Hertz, Sharon Fogel, Rana Hanocka, Raja Giryes, Daniel Cohen-Or
论文链接：https://arxiv.org/abs/1904.02756


14、Fast Spatio-Temporal Residual Network for Video Super-Resolution
作者：Sheng Li, Fengxiang He, Bo Du, Lefei Zhang, Yonghao Xu, Dacheng Tao
论文链接：https://arxiv.org/abs/1904.02870


15、3D Appearance Super-Resolution with Deep Learning
论文链接：https://github.com/ofsoundof/3D_Appearance_SR/blob/master/code/scripts/3d_appearance_sr.pdf
源码链接：https://github.com/ofsoundof/3D_Appearance_SR


16、Camera Lens Super-Resolution
作者：Chang Chen, Zhiwei Xiong, Xinmei Tian, Zheng-Jun Zha, Feng Wu
论文链接：http://staff.ustc.edu.cn/~zwxiong/cameraSR.pdf
源码链接：https://github.com/ngchc/CameraSR


17、Heavy Rain Image Restoration: Integrating Physics Model and Conditional Adversarial Learning
作者：Ruotent Li, Loong Fah Cheong, Robby T. Tan
论文链接：https://arxiv.org/abs/1904.05050


18、Learning Pyramid-Context Encoder Network for High-Quality Image Inpainting
作者：Yanhong Zeng, Jianlong Fu, Hongyang Chao, Baining Guo
论文链接：https://arxiv.org/abs/1904.07475

### 点云、三维重建 25
1、The Perfect Match: 3D Point Cloud Matching with Smoothed Densities
作者：Zan Gojcic, Caifa Zhou, Jan D. Wegner, Andreas Wieser
论文链接：https://arxiv.org/abs/1811.06879


2、Octree guided CNN with Spherical Kernels for 3D Point Clouds
作者：Huan Lei, Naveed Akhtar, Ajmal Mian
论文链接：https://arxiv.org/abs/1903.00343


3、DeepMapping: Unsupervised Map Estimation From Multiple Point Clouds
作者：Li Ding, Chen Feng
论文链接：https://arxiv.org/abs/1811.11397


4、Generating 3D Adversarial Point Clouds
作者：Chong Xiang (1), Charles R. Qi (2), Bo Li (3) ((1) Shanghai Jiao Tong Univerisity, (2) Stanford University, (3) University of Illinois at Urbana-Champaign)
论文链接：https://arxiv.org/abs/1809.07016
简要：Proposed several novel algorithms to craft adversarial point clouds against 3D deep learning models with adversarial points perturbation and adversarial points generation.


5、FlowNet3D: Learning Scene Flow in 3D Point Clouds
作者：Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
论文链接：https://arxiv.org/abs/1806.01411
简要：Proposed a novel deep neural network that learns scene flow from point clouds in an end-to-end fashion.


6、33.Single-Image Piece-wise Planar 3D Reconstruction via Associative Embedding（开源）
作者：Zehao Yu, Jia Zheng, Dongze Lian, Zihan Zhou, Shenghua Gao
论文链接：https://arxiv.org/abs/1902.09777
代码链接：https://github.com/svip-lab/PlanarReconstruction


7、FML: Face Model Learning from Videos(Oral)
作者：A. Tewari F. Bernard P. Garrido G. Bharaj M. Elgharib H-P. Seidel P. Perez M. Zollhöfer C.Theobalt
项目链接：http://gvv.mpi-inf.mpg.de/projects/FML19/
论文链接：http://gvv.mpi-inf.mpg.de/projects/FML19/paper.pdf


8、SceneCode: Monocular Dense Semantic Reconstruction using Learned Encoded Scene Representation
作者：Shuaifeng Zhi, Michael Bloesch, Stefan Leutenegger, Andrew J. Davison
论文链接：https://arxiv.org/abs/1903.06482


9、Photometric Mesh Optimization for Video-Aligned 3D Object Reconstruction
作者：Pelin Dogan, Leonid Sigal, Markus Gross
论文链接：
https://chenhsuanlin.bitbucket.io/photometric-mesh-optim/paper.pdf
代码链接：
https://github.com/chenhsuanlin/photometric-mesh-optim
项目链接：
https://chenhsuanlin.bitbucket.io/photometric-mesh-optim/


10、Learning View Priors for Single-view 3D Reconstruction
作者：Hiroharu Kato, Tatsuya Harada
论文链接：https://arxiv.org/abs/1811.10719
项目链接：http://hiroharu-kato.com/projects_en/view_prior_learning.html


11、Patch-based Progressive 3D Point Set Upsampling
作者：Wang Yifan, Shihao Wu, Hui Huang, Daniel Cohen-Or, Olga Sorkine-Hornung
论文链接：https://arxiv.org/abs/1811.11286
代码链接：https://github.com/yifita/3PU


12、GeoNet: Deep Geodesic Networks for Point Cloud Analysis（Oral,旷视，根据测地间隔的点云剖析深度网络）
作者：Tong He, Haibin Huang, Li Yi, Yuqian Zhou, Chihao Wu, Jue Wang, Stefano Soatto
论文链接：https://arxiv.org/abs/1901.00680
论文解读：CVPR 2019 | 旷视等Oral论文提出GeoNet：基于测地距离的点云分析深度网络


13、JSIS3D: Joint Semantic-Instance Segmentation of 3D Point Clouds with Multi-Task Pointwise Networks and Multi-Value Conditional Random Fields（Oral)
作者：Quang-Hieu Pham, Duc Thanh Nguyen, Binh-Son Hua, Gemma Roig, Sai-Kit Yeung
论文链接：https://arxiv.org/abs/1904.00699
项目链接：https://pqhieu.github.io/cvpr19.html


14、Point Cloud Oversegmentation with Graph-Structured Deep Metric Learning
作者：Loic Landrieu, Mohamed Boussaha
论文链接：https://arxiv.org/abs/1904.02113


15、Calibration of Asynchronous Camera Networks for Object Reconstruction Tasks
作者：Amy Tabb, Henry Medeiros
论文链接：https://arxiv.org/abs/1903.06811


16、StereoDRNet: Dilated Residual Stereo Net
作者：Rohan Chabra, Julian Straub, Chris Sweeny, Richard Newcombe, Henry Fuchs
论文链接：https://arxiv.org/abs/1904.02251


17、Conditional Single-view Shape Generation for Multi-view Stereo Reconstruction
作者：Yi Wei, Shaohui Liu, Wang Zhao, Jiwen Lu, Jie Zhou
论文链接：https://arxiv.org/abs/1904.06699


18、PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
作者：Shaoshuai Shi, Xiaogang Wang, Hongsheng Li
论文链接：https://arxiv.org/abs/1812.04244
源码链接：https://github.com/sshaoshuai/PointRCNN


19、Relation-Shape Convolutional Neural Network for Point Cloud Analysis
作者：Yongcheng Liu, Bin Fan, Shiming Xiang, Chunhong Pan
论文链接：https://arxiv.org/abs/1904.07601
项目链接：https://yochengliu.github.io/Relation-Shape-CNN/
源码链接：https://github.com/Yochengliu/Relation-Shape-CNN


20、A-CNN: Annularly Convolutional Neural Networks on Point Clouds
作者：Artem Komarichev, Zichun Zhong, Jing Hua
论文链接：https://arxiv.org/abs/1904.08017


21、PCAN: 3D Attention Map Learning Using Contextual Information for Point Cloud Based Retrieval
作者： Wenxiao Zhang, Chunxia Xiao
论文链接：https://arxiv.org/abs/1904.09793


22、Deep Convolutional Networks on 3D Point Clouds
作者：Satwik Acharyya, Zhengwu Zhang, Anirban Bhattacharya, Debdeep Pati
论文链接：https://arxiv.org/pdf/1811.07246.pdf
源码链接：https://github.com/DylanWusee/pointconv


23、LBS Autoencoder: Self-supervised Fitting of Articulated Meshes to Point Clouds
作者：Chun-Liang Li, Tomas Simon, Jason Saragih, Barnabás Póczos, Yaser Sheikh
论文链接：https://arxiv.org/abs/1904.10037


24、Modeling Local Geometric Structure of 3D Point Clouds using Geo-CNN
作者：Shiyi Lan, Ruichi Yu, Gang Yu, Larry S. Davis
论文链接：https://arxiv.org/abs/1811.07782


25、RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion
作者：Muhammad Sarmad, Hyunjoo Jenny Lee, Young Min Kim
论文链接：https://arxiv.org/abs/1904.12304
源码链接：https://github.com/iSarmad/RL-GAN-Net


26、Occupancy Networks - Learning 3D Reconstruction in Function Space
作者：Lars Mescheder and Michael Oechsle and Michael Niemeyer and Sebastian Nowozin and Andreas Geiger
论文链接：https://avg.is.tuebingen.mpg.de/uploads_file/attachment/attachment/490/top.pdf
源码链接：https://github.com/autonomousvision/occupancy_networks

### VQA、视觉语言导航 10
1、MUREL: Multimodal Relational Reasoning for Visual Question Answering
作者：Remi Cadene, Hedi Ben-younes, Matthieu Cord, Nicolas Thome
论文链接：https://arxiv.org/abs/1902.09487


2、Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation
作者：Xin Wang, Qiuyuan Huang, Asli Celikyilmaz, Jianfeng Gao, Dinghan Shen, Yuan-Fang Wang, William Yang Wang, Lei Zhang
论文链接：https://arxiv.org/abs/1811.10092
论文解读：https://mp.weixin.qq.com/s/LsHWkdwqqrOPFgCNNcBdpg


3、Image-Question-Answer Synergistic Network for Visual Dialog
作者：Dalu Guo, Chang Xu, Dacheng Tao
论文链接：https://arxiv.org/abs/1902.09774


4、Tactical Rewind: Self-Correction via Backtracking in Vision-and-Language Navigation(oral)
作者：Liyiming Ke, Xiujun Li, Yonatan Bisk, Ari Holtzman, Zhe Gan, Jingjing Liu, Jianfeng Gao, Yejin Choi, Siddhartha Srinivasa
论文链接：https://arxiv.org/abs/1903.02547
YouTube:https://youtu.be/ik9uz06Fcpk


5、Learning to Compose Dynamic Tree Structures for Visual Contexts（VQA,Oral）
作者：Kaihua Tang, Hanwang Zhang, Baoyuan Wu, Wenhan Luo, Wei Liu
论文链接：https://arxiv.org/abs/1812.01880
代码链接：
https://github.com/KaihuaTang/VCTree-Visual-Question-Answering


6、Transfer Learning via Unsupervised Task Discovery for Visual Question Answering（VQA)
作者：Hyeonwoo Noh, Taehoon Kim, Jonghwan Mun, Bohyung Han
论文链接：https://arxiv.org/abs/1810.02358


7、Information Maximizing Visual Question Generation(VQA)
作者：Zhongdao Wang, Liang Zheng, Yali Li, Shengjin Wang
论文链接：https://arxiv.org/abs/1903.11306


8、Answer Them All! Toward Universal Visual Question Answering Models（VQA)
作者：Robik Shrestha, Kushal Kafle, Christopher Kanan
论文链接：https://arxiv.org/abs/1903.00366


9、Cycle-Consistency for Robust Visual Question Answering（VQA)
作者：Gao Peng, Zhengkai Jiang, Haoxuan You, Zhengkai Jiang, Pan Lu, Steven Hoi, Xiaogang Wang, Hongsheng Li
论文链接：https://arxiv.org/pdf/1812.05252.pdf


10、Towards VQA Models that can Read
作者：Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, Marcus Rohrbach
论文链接：https://arxiv.org/abs/1904.08920

### OCR、文本检测 8
1、**Shape Robust Text Detection with Progressive Scale Expansion Network（文本检测）**
作者：Xiang Li, Wenhai Wang, Wenbo Hou, Ruo-Ze Liu, Tong Lu, Jian Yang
论文链接：https://arxiv.org/abs/1806.02559
代码链接：https://github.com/whai362/PSENet
网友复现：https://github.com/liuheng92/tensorflow_PSENet



2、Towards Robust Curve Text Detection with Conditional Spatial Expansion
作者：Zichuan Liu, Guosheng Lin, Sheng Yang, Fayao Liu, Weisi Lin, Wang Ling Goh
论文链接：https://arxiv.org/abs/1903.08836


3、Shape Robust Text Detection with Progressive Scale Expansion Network
作者：Wenhai Wang, Enze Xie, Xiang Li, Wenbo Hou, Tong Lu, Gang Yu, Shuai Shao
论文链接：https://arxiv.org/abs/1903.12473


4、Handwriting Recognition in Low-resource Scripts using Adversarial Learning
作者：Ayan Kumar Bhunia, Abhirup Das, Ankan Kumar Bhunia, Perla Sai Raj Kishore, Partha Pratim Roy
论文链接：https://arxiv.org/pdf/1811.01396.pdf


5、Handwriting Recognition in Low-resource Scripts using Adversarial Learning
作者：Ayan Kumar Bhunia, Abhirup Das, Ankan Kumar Bhunia, Perla Sai Raj Kishore, Partha Pratim Roy
论文链接：https://arxiv.org/abs/1811.01396


6、Tightness-aware Evaluation Protocol for Scene Text Detection
作者：Yuliang Liu, Lianwen Jin, Zecheng Xie, Canjie Luo, Shuaitao Zhang, Lele Xie
论文链接：https://arxiv.org/abs/1904.00813


7、Character Region Awareness for Text Detection（文本检测）
作者：Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee
论文链接：https://arxiv.org/abs/1904.01941


8、Look More Than Once: An Accurate Detector for Text of Arbitrary Shapes
作者：Chengquan Zhang, Borong Liang, Zuming Huang, Mengyi En, Junyu Han, Errui Ding, Xinghao Ding
论文链接：https://arxiv.org/abs/1904.06535

### 自动驾驶、SLAM 12
1、Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving（自动驾驶）
作者：Yan Wang, Wei-Lun Chao, Divyansh Garg, Bharath Hariharan, Mark Campbell, Kilian Q. Weinberger
论文链接：https://arxiv.org/abs/1812.07179
项目链接：https://mileyan.github.io/pseudo_lidar/
代码链接：https://github.com/mileyan/pseudo_lidar


2、ApolloCar3D: A Large 3D Car Instance Understanding Benchmark for Autonomous Driving
作者：Xibin Song, Peng Wang, Dingfu Zhou, Rui Zhu, Chenye Guan, Yuchao Dai, Hao Su, Hongdong Li, Ruigang Yang
论文链接：https://arxiv.org/abs/1811.12222
简要：The first large-scale database suitable for 3D car instance understanding, ApolloCar3D, collected by Baidu. The dataset contains 5,277 driving images and over 60K car instances, where each car is fitted with an industry-grade 3D CAD model with absolute model size and semantically labelled keypoints.


3、Group-wise Correlation Stereo Network
作者：Xiaoyang Guo, Kai Yang, Wukui Yang, Xiaogang Wang, Hongsheng Li
论文链接：https://arxiv.org/abs/1903.04025


4、Stereo R-CNN based 3D Object Detection for Autonomous Driving
作者：Peiliang Li, Xiaozhi Chen, Shaojie Shen
论文链接：https://arxiv.org/abs/1902.09738


5、Deep Rigid Instance Scene Flow
作者：Wei-Chiu Ma 、Shenlong Wang 、Rui Hu、Yuwen Xiong、 Raquel Urtasun
论文链接：
https://people.csail.mit.edu/weichium/papers/cvpr19-dsisf/paper.pdf
论文摘要：在本文中，我们解决了自动驾驶环境下的场景流量估计问题。 我们利用深度学习技术以及强大的先验，因为在我们的应用领域中，场景的运动可以由机器人的运动和场景中的演员的3D运动来组成。


6、An Efficient Schmidt-EKF for 3D Visual-Inertial SLAM
作者：Patrick Geneva, James Maley, Guoquan Huang
论文链接：https://arxiv.org/abs/1903.08636


7、LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving
作者：Gregory P. Meyer, Ankit Laddha, Eric Kee, Carlos Vallespi-Gonzalez, Carl K. Wellington
论文链接：https://arxiv.org/abs/1903.08701


8、.GS3D: An Efficient 3D Object Detection Framework for Autonomous Driving
作者：Buyu Li, Wanli Ouyang, Lu Sheng, Xingyu Zeng, Xiaogang Wang
论文链接：https://arxiv.org/abs/1903.10955


9、Learning to Adapt for Stereo
作者：Alessio Tonioni, Oscar Rahnama, Thomas Joy, Luigi Di Stefano, Thalaiyasingam Ajanthan, Philip H. S. Torr
论文链接：https://arxiv.org/abs/1904.02957
代码链接：https://github.com/CVLAB-Unibo/Learning2AdaptForStereo


10、What Object Should I Use? - Task Driven Object Detection
作者：Johann Sawatzky, Yaser Souri, Christian Grund, Juergen Gall
论文链接：https://arxiv.org/abs/1904.03000


11、YUVMultiNet: Real-time YUV multi-task CNN for autonomous driving
作者：Thomas Boulay, Said El-Hachimi, Mani Kumar Surisetti, Pullarao Maddu, Saranya Kandan
论文链接：https://arxiv.org/abs/1904.05673


12、L3-Net: Towards Learning based LiDAR Localization for Autonomous Driving
作者：Weixin Lu， Yao Zhou， Guowei Wan，Shenhua Hou，Shiyu Song
论文链接：https://songshiyu01.github.io/pdf/L3Net_W.Lu_Y.Zhou_S.Song_CVPR2019.pdf

### 人群计数 3
1、Learning from Synthetic Data for Crowd Counting in the Wild
作者：Qi Wang, Junyu Gao, Wei Lin, Yuan Yuan
论文链接：https://arxiv.org/abs/1903.03303


2、Point in, Box out: Beyond Counting Persons in Crowds
作者：待更新
论文链接：https://github.com/xiaofanglegoc/xiaofanglegoc.github.io/blob/master/publications/cvpr2019.pdf


3、Learning the Depths of Moving People by Watching Frozen People（Oral）
作者：Zhengqi Li, Tali Dekel, Forrester Cole, Richard Tucker, Noah Snavely, Ce Liu, William T. Freeman
论文链接：https://arxiv.org/abs/1904.11111

### 数据集 6
--
Relevance Score: 6

Reason:

1、COIN: A Large-scale Dataset for Comprehensive Instructional Video Analysis
作者：Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, Jie Zhou
论文链接：https://arxiv.org/abs/1903.02874
项目链接：https://coin-dataset.github.io/
代码链接：https://github.com/coin-dataset/code

--
Relevance Score: 6

Reason:

2、RAVEN: A Dataset for Relational and Analogical Visual rEasoNing
作者：Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, Jie Zhou
论文链接：https://arxiv.org/abs/1903.02741
项目链接：https://wellyzhang.github.io/project/raven.html

--
Relevance Score: 7

Reason: 垂直领域：安防

3、SIXray : A Large-scale Security Inspection X-ray Benchmark for Prohibited Item Discovery in Overlapping Images（金山云大规模X光违禁品安检数据集）
作者：Caijing Miao, Lingxi Xie, Fang Wan, Chi Su, Hongye Liu, Jianbin Jiao, Qixiang Ye
论文链接：https://arxiv.org/abs/1901.00303
论文简要：本文针对X光安检数据集，提出了类别平衡的分层细化模型处置数据集存在的成绩。

--
Relevance Score: 6

Reason:

4、A Cross-Season Correspondence Dataset for Robust Semantic Segmentation
作者：Måns Larsson, Erik Stenborg, Lars Hammarstrand, Torsten Sattler, Mark Pollefeys, Fredrik Kahl
论文链接：https://arxiv.org/abs/1903.06916

--
Relevance Score: 6

Reason:

5、A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection
作者：Reza Ghoddoosian, Marnim Galib, Vassilis Athitsos
论文链接：https://arxiv.org/abs/1904.07312

### 行人重识别、行人检测 8
--
Relevance Score: 4

Reason: NOT directly related

1、Dissecting Person Re-identification from the Viewpoint of Viewpoint
作者：Xiaoxiao Sun, Liang Zheng
论文链接：https://arxiv.org/abs/1812.02162
源码链接：https://github.com/sxzrt/Dissecting-Person-Re-ID-from-the-Viewpoint-of-Viewpoint

--
Relevance Score: 4

Reason: NOT directly related

2、Unsupervised Person Re-identification by Soft Multilabel Learning(行人再识别，Oral)
作者：Hong-Xing Yu, Wei-Shi Zheng, Ancong Wu, Xiaowei Guo, Shaogang Gong, Jian-Huang Lai
论文链接：https://arxiv.org/abs/1903.06325
源码链接：https://github.com/KovenYu/MAR

--
Relevance Score: 4

Reason: NOT directly related

3、Perceive Where to Focus: Learning Visibility-aware Part-level Features for Partial Person Re-identification
作者：Yifan Sun, Qin Xu, Yali Li, Chi Zhang, Yikang Li, Shengjin Wang, Jian Sun
论文链接：https://arxiv.org/abs/1904.00537

--
Relevance Score: 4

Reason: NOT directly related

4、Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification
作者：Zhun Zhong, Liang Zheng, Zhiming Luo, Shaozi Li, Yi Yang
论文链接：https://arxiv.org/abs/1904.01990
代码链接：https://github.com/zhunzhong07/ECN

--
Relevance Score: 4

Reason: NOT directly related

5、SSA-CNN: Semantic Self-Attention CNN for Pedestrian Detection
作者：Chengju Zhou,Meiqing Wu,Siew-Kei Lam
论文链接：https://arxiv.org/abs/1902.09080v1
论文摘要：本文将语义分割结果作为自我关注线索进行探索，以显着提高行人检测性能。

--
Relevance Score: 4

Reason: NOT directly related

6、High-level Semantic Feature Detection:A New Perspective for Pedestrian Detection
作者：Wei Liu, Shengcai Liao, Weiqiang Ren, Weidong Hu, Yinan Yu
论文链接：https://arxiv.org/abs/1904.02948

--
Relevance Score: 4

Reason: NOT directly related

7、High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection
作者：Zhao-Min Chen, Xiu-Shen Wei Peng Wang3Yanwen Guo1
论文链接：https://github.com/liuwei16/CSP/blob/master/docs/2019CVPR-CSP.pdf
源码链接：https://github.com/liuwei16/CSP

--
Relevance Score: 4

Reason: NOT directly related

8、Pedestrian Detection in Thermal Images using Saliency Maps
作者：Debasmita Ghose, Shasvat Mukeshkumar Desai, Sneha Bhattacharya, Deep Chakraborty, Madalina Fiterau, Tauhidur Rahman
论文链接：https://arxiv.org/abs/1904.06859

### 其他 254
(ignore from now)
