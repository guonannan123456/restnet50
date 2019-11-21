https://www.cnblogs.com/blog4ljy/p/10482474.html

1.PIL中的Image和numpy中的数组array相互转换
+1. PIL image转换成array

img = np.asarray(image)
或
img=np.array(image)
需要注意的是，如果出现read-only错误，并不是转换的错误，一般是你读取的图片的时候，默认选择的是"r","rb"模式有关。

修正的办法:　手动修改图片的读取状态
img.flags.writeable = True # 将数组改为读写模式

或者

+ im = Image.open("lena.jpg")

+ # 显示图片
+ im.show() 

+ im = im.convert("L") 
+ data = im.getdata()
+ data = np.matrix(data)

或者

im = np.array(pil_im)
2. array转换成image

方法1

from PIL import Image
Image.fromarray(np.uint8(img))

注意img如果是uint16的矩阵而不转为uint8的话，Image.fromarray这句会报错

File "/usr/local/lib/python2.7/site-packages/PIL/Image.py", line 1884, in fromarray
raise TypeError("Cannot handle this data type")
TypeError: Cannot handle this data type

类似这个问题

https://stackoverflow.com/questions/26666269/pil-weird-error-after-resizing-image-in-skimage

方法2

import cv2

cv2.imwrite("output.png", out)

out可以是uint16类型数据

16位深度图像转8位灰度

matlab

img=imread('output.png')

img1=im2uint8(img)

imwrite(img1,'result.jpg')

或者python

from PIL import Image

import numpy as np

import math

img=Image.fromarray(np.uint8(img_array/float(math.pow(2,16)-1)*255))

img.save('22.png')

