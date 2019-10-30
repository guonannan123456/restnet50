
### 1.python--sum函数--sum(axis=1)

平时用的sum应该是默认的axis=0 就是普通的相加，当加入axis=1以后就是将一个矩阵的每一行向量相加。

例如：
```
c = np.array([[0, 2, 1], [3, 5, 6], [0, 1, 1]])
print c.sum()
print c.sum(axis=0)
print c.sum(axis=1)
结果分别是：19, [3 8 8], [ 3 14  2]
axis=0, 表示列。
axis=1, 表示行。
```
### 2.Python - 判断list是否为空
```
Python中判断list是否为空有以下两种方式：
方式一：

list_temp = []
if len(list_temp):
   # 存在值即为真
else:
   # list_temp是空的

方式二：

if list_temp:
   # 存在值即为真
else:
   # list_temp是空的
```
