### 类神经网络训练不起来？（一）

gradient is close to zero —— critical point

#### local minima 局部最小值

no way to go

#### saddle point 鞍点（不是局部最小值）

can escape

如何判断微分为 0 的时候是哪种情况呢？

通过泰勒展开来近似

![1729928985406](image/Gradient/1729928985406.png)

H 是海森矩阵

![1729929302323](image/Gradient/1729929302323.png)

通过线性代数来判断：

如果 H 是正定的——其所有特征值都是正的

反之，H 是负定的——其所有特征值都是负的

如果 H 特征值有正有负，那么就是鞍点

H 可以给出更新的方向

![1729929990816](image/Gradient/1729929990816.png)

u 是特征向量

那到底是局部最小值这种情况多呢，还是鞍点多呢？

假说：

![1729930400105](image/Gradient/1729930400105.png)

### 类神经网络训练不起来？（二）

#### 回顾:batch 批次

![1729943686809](image/Gradient/1729943686809.png)

shuffle 打乱，每一个 epoch 重新分 batch

#### small batch v.s. large batch

![1729944018582](image/Gradient/1729944018582.png)

![1729944152962](image/Gradient/1729944152962.png)

平行运算

noisy 的 batch 反而会提高训练效果

![1729944298112](image/Gradient/1729944298112.png)

为什么？

每个 batch 的 loss function 略有差异

![1729944378786](image/Gradient/1729944378786.png)

局部最小值的好坏

![1729944698711](image/Gradient/1729944698711.png)

small batch 具有一定的随机性

#### 总结

![1729944787557](image/Gradient/1729944787557.png)

#### momentum 动量

类比物理现象：

![1729944925420](image/Gradient/1729944925420.png)

![1729944931622](image/Gradient/1729944931622.png)

##### 一般的 gradient descent

![1729944982011](image/Gradient/1729944982011.png)

##### gradient descent + momentum

update不仅仅只取决于gradient的方向，还考虑了之前移动的方向

![1729945113613](image/Gradient/1729945113613.png)

形象化解释：

![1729945282075](image/Gradient/1729945282075.png)

### 类神经网络训练不起来？（三）

#### Adaptive Learning Rate

![1730005352020](image/Gradient/1730005352020.png)

迈出的步伐太大了，导致结果在震荡

![1730005800415](image/Gradient/1730005800415.png)

σ 既取决于参数，又取决于更新的次数

#### Root Mean Square 均方根

![1730006029644](image/Gradient/1730006029644.png)

![1730006114722](image/Gradient/1730006114722.png)

#### RMS Prop

可以自己决定每个参数的权重

![1730006306798](image/Gradient/1730006306798.png)

![1730006407892](image/Gradient/1730006407892.png)

#### Adam: RMSProp + momentum

![1730006566541](image/Gradient/1730006566541.png)

细节见论文

#### Learning Rate Scheduling

##### 学习率衰减 lr decay

![1730006895938](image/Gradient/1730006895938.png)

##### warm up

lr 先增加，再减小

![1730007167758](image/Gradient/1730007167758.png)

#### 总结

![1730007268220](image/Gradient/1730007268220.png)

### 类神经网络训练不起来？（五）

#### batch normalization

![1730016979029](image/Gradient/1730016979029.png)

需要尽可能相同的数值范围

#### feature normalization

![1730017110877](image/Gradient/1730017110877.png)

批归一化

batch 需要一定的大小，才能得出分布

![1730017667313](image/Gradient/1730017667313.png)

还原可能的一些情况

测试上采用的方法

![1730017822801](image/Gradient/1730017822801.png)

