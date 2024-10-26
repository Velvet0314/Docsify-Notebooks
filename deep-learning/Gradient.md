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