### Diffusion Model 扩散模型

#### Denosing Diffusion Probabilistic Models DDPM

如何运作的？

#### reverse process

生成一个杂讯的图片

![1730891009414](image/DiffusionModel/1730891009414.png)

本来图片就在杂讯里，只是滤掉不需要的部分

![1730891245450](image/DiffusionModel/1730891245450.png)

#### forword process

![1730891363584](image/DiffusionModel/1730891363584.png)

由一张训练集里的图像，添加噪声

算法

![1730891644257](image/DiffusionModel/1730891644257.png)

### Stable Diffusion

框架

![1730891802977](image/DiffusionModel/1730891802977.png)

FID 用来评估

CLIP

![1730892320823](image/DiffusionModel/1730892320823.png)

Decoder

中间产物：
- 小图
- Latent Representation

![1730892478221](image/DiffusionModel/1730892478221.png)

训练一个auto-encoder，将 decoder 拿出来就能用了

generation model

![1730892963215](image/DiffusionModel/1730892963215.png)


#### VAE v.s. Diffusion

![1730946103509](image/DiffusionModel/1730946103509.png)


#### 算法详解

训练

![1730946509208](image/DiffusionModel/1730946509208.png)

![1730946730210](image/DiffusionModel/1730946730210.png)

产生图

![1730946904633](image/DiffusionModel/1730946904633.png)

#### 影像生成模型本质上的共同目标

产生一个 distribution 和真正的有越接近越好

如何衡量“越接近越好”？极大似然估计

![1730948022460](image/DiffusionModel/1730948022460.png)

maximum likelihood = minimize KL divergence

#### VAE

产生 x 的概率：每个 z 产生的概率和每个 z 中产生 x 的概率

![1730948582569](image/DiffusionModel/1730948582569.png)

![1730948766312](image/DiffusionModel/1730948766312.png)

maximize 以得到尽可能大的 logP(x)（lower bound 下界）

#### DDPM

![1730949068198](image/DiffusionModel/1730949068198.png)

![1730949188473](image/DiffusionModel/1730949188473.png)

![1730966731038](image/DiffusionModel/1730966731038.png)

本来应该 sample n 次，但是可以只 sample 一次来代替

![1730966832926](image/DiffusionModel/1730966832926.png)

知道 x0，xt 求 xt-1

![1730967375094](image/DiffusionModel/1730967375094.png)

![1730967434814](image/DiffusionModel/1730967434814.png)

![1730967595167](image/DiffusionModel/1730967595167.png)

![1730967756722](image/DiffusionModel/1730967756722.png)

![1730967881720](image/DiffusionModel/1730967881720.png)

概率最大的不一定时最好的结果

所以用 sample 有随机性

![1730968304190](image/DiffusionModel/1730968304190.png)


