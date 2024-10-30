### Transformer

#### 什么是 sequence to sequence (seq2seq)

input a sequence, output a sequence

the output length is determined by model itself

e.g. 语音辨识、机器翻译、chatbot、QA

![1730275724531](image/Transformer/1730275724531.png)

哪些任务可以使用 seq2seq？

- 语法解析
- Multi-label Classification

不是多分类（Multi-class）

An object can belong to multiple classes

![1730276576987](image/Transformer/1730276576987.png)

#### seq2seq 解析 —— transformer

Encoder 和 Decoder

#### Encoder

![1730277001187](image/Transformer/1730277001187.png)

residual layer：残差层

![1730277091744](image/Transformer/1730277091744.png)

#### Decoder

- Autoregressive（AT）
- Non-Auto regressive（NAT）

Masked self-attention：只考虑前面的输出以及自己

如何让 decoder 停止？END 表示

![1730278773620](image/Transformer/1730278773620.png)

#### AT v.s. NAT

![1730279167118](image/Transformer/1730279167118.png)

#### Encoder 和 Decoder 的传递

cross attention

![1730279365114](image/Transformer/1730279365114.png)

#### training

类似于分类问题

![1730282977580](image/Transformer/1730282977580.png)

teacher forcing

![1730282943944](image/Transformer/1730282943944.png)


tips
- Copy Mechanism：某些词汇可能是重复的（name，摘要）
- Guided Attention：在某些任务中，Attention 需要某种固定的形式
- Beam Search

![1730283780721](image/Transformer/1730283780721.png)

有些任务需要一些随机性（noise）

#### exposure bias

![1730284209665](image/Transformer/1730284209665.png)

测试中没有正确的答案（但训练时我们给decoder看了正确的答案）

![1730284251212](image/Transformer/1730284251212.png)

加入一些错误的样本























