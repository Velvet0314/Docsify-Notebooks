### self attention

#### sophisticated input 复杂的输入

![1730112402534](image/self-attention/1730112402534.png)

e.g. 文字处理

one-hot encoding or word embedding

e.g. 语音处理

![1730112946384](image/self-attention/1730112946384.png)

e.g. 一个图

![1730112980142](image/self-attention/1730112980142.png)

e.g. 一个分子

![1730113036997](image/self-attention/1730113036997.png)

#### 输出是什么呢？

- each vector has a label（输入和输出数目一样）
- the whole sequence has a label
- model decides the number of labels itself（seq2seq）

![1730113677817](image/self-attention/1730113677817.png)

#### 例子：sequence labeling 词性标注

**自注意力机制**

- step1：find the relevant vectors in a sequence

![1730114162553](image/self-attention/1730114162553.png)

如何得到相关度 α 呢？

一个计算 attention 的模组

常见的计算方式：

![1730114248148](image/self-attention/1730114248148.png)

dot-product: 分别乘上两个不同的矩阵再做点积（element-wise）

![1730114477492](image/self-attention/1730114477492.png)

![1730114514627](image/self-attention/1730114514627.png)

对 attention score 做 softmax（也可以不用softmax）做一个 normalization

![1730114608879](image/self-attention/1730114608879.png)

q —— Query k —— Key

可以同时计算的（parallel）

#### 矩阵乘法简化

![1730115291579](image/self-attention/1730115291579.png)

![1730115472022](image/self-attention/1730115472022.png)

![1730115554522](image/self-attention/1730115554522.png)

#### 总结

![1730115648197](image/self-attention/1730115648197.png)

需要学习的参数只有三个

#### Multi-head Self-attention

![1730116397864](image/self-attention/1730116397864.png)

#### Positional Encoding

为每个位置设定一个 vector

![1730116695428](image/self-attention/1730116695428.png)


### 应用

truncated self-attention => 语音辨识

考虑一小个的范围

![1730117358040](image/self-attention/1730117358040.png)

**self-attention 的适用范围：输入是vector set**

![1730117541098](image/self-attention/1730117541098.png)


#### self-attention v.s. CNN

![1730117683414](image/self-attention/1730117683414.png)

读一下

![1730117697220](image/self-attention/1730117697220.png)

CNN 是 self-attention 的特例

#### self-attention v.s. RNN

![1730118214121](image/self-attention/1730118214121.png)

#### self-attention for graph

consider edge 考虑边

![1730118358792](image/self-attention/1730118358792.png)







