### STAN: Spatio-Temporal Attention Network for Next Location Recommendation
#### Innovation
早期的模型主要是研究序列转换规律，还有一些研究是利用RNN的变体来提取用户长期与短期序列特征。近来，有人利用两个**连续**POI之间的**时间区间**和**空间距离**来挖掘用户移动的**时空相关性**，他们把时间划分成离散的24个小时，把空间划分成多个离散的网格区域。另外，他们还修改了网络结构或者增加额外的模块来融合这些**时空信息**。

但这些方法还是没有解决以下问题：

* 没有考虑**不相邻**和**不连续**的POI之间的相关性，不相邻指的两个POI间的空间距离不相邻，不连续是指两个POI在check-in序列上不是连续的
* 用**网格**将空间划分个多个区域，不能准确反应**空间距离**，彼此相邻的网格与不相邻的网格之间没有明显的差异，因此不能反应出用户的空间偏好
* 忽略了**personalized item frequency(PIF)**，一个用户重复地访问某个POI，那么这个POI就很重要，用户可能会再次访问。但之前的基于**RNN**或者**self-attention**的模型由于内存机制和归一化操作很难反映PIF

因此作者提出了双层self-attention模块，一个self-attention layer负责聚合用户check-in序列中那些重要的POI，另一个self-attention layer负责匹配用户接下来最有可能访问的候选POI，两个layer都考虑了POI之间的时空相关性，双层self-attention模块的设计能很好的反映PIF信息。作者用**线性插值**代替**网格**来划分离散空间，能很好反映用户的空间偏好。
#### Definition
把用户**Trajectory内**访问过的两个POI间的时间区间和空间距离显示地构造成两个时空相关矩阵，时间区间是第i个POI的时间与第j个POI的时间差的绝对值。把**候选**POI和用户**Trajectory内**的POI之间的时间区间和空间距离也构造两个时空相关矩阵，时间区间是第i个候选POI和Trajectory内第j个POI的时间差的绝对值。空间距离就是两个POI之间的距离。

* Trajectory Spatio-Temporal Relation Matrix

* Candidate Spatio-Temporal Relation Matrix


#### Methodology
##### Multimodal Embedding Module
学习用户，空间，时间，时空影响的稠密表征。

* User Trajectory Embedding layer：把用户，空间，时间编码成向量，把**标量**转换成**稠密向量**来减少计算和改善表征。其中，把连续的时间离散化为7x24个区间，代表该时间属于一周内具体的某天某时。将这**3个向量之和**作为该embedding层的输出，即**用户轨迹表征**。

* Spatio-Temporal Embedding layer：把时空信息编码成稠密的向量。用最大的空间或者时间区间来作为embedding的数量很可能导致**稀疏相关性**编码，因此作者考虑用能反映时空上下文信息的**单元embedding向量**乘以空间和时间区间来避免稀疏编码。作者也提供了另一种效果近似的方法：**线性插值embedding layer**，通过设置一个**上界单元embedding向量**和一个**下界单元embedding向量**，把区间明确地表示为一个线性插值。


##### Self-Attention Aggregation Layer
给每个POI赋予不同的权重，那些相关的POI赋予更多的权重，使得能聚集相关的POI，并更新每个POI的表征。给定**用户轨迹表征**，**轨迹时空相关矩阵**，用**自定义的self-attention**公式计算得到一个新的序列，作为新的**用户轨迹表征**。

##### Attention Matching Layer
用新的**用户轨迹表征**来从候选POI中选择最合适的POI。给定用户轨迹表征，**候选POI表征**，**候选时空相关矩阵表征**，用**自定义self-attention**公式计算每个候选POI的**概率**，该公式反映了**用户轨迹表征**都参与了每个候选POI的匹配过程，也就是说如果用户轨迹里有重复的POI，那么这些重复POI的表征都会对候选POI有影响，即考虑了**PIF**信息。

##### Balanced Sample
候选POI里只有一个是**positive**的，造成正负样本数目不平衡，使得优化cross-enrtopy没有效果。因此作者在每次训练时从负样本中随机抽取几个负样本，在每次训练完后都更新随机种子。

#### Question
* 学习时空embedding向量那部分，**单元embedding向量**或者**线性插值方法**避免稀疏编码，为什么这么做有用？
* 两个自定义的self-attention的计算公式都不懂？
