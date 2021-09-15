### CANS-Net: Context-Aware Non-Successive Modeling Network for Next Point-of-Interest Recommendation

#### Innovation
大多数的工作是设计序列模型，通过研究**连续的**POI(check-in序列)来捕获POI间的转换规律。但是如果用户的check-in数据很多(check-in序列很长)，那么就很难将所有POI编码成固定大小的向量；另外，next poi可能只与一部分历史check-ins数据高度相关，因此**连续**的check-ins会弱化这些有关部分的影响。也有少量的工作是通过挖掘用户的移动周期性或者POI的地理因素，研究**不连续**的POI间的转换规律。然而，他们都忽略了其他时空因素的影响。

用户当前check-in的**上下文信息**有助于建模历史check-in数据。比如，用户更可能去距离更近的POI；用户在不同时间有不同的偏好，在相似时间有相似的偏好。这样，**时空上下文信息**就能被用来从历史check-ins中找到**高度相关**的check-in，从而捕获不同的影响因素。给定用户当前时刻的时空上下文，就能使用用户近来访问的**不连续**的POI来构造不同类型的check-in序列用于模型训练。

因此作者提出两个模块来分别研究用户长期兴趣和短期兴趣。**长期兴趣模块**把用户的历史check-in数据按一周来划分，分别捕获每天的兴趣，再利用注意力机制融合成长期兴趣(时间效应or移动周期性)。**短期兴趣模块**根据用户最近的check-in数据来构造4个序列来分别捕获顺序、空间、时间以及时空影响。通过注意力机制将这5个兴趣序列融合成一个整体兴趣，再和一个候选POI一起送入prediction layer得到该POI的预测概率。
#### Methodology
##### Embedding Layer
因为要描述**时空上下文**，那么用户当前的check-in就要包括时间信息和空间信息。每个POI有经纬度信息和类型信息，利用**geohash-5**技术将每个POI分配到一个区域**g**。将时间维度按一周来划分，另外将一天划分为24个时间槽，那么时间信息就用该POI在星期几**w**的哪个时间槽**m**来表示。
这样，一个POI就用一个5元组(p,c,w,m,g)来表示，其中每个元素都是**one-hot**向量(一个POI是5个向量的拼接)，经过embedding layer后变成一个稠密向量**(p,c,w,m,g)**，而**时空上下文**就是**(w,m,g)**。
##### Long-Term Module
用户**daily interest**可以反映用户每天的移动模式。用户的所有历史check-in数据划是**不连续**。先构造7个掩码序列，长度和用户历史check-in个数(假设**t-1**个)相同，假如用户第一个check-in是在星期天，那么第一个掩码序列的第1个位置为**1**，其他6个掩码序列的第1个位置为**0**。用构造好的7个掩码序列分别和**t-1**个用户历史check-in数据对应元素相乘得到7个**daily embedding**。因为**daily embedding**有N个向量，所以要做一个**average pooling**：计算这**t-1**个向量的累积和以及对应的掩码序列的累积和，用前者除以后者得到均值处理后的结果。再把该结果送入一个**FC**层得到用户**daily interest**。

用一个**候选**POI(**第t个check-in**)和已知**时空上下文**构造一个查询**q=(p,c,w,m,g)**，候选POI有很多，所以可以构造很多个查询，但只有一个查询是**positive**。然后使用**Bahdanau**注意力机制融合**daily interset**得到**long-term interest**。
##### Short-Term Module
给定当前**时空上下文**(第t个)，即**(w,m,g)**。从用户历史check-in中(前t-1个)按照顺序、空间，时间、时空条件选择**最近的**某些check-in构造四个short-term序列。具体构造如下：

* 顺序序列：选择**连续**的check-in，即选择位于t-3，t-2，t-1的那些POI
* 空间序列：选择具有相同**g**的那些POI
* 时间序列：选择那些位于与**m**很接近的时间槽内的POI
* 时空序列：选择那些具有相同**g**，同时位于与**m**都很接近的时间槽内的POI

将4个**short-term**序列送入LSTM中得到4个**short-term interest**。同样的，使用**Bahdanau**注意力机制分别计算这4个对应的权重，为了让重要因素更能凸显出来，不使用softmax函数处理这4个权重。又因为这4个**short-term interest**是根据不同的条件来构造的，所以选择拼接操作而不是计算权重和。

##### Prediction Layer
因为已经获得了**long-term** interest和4个**short-term** interest，用**Bahdanau**注意力机制计算这5个interest的权重，然后将它们拼接起来。将拼接后的结果再和查询**q**拼接作为MLP的输入，输出是**q**中所包含的**候选**POI的概率。只有其中一个候选POI是**positive**的，其他都是**negative**的，使用cross-entropy来计算所有候选POI的loss和。
#### Question
* 实验是取一个用户最近500次check-in，用最后一个POI作为真值label，那么前499个check-ins用来训练模型，**时空上下文**就指用户当前check-in的信息**(w,m,g)**，是指第499个check-in的信息吧？
* 根据**时空上下文**来构造4个**short-term** interest，其中**顺序** interest是选择**连续的**check-in数据，这个**连续的**指的是第496，第497，第498个check-in，实际上这3个check-in还是不连续的，即不是496->497->498->499这种传统意义的连续序列，只不过是作者把这500个数据这样安排的吗？如果这样的话，这500个check-in就不是这个用户实际访问POI的顺序了啊，在实际场景中有应用价值吗？
* 实验设置**negetive**候选POI数量为20，那么在训练过程中，构造的查询**q**就有21种了，包括一个**positive** POI。这20个**negative**候选POI如何选择文中没具体说？
* Bahdanau Attention不懂


#### Preliminary
**Geohash**

Geo-hash是将地球理解为一个(经度，纬度)的二维平面，将平面划分成多个区域，在一定经纬度范围内的POI属于同一个区域
