### **Discovering Collaborative Signals for Next POI Recommendation with Iterative Seq2Graph Augmentation**

### Innovation
之前的研究认为每个POI序列是独立的，所学得的POI embedding只与该序列中的其他POI有关，忽略了来自其他序列的POI的影响，即**协同信号**。另外，**稀疏**的POI-POI转换也会限制模型学得有效的序列模式，从而限制学得高质量的POI embedding。

因此，作者提出了**Seq2Graph**，将graph这种数据结构引入到sequence中。该序列中的POI依然是按时间顺序表示的，且每个POI的“neighbour”结点以及“边”是从其他序列观察到的。这样，协同信号就能被用来学得高质量的POI embedding。**Figure 1**展示了这个过程。

为了进一步减缓**数据稀疏性**所造成的影响，作者提出去学习**POI类型**之间的转换，因为**类型**相对POI来说更少，类型间的转换就显得更**稠密**。
![Figure 1](/Users/raoxuan/Desktop/papers/Next poi recommendation/summary/Discovering Collaborative Signals for Next POI Recommendation with Iterative Seq2Graph Augmentation/FIgure1.png)

###  Methodology
#### Seq2Graph Augmentation
随机选择某个序列s来转换为图G，G中的结点除了s中的POI外，还包括每个POI的**first-order**(一阶)邻居；G中的边除了s中的边外，还包括邻居集到对应POI的边。

但这样构成的序列可能会使得计算量很大且有噪音点，所以在每个训练epoch时用**均匀采样**获得不同的邻居子集，从而在每次迭代的时候获得不同的图G，让模型学得高质量的POI表征和用户偏好。
#### Category-Aware Attention Layer
在建模稀疏的**POI-POI**转换的基础上，学习POI **category-wise**转换这种更稠密的序列依赖性有助于减缓POI-level稀疏性的影响。
当聚合结点信息时，利用注意力机制将**类别信息**融入到POI embedding中，同时要考虑**category-wise**转换。用**p**来表示使用当前结点的POI embedding和对应结点类型的embedding进行**拼接**后的embedding；用**q**表示p的某个邻居的embedding；用**r**表示**q->p的类型转换**(边向量)所对应的embedding。现在要利用注意力机制选择哪些邻居结点更重要，具体实现如下：

1.使用MLP将边向量r与q拼接后的结果进行非线性转换得到q的一个变体，考虑了**category-wise**的依赖关系

2.设计一个函数来计算邻居结点对当前结点的重要性(分数)

3.使用softmax函数归一化所有邻居结点的分数，并得到最终的加权和embedding **h**，包含了**邻居结点和类型**的信息，同时也考虑了**category-wise**相关性

所学得的**h**实质上是**p**的更新版本，由POI embedding和对应的**category** embedding组成。为了捕获**higher-order**信息(协同信号)每次训练结束后都要更新POI embedding和category embedding。
#### User Temporal Preference Encoding
之前所学得的**h**只包含相关的**空间和类型**信息，而没有考虑用户的短期偏好。因为先前POI对当前POI的影响会随着时间增加而降低，更近的POI影响更大。因此作者提出**position-aware**注意力机制来将序列信息融入到最终的用户时间偏好embedding **s**中。作者定义一个**Position embedding**矩阵，给每个position赋予一个不同的embedding，然后设计一个考虑序列信息的函数来评估**i-th** POI对最后一个POI的影响。最后用**softmax**得到每个POI对最后一个POI的影响，计算加权和。
#### Next POI and Category Prediction
**category-wise**转换比**POI-wise**转换更稠密，作者提出**多任务学习**机制来让模型同时预测next POI以及对应的category。作者将最后一个POI的embedding **h**和用户短期时间偏好**s**进行对应元素相乘操作(**将用户时间偏好和当前实例偏好融合**)，再和用户长期偏好embedding **u**进行拼接后作为一个全连接层的输入，预测**next POI**。同样地，作者将最后一个POI对应的类型embedding **c**和用户长期偏好embedding **u**拼接后作为一个全连接层的输入，预测**next POI category**。作者认为设计预测next POI category这个辅助task是为了提高模型捕获**categoty-wise**转换的能力。
#### Model Optimisation
使用cross-entropy损失函数来量化**POI和categoty**预测任务，**Figure 2**展示了整个模型的framework。
![Figure 2](/Users/raoxuan/Desktop/papers/Next poi recommendation/summary/Discovering Collaborative Signals for Next POI Recommendation with Iterative Seq2Graph Augmentation/FIgure2.png)
###  Question
用户长期偏好embedding **u**怎么来的？

文中提出学习**category-wise**转换，并且在训练过程中也计算了预测category的误差，那**POI-wise**转换体现在哪部分了？**position embedding**是根据每个POI与最后一个POI间的**offset**来赋予embedding，难道是在这体现的**POI-level**转换？照这样理解的话，整个训练过程用到了**category-level**和**POI-level**转换，所以说多了个预测category任务会使模型学得**category-wise**转换模式，增强模型的性能，否则多了个预测category任务为什么会起作用呢？

Figure 2中所学得的embedding **h**包括POI embedding和category embedding，所以用4格来表示，每个部分由2格表示。而每个position embedding应该用2格来表示吧，但Figure 2却画的4格？

Category-Aware Attention Layer那部分使用了MLP和LeakyReLU。是因为描述一个全连接神经网络很麻烦，所以论文直接使用MLP，毕竟功能是一样的？还有为什么使用LeakyReLu文中也没有解释？

###  Preliminary
**MLP**

最简单的全连接神经网络，包括输入层，隐藏层(至少一个)以及输出层。因为描述一个全连接神经网络很麻烦，所以论文直接使用MLP，毕竟功能是一样的？

**Attention Mechanism**

在众多信息中把注意力集中放在重要的点上，选出关键信息，而忽略其他不重要的信息。权重参数a来决定哪部分信息比较关键，而a一般是用**softmax**计算得到的概率。
