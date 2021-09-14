### **MFNP: A Meta-optimized Model for Few-shot Next POI Recommendation**

### Innovation
由于有些用户的check-in数据很少，之前的基于序列分析模型(RNN) 来捕获用户偏好的研究不能给这些冷启动用户推荐有效的next POI。还有一些方法利用某些辅助信息(社会关系)来解决冷启动用户的稀疏性问题，但这些方法可能会给在相似环境下的用户推荐相同的POI，忽视了用户的实际兴趣。

作者认为用户在不同区域应该有不同的偏好，提出了用户兴趣漂移这种现象，而现有的基于meta-learning的方法是通过用户特定的设置来捕获用户偏好，忽视了这种现象，而且不能在region-level建模用户偏好。因此作者使用meta-learning设计了user-specific和region-specific的任务来学得region-independent的用户偏好和region-dependent的人群偏好，通过融合这两种偏好来捕获region-aware用户偏好，用于few-shot next POI推荐。对于region-independent的用户偏好，把为每个用户推荐POI看作一个学习任务，为每个用户构建一个meta-learner，学习具有较强泛化能力的全局参数来初始化每个任务的模型参数，然后这些参数学习每个用户的check-in数据从而学得task-specific参数。对于region-dependent的人群偏好，相似的用户应该有相似的行为模式，人群偏好从这些相似的用户中获得。先用聚类算法将用户分组，然后采用自适应网络平衡不同用户组的人群偏好的重要性。最后将这两种偏好融合获得region-aware的用户偏好。

### Definition
* 区域划分：利用Kmeans算法根据地理距离将所有POI聚类成K(30)个区域。
* 人群划分：划分成6个不同行为模式的人群。
* 区域轨迹集：轨迹就是一个用户的check-in序列，在某个区域r的所有用户的轨迹构成了区域r的轨迹集。

### Methodology
模型包括3个模块：
* user-specific模块：根据用户的check-in数据捕获用户的region-independent的个人偏好，与区域无关；
* region-specific模块：捕获region-dependent人群偏好；
* prediction模块：预测用户感兴趣的POI。

#### User-specific Preference Modelling
使用标准的LSTM网络学习用户check-in序列的时间关系，文中LSTM网络的输入x指对应POI的embedding向量，但怎么来的没有说。但是LSTM不能捕获离散的POI之间的空间关系，因此使用geo-dilated LSTM来学习序列的空间关系。geo-dilated LSTM使用不同的skip长度从check-in序列中选择POI作为输入，这个skip长度是根据地理因素自动决定的。最终，region-independent的用户偏好是LSTM和geo-dilated LSTM所学得的向量的平均值p。

#### Region-specific Preference Modelling
给定某个用户u，用所学得的region-independent的偏好p作为输入HDBSCAN算法的输入得到用户u属于哪个人群，根据用户u的check-in序列的当前(最后一个)POI，找到它对应的区域r，再找到区域r的轨迹集，从中确定与u属于同一人群的用户的轨迹，以时间顺序排序所有轨迹，用T表示排序后的轨迹集，使用LSTM网络来学习每个轨迹的所有POI的信息。

假设用户u的当前POI是o， 因为poi在人群中的受欢迎程度随着时间而变化(人们在工作日的中午选择去吃快餐，而周末的晚上选择去电影院)，另外，与o当前地理位置距离更近的轨迹更有参考意义，因此我们必须将时间和空间因素与用户u的个人轨迹结合起来，而不是将所有相应的轨迹简单地组合在一起。作者提出将一周划分成48个时间槽，其中24个槽对应工作日的24h，另24个槽对应周末的24h。对每个时间槽，作者将至少一个用户在该时间槽内参观过的POI集中起来构造一个POI集合，然后用Jaccard index计算i-th时间槽和j-th时间槽的时间相似度。这样的话，对于T中每条轨迹，都能得到一个对应的时间槽序列，根据用户u当前POI所对应的时间槽s，计算s与时间槽序列中每个时间槽的时间相似性w，w作为之前用LSTM网络学得每个轨迹的每个POI的hidden state的权重参数，计算加权和得到该轨迹的sequence-level表征。我们用相似的方法计算用户u的轨迹的sequence-level表征，只不过加权和操作变成求average pooling。

现在已经获得了用户u的轨迹表征以及轨迹集T中所有轨迹的sequence-level表征，然后使用geo-nonlocal操作融合空间信息来计算region-dependent人群偏好。

#### Prediction
已经学得了region-independent用户偏好和region-dependent人群偏好，将这两种偏好拼接后，经过一个投影矩阵再根据softmax得到预测结果。

### Meta-optimized Framework
作者使用MAML-based模型来优化user-specific模块和region-specific模块。
#### Local Update
先随机初始化全局参数。对于每个用户，我们有支持集和查询集。在支持集上进行训练，将每个用户的局部参数初始化为全局参数，然后计算region-independent用户偏好和region-dependent人群偏好，局部训练的优化目标是最小化单个用户的推荐损失。

#### Global Update
meta优化的目的是最小化查询集的预期损失。在支持集训练完后，根据查询集的损失来更新全局参数。
每个用户都有自己的局部参数，在局部训练完后学得一个模型，该模型的所有这些局部参数被用来更新全局参数。

### Question
* 公式(4)的HDBSCAN和公式(8)的geo-nonlocal operation不懂
* MAML-based模型不懂。
* Meta模型的局部参数如何用来优化全局参数还是不太懂。

### Preliminary

#### cold-start problem
如何在没有大量用户数据的情况下设计个性化推荐系统并且让用户对推荐结果满意从而愿意使用推荐系统。主要分为三大类：
* 用户冷启动：主要解决如何给新用户做个性化推荐的问题
* 物品冷启动：主要解决如何将新的物品推荐给可能对它感兴趣的用户
* 系统冷启动：主要解决如何在一个新开发的网站上设计个性化推荐系统。

#### Meta-learning
Meta-learning是指通过学习相似任务之间的经验从而快速适应仅有少量训练数据的新任务，用来克服冷启动问题。
