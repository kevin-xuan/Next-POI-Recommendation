## Curriculm Meta-Learning for Next POI Recommendation

The paper studies the **city-transfer next POI recommendation** problem, which utilizes the learned knowledge from **other cities' rich data** to provide recommendation for users within those **cold-start** cities with a **limited** number of user-POI interactions. 

There are two challenges that need to be addressed. The first challenge is that the shared data among different cities is limited. The second one is diverse behavior patterns among varoius users in different cities. In order to alleviate data sparsity, a meta-learning based approach is used to transfer knowledge from **base cities** to recommendation task within **target cities(cold-start cities)**. And the author proposes to employ the idea of **hard sample mining** to make models learn more from those **hard samples**. The hard samples are divided into two categories: **hard users** and **hard cities**. Besides, an **easy-to-hard** curriculm for **city-sampling pool** to help the meta-learner converge to a better state. Hence, the author proposes a **Curriculm Hardness Aware Meta-Learning (CHAML)** framework, which incorporates **hard sample mining** and **curriculm learning** into a meta-learning paradigm.

The **CHAML** framework is composed of two parts: **base recommender** and **meta-optimization**. And the base recommender will be introduced detailedly.
Specifically, **base recommender** consists of three modules, namely, **embedding module**, **attention module**, and **prediction module**. The **embedding module** is responsible for mapping **user history records** where each record contains **POI ID**, **category**, and **corresponding timestamp** and **candidate POI** into a list of embeddings and a embedding, respectively. Besides, a **user-POI distance** calculated by current user location and candidate POI is normalized by z-score standard and then concatenates to the embedding vector of each record and candidate POI embedding vector. As for **attention module**, it computes the similarity between **candidate POI embedding** and **each record embedding** and selects more important information to get an **attention vector**. The attention vector and candidate POI embedding are fed into **prediction module** to calculate the probability of the candidate POI. 

The key components of **curriculm learning** include a **difficulty measure** to decide which data is easy and a **training scheduler** to determine when to add more harder data for training. In this paper, the author regards **base cities** as training set and **some cities** as validate set, and determine the **difficulty** of each base city according the best score on validate set after training. Moreover, a single step scheduler is used to increase the number of cities **from half to all** at the end **half of the training rounds**.


The meta-training process of **CHAML** is as follows.

* Initialize the global parameter
* Calculate the difficulty of each base city and sort these cities by difficulty
* Randomly sample a batch of cities
* Randomly sample a batch of users from each city to form a batch of tasks
* **repeat**
* Update the city sampling pool
* Do a meta step, namely local update and global update
* Calculate user-level accuracies on query samples of each user
* Select some users with the lowest accuracies as hard user and re-sample new users to form a batch of tasks
* Do a meta step
* Calculate city-level accuracies on query set of each city
* Select some cities with lowest accuracies as hard cities and resample new cities to form a batch of cities
* Randomly sample a batch of users from current each city to form a new batch of cities
* **until** max step of iterations 




