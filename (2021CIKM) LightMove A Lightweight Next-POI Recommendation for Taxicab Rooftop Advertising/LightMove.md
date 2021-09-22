## LightMove: A Lightweight Next-POI Recommendation for Taxicab Rooftop Advertising

The paper proposes a lightweight model named **LightMove** based on **neural oridinary differential equations (NODEs)**, which are robust to sparse input.
**NODEs** enable us to interpret the time variable as continuous, which is discrete in conventional neural networks. Therefore, **NODEs** are typically understood as a continuous generation of neural networks. Besides, **NODEs** could not only reduce the required number of parameters, but also produce similar hidden vectors for two similar inputs. The framework of **LightMove** is composed of four parts:

* short-term history module
* long-term history module
* NODE module
* prediction module

First, we learn the embedding vector for **each location**. Then, we divide 24h-hour band into slots and learn an embedding vector for each slot. In this way, we regard the **corresponding embeddings** as model inputs according to the given input set of locations with timestamp. Besides, we learn an embedding vector for each user.

As for **short-term** history module, a **dot-product-based** pair-wise **attention mechanism** is used to model user's short-term preference according to user's recent check-in records. Moreover, we use **all other** check-in records **before** recent check-ins used in **short-term** preference modeling to model **long-term** preference without attention. The concatenation of them, called **initial matrix**,  is the input of **NODE** modules.

The design for the **ODE function** is continuous generalization of **gated recurrent units (GRU)**. And **ODE** solver could convert an integral into many **steps(jumps)** of additions. Each jump is also done by **GRU**. In this way, the concatentation of the **output** of NODE module and **user embedding** is regarded as input of prediction module, which utilizes a fully connected layer followed by softmax function to calculate future locations prediction.

Note that the parameters of above **ODE function** in NODE module is fixed for each initial vector of **initial matrix**. The author proposes to produce adaptive parameters for each initial vector, which could greatly increase the model accuracy.