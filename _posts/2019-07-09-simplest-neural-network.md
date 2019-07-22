---
layout: post
title:  "Simplest Neural Network (with Keras)"
author: rishi
featured: true
categories: [tensorflow, keras]
post-image: assets/images/workflow.png
publish: true
---

In this tutorial, we will learn how to write our own neural network.  
We will write the simplest possible neural network. It will have 1 layer with 1 neuron.  

Look at the following patterns:  
x => -1,  0,  1,  2,  3,  4  
y => -3, -1,  1,  3,  5,  7  

Can you guess the value of y for a given value of x ?  

Did you guess it ? 

What do you think the value of y will be if x = 10 ?  

Do you think our Neural Network can guess it ?  

Let's see  
For complete code: [Click here](https://nbviewer.jupyter.org/github/rishi93/thegoodpython_notebooks/blob/master/simple_neuron.ipynb)  

# Import the libraries
```python
import tensorflow as tf
import numpy as np
```

Tensorflow allows us to build neural networks in our Python program.  
Numpy allows us to create matrices very easily in our Python program.  

# Structure of the program

![Workflow]({{ site.baseurl }}/assets/images/workflow.png)

Let us discuss the basic structure of writing a deep learning program.  

# Step 1: Collect the training data
Collect as much training data as possible. These examples of correct X and Y, will help our neural network to learn the pattern. 

In this case, we have our training data of 6 examples of values of X and Y.  

```python
x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
```

# Step 2: Build the neural network
Build the neural network model: We will decide on how many layers and how many neurons in each layer.  

```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))
model.summary()
```

# Step 3: Compile the model
We will decide the **optimizer** algorithm and also the **loss function** to use for our neural network model.

```python
model.compile(optimizer='sgd', loss='mean_squared_error')
```

# Step 4: Feed the training data
The neural network tries to find a function that fits the given data. We will decide for how many epochs (iterations) our neural network will learn.art the learning process.  
```python
model.fit(x, y, epochs=500)
```

# Step 5: Test the model
We show the neural network, examples of data outside the training data. The neural network must guess the output for new data as accurately as possibleMake some predictions
```python
values = np.array([10.0])

predictions = model.predict(values)

print(predictions)
```