---
layout: post
title: Iris Dataset
author: rishi
featured: true
categories: [tensorflow, keras]
post-image: assets/images/iris-species.png
---

In this tutorial, we will explore the Iris Dataset from UCI Machine Learning Repository.  

You can download the dataset from kaggle.com: [https://www.kaggle.com/uciml/iris](https://www.kaggle.com/uciml/iris). Please note that, you will need to sign in to download the dataset.  

# About the dataset
The length and width of the petal and sepal of three iris species are given.  

![Iris species]({{ site.baseurl }}/assets/images/iris-species.png)

Given the dimensions of the flower, we will predict the species of the flower. 

The full code can be found here: [Click here](https://nbviewer.jupyter.org/github/rishi93/thegoodpython_notebooks/blob/master/iris-classification.ipynb)  

# Import the libraries
```python
import pandas as pd
import tensorflow as tf
```
We use pandas to load the CSV (comma separated values) into a DataFrame. We can extract data from this DataFrame into Numpy arrays. We will use Numpy arrays as input to our Neural Network.  

# Read the input data
```python
df = pd.read_csv('Iris.csv')
```
The read_csv function reads the CSV file into a Pandas DataFrame.  

# Take a look at the data
```python
print(df.head())
```
The head() function returns just the first few rows of the DataFrame. If we print the entire DataFrame, then too much space will be taken up.  

# Extract the label names
```python
label_names = df['Species'].unique()
print(label_names)
```
We create a list of just the unique values from the 'Species' column of the DataFrame

# Convert label names to integer values
```python
index_and_label = list(enumerate(label_names))
print(index_and_label)

label_to_index = dict((label, index) for index, label in index_and_label)
print(label_to_index)

df = df.replace(label_to_index)
```
Neural network can only work with integer values, and not string values.  
Therefore, We convert the string values to integer values starting from 0.  
The first label name is converted to 0, the second to 1, and so on...  

# Shuffle the dataset
```python
df = df.sample(frac=1)
```
The sample command returns some rows from the DataFrame, in a random order. If frac = 0.5, then half the rows will be returned in a random order. Since frac = 1, all the rows will be returned in a random order

# Split the dataset into training set and test set
```python
x_train = df.iloc[:120, 1:-1].to_numpy()
y_train = df.iloc[:120, -1:].to_numpy()

x_test = df.iloc[120:, 1:-1].to_numpy()
y_test = df.iloc[120:, -1:].to_numpy()
```
The iloc() command helps us to slice the DataFrame. For more detailed explanation, click here
The to_numpy() function converts the DataFrame into Numpy arrays. These numpy arrays will be the input to our neural network.  

# Build the model
```python
model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(shape=[4]))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dense(units=3, activation='softmax'))
```
We build our neural network by using the Sequential method in Keras.  
In the Sequential approach, we add layers of neurons to our model by using the add() method.  
## Layer 1
Layer 1 is the input layer, we need to specify the shape of the input that this layer will expect. 
Since each single input is a one dimensional vector of dimension 4, we specify the shape as [4].  
## Layer 2
Layer 2 is the hidden layer, which will try to learn the correct answers for our inputs. We can add any number of neurons in this layer. We will first start with 64, to see if it is good enough. 
## Layer 3
Layer 3 is the output layer. This layer will output the final predictions. Since we have 3 different classes in our problem, the number of neurons in the final layer is also 3.  
The output layer will output the probability of the input belonging to each class.  
For example: 
If Neuron 0 outputs 0.8, Neuron 1 outputs 0.1, and Neuron 2 outputs 0.1  
Then we can say with certainity that, the input belongs to class 0. 
(An important thing to note here is that the probabilities all add up to 1)  

# Choose the Loss function and optimizer
```python
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
In case of multiclass classification, we will always use categorical_crossentropy as our loss function. 
Adam is a better form of SGD (Stochastic Gradient Descent)
We want to focus on improving our accuracy (percentage of correct answers) as our metric.  

# Start the training process
```python
model.fit(x_train, y_train, epochs=100)
```
We will train our neural network for 100 iterations. We can increase or decrease the number of iterations, to see which value is better.  

# Evaluate with test set
```
test_loss, test_acc = model.evaluate(x_test, y_test)

print(test_acc)
```
We run our neural network on the test set. The test set contains examples that the neural network has never seen before. We print the accuracy of the neural network's output for the test set.  