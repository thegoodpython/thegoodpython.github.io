---
layout: post
title: Iris species classification
author: rishi
categories: [tensorflow, keras]
post-image: assets/images/iris-species.png
---

In this tutorial, we will explore the Iris Dataset from UCI Machine Learning Repository.  

You can download the dataset from kaggle.com: [https://www.kaggle.com/uciml/iris](https://www.kaggle.com/uciml/iris). Please note that, you will need to sign in to download the dataset.  

## About the dataset
The length and width of the petal and sepal of three iris species are given.  

![Iris species]({{ site.baseurl }}/assets/images/iris-species.png)

Given the dimensions of the flower, we will predict the species of the flower. 
There are three Iris species: Iris setosa, Iris versicolor, Iris virginica  

The full code can be found here: [Click here](https://nbviewer.jupyter.org/github/rishi93/thegoodpython_notebooks/blob/master/iris-classification-new.ipynb)  

## Import the libraries
```python
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
We use tensorflow to build the neural network model.  
Numpy is used to handle n-dimensional numpy arrays.  
We use pandas to load the CSV (comma separated values) into a DataFrame. We can extract data from this DataFrame into Numpy arrays. We will use Numpy arrays as input to our Neural Network.  
Matplotlib is used to generate plots. We will plot the loss and accuracy during the training process.  

## Read the input data
```python
df = pd.read_csv('Iris.csv')
```
The read_csv function reads the CSV file into a Pandas DataFrame.  
To learn more about Pandas: [Click here]({{ site.baseurl }}{% post_url 2019-07-27-pandas-intro %})   

## Take a look at the data
```python
print(df.head())
```
The head() function returns just the first few rows of the DataFrame. If we print the entire DataFrame, then too much space will be taken up.  

## Take a look at the number of rows and columns
```python
rows, cols = df.shape
print(rows)
print(cols)
```
We take a look at the total number of samples and the number of features for each sample.  

## Extract the label names
```python
label_names = df['Species'].unique()
print(label_names)
```
We create a list of just the unique values from the 'Species' column of the DataFrame

## Convert label names to integer values
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

## Shuffle the dataset
```python
df = df.sample(frac=1)
```
The sample command returns some rows from the DataFrame, in a random order. If frac = 0.5, then half the rows will be returned in a random order. Since frac = 1, all the rows will be returned in a random order

## Split the dataset into training set and test set
```python
train_data = df.iloc[:120, :]
test_data = df.iloc[120:, :]

x_train = train_data.iloc[:120, 1:-1]
y_train = train_data.iloc[:120, -1:]

x_test = test_data.iloc[120:, 1:-1]
y_test = test_data.iloc[120:, -1:]
```
The iloc() command helps us to slice the DataFrame. For more detailed explanation, [Click here]({{ site.baseurl }}{% post_url 2019-07-27-pandas-intro %})  
We use 120 samples for training and 30 samples for testing. And we use the first 4 columns excluding the ID column as the input features, and the last column as the output label.  

## Scale the numerical inputs
```python
def scale_column(train_data, test_data, column):
    min_value = train_data[column].min()
    max_value = train_data[column].max()
    train_data[column] = (train_data[column] - min_value)/(max_value - min_value)
    test_data[column] = (test_data[column] - min_value)/(max_value - min_value)

scale_column(x_train, x_test, 'SepalLengthCm')
scale_column(x_train, x_test, 'SepalWidthCm')
scale_column(x_train, x_test, 'PetalLengthCm')
scale_column(x_train, x_test, 'PetalWidthCm')
```
Neural Networks work best when the input values lie between 0 and 1.  
We notice that the values of the width and length of the flower are much beyond the 0 and 1 range.  
To scale them down to the 0 to 1 range, we use Min-Max normalization. We subtract each value by the minimum value and then divide this by the difference between the maximum and minimum values.  
After scaling the training data, we use the same min and max values of the training data to scale the testing data also (None of the features of the testing data should be used for training)

After this operation, all our values will be between 0 and 1.  

## Convert to Numpy arrays
```python
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

x_test = x_test.to_numpy()
y_test = y_test.to_numpy()
```
We have been working so far with Pandas dataframes, we convert our data into numpy arrays, so that they can be input to the neural network easily.  

## Build the model
```python
model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(shape=[4]))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dense(units=3, activation='softmax'))
```
We build our neural network by using the Sequential method in Keras.  
In the Sequential approach, we add layers of neurons to our model by using the add() method.  
**Layer 1**  
Layer 1 is the input layer, we need to specify the shape of the input that this layer will expect. 
Since each single input is a one dimensional vector of dimension 4, we specify the shape as [4].  
**Layer 2**  
Layer 2 is the hidden layer, which will try to learn the correct answers for our inputs. We can add any number of neurons in this layer. We will first start with 64, to see if it is good enough. 
The activation function for the hidden layer is usually ReLU.  
**Layer 3**  
Layer 3 is the output layer. This layer will output the final predictions. Since we have 3 different classes in our problem, the number of neurons in the final layer is also 3.  
The output layer will output the probability of the input belonging to each class.  
For example: 
If Neuron 0 outputs 0.8, Neuron 1 outputs 0.1, and Neuron 2 outputs 0.1  
Then we can say with certainity that, the input belongs to class 0. 
(An important thing to note here is that the probabilities all add up to 1)  
The activation function for the output layer in case of multiclass (more than two classes) classification is usually softmax.  

## Choose the Loss function and optimizer
```python
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
In case of multiclass classification, we will always use categorical_crossentropy as our loss function. 
Since our outputs are not one-hot encoded, we will use the sparse_categorical_crossentropy.  
Adam is a better form of SGD (Stochastic Gradient Descent). We will use Adam as our optimizer.  
We want to focus on improving our accuracy (percentage of correct answers) as our metric.  

## Start the training process
```python
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=75)
```
We will train our neural network for 75 iterations. We can increase or decrease the number of iterations, to see which value is better. At the end of each epoch, we will evaluate the trained model with the test set (validation) to check the perfomance on new data.  
We will store the results of each epoch in the history variable, so that we can plot it later.  

## Evaluate with test set
```
test_loss, test_acc = model.evaluate(x_test, y_test)

print(test_acc)
```
We run our neural network on the test set. The test set contains examples that the neural network has never seen before. We print the accuracy of the neural network's output for the test set.  

## Plot training and validation loss
```python
plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'])
plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'])
plt.subplot(2, 2, 3)
plt.plot(history.history['val_loss'])
plt.subplot(2, 2, 4)
plt.plot(history.history['val_accuracy'])
plt.show()
```
We observe that our training and validation loss decreased steadily while the training and validation accuracy increased steadily. This is a good result.  

# Summary
We have managed to train a neural network to guess the iris species, based on the dimensions of the flower. The accuracy was over 90%. 