---
layout: post
title: Heart disease classification
author: rishi
categories: [tensorflow keras]
post-image: assets/images/heart-disease.png
---

In this tutorial, we will explore the Heart Disease dataset from the UCI Machine Learning Repository.  
You can download the dataset from Kaggle: [Click here](https://www.kaggle.com/ronitf/heart-disease-uci) (Please note that, you will need to sign in to download the dataset).  

![Healthy heart vs Heart attack]({{ site.baseurl }}/assets/images/heart-disease.png)

## About the dataset
13 different parameters from patients are recorded.  
There are 303 patients in total.  
Given these different parameters, we will predict if the patient has heart disease or not. 
target = 1 (Patient has heart disease)  
target = 0 (Patient has no heart disease)  

The 13 different parameters are:
1. **age** - Age  
2. **sex** - Sex  
3. **cp** - Chest pain type (4 different values):  
    1 = Typical Angina  
    2 = Atypical Angina  
    3 = Non-anginal pain  
    4 = Asymptotic  
4. **trestbps** - Resting blood pressure (on admission to the hospital)
5. **chol** - Serum cholestrol
6. **fbs** - Fasting blood sugar 
    0 = Fasting blood sugar is less than 120 mg/dl  
    1 = Fasting blood sugar is more than 120 mg/dl  
7. **restecg** - Resting Electrocardiograph results (3 different values)   
    0 = Normal  
    1 = Having ST-T wave abnormality  
    2 = Showing probable or definite left ventricular hypertropy  
8. **thalach** - Maximum heart rate achieved 
9. **exang** - Exercise induces Angina  
    0 = No  
    1 = Yes  
10. **oldpeak** - ST depression induced by exercise relative to rest  
11. **slope** - Slope of the peak exercise ST segment (3 different values)  
    1 = Up slope  
    2 = Flat  
    3 = Down slope  
12. **ca** - Number of major vessels colored by fluroscopy
13. **thal** - Thalium stress test result (4 different values)   

The full code can be found here: [Click here](https://nbviewer.jupyter.org/github/rishi93/thegoodpython_notebooks/blob/master/heart-disease.ipynb)  

## Import the libraries
```python
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
We use tensorflow to build the neural network model.  
Numpy is used to handle n-dimensional Numpy arrays.  
We use Pandas to load the CSV (Comma Separated Values) data into a DataFrame. We can extract data from this DataFrame into Numpy arrays. We will use the Numpy arrays as input to the Neural Networks.   
Matplotlib is used to generate plots. We will plot the loss and accuracy during the training process.  

## Read the input data
```python
df = pd.read_csv('heart.csv')
```
The read_csv function reads the CSV file into a Pandas DataFrame.  
To learn more about Pandas: [Click here]({{ site.baseurl }}{% post_url 2019-07-27-pandas-intro %})

## Take a look at the data
```python
print(df.head())
```
Print the first few rows of the dataset, to have a look at it.  

## Look at the number of rows and columns
```python
rows, columns = df.shape
print(rows)
print(columns)
```
We take a look the total number of samples and the number of features for each sample.  

## Shuffle the data
```python
df = df.sample(frac=1.0)
```

## Split the data into training and testing
```python
train_data = df.iloc[:250, :]
test_data = df.iloc[250:, :]

x_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1:]

x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1:]
```

## Scale the numerical inputs
```python
def scale_column(train_data, test_data, column):
    min_value = df[column].min()
    max_value = df[column].max()
    train_data[column] = (train_data[column] - min_value)/(max_value - min_value)
    test_data[column] = (test_data[column] - min_value)/(max_value - min_value)

scale_column(df, "age")
scale_column(df, "trestbps")
scale_column(df, "chol")
scale_column(df, "thalach")
scale_column(df, "oldpeak")
```
Neural networks work best when the input values lie between 0 and 1.  
We notice that certain numerical columns have values beyond the 0 to 1 range.  
To scale them down to the 0 to 1 range, we use Min-Max normalization. We subtract each value by the minimum value and then divide this by the difference between the maximum and minimum values.  

## Convert categorical input to one hot encoding
```python
def one_hot_encoding(train_data, test_data, column):
    train_values = train_data.pop(column)
    test_values = test_data.pop(column)
    unique_values = train_values.unique()
    unique_values = sorted(unique_values)
    for unique_value in unique_values:
        train_data[column + str(unique_value)] = (train_values == unique_value) * 1
        test_data[column + str(unique_value)] = (test_values == unique_value) * 1

one_hot_encoding(x_train, x_test, "cp")
one_hot_encoding(x_train, x_test, "slope")
one_hot_encoding(x_train, x_test, "ca")
one_hot_encoding(x_train, x_test, "thal")
```
We convert the categorical inputs into one-hot encoding values.  
Below is an example of one-hot encoding.  
![One-hot encoding]({{ site.baseurl }}/assets/images/one-hot-encoding.png)

## Take a look at the final transformed data
```python
print(x_train.head())
```
We take a look at our final data. We notice that we now have 25 columns.  

## Convert to Numpy arrays
```python
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

x_test = x_test.to_numpy()
y_test = y_test.to_numpy()
```

## Build the model
```python
model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(shape=[25]))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```
We build a Neural Network model with an input layer, one hidden layer, and one output layer.  
**Input layer**: Each patient's data consists of 25 values, so our input layer will accept a one-dimensional vector of 25 values.  
**Hidden layers**: We have 1 hidden layers with 32 neurons, and Rectified-Linear Unit activation.  
**Output layer**: Since we have only 2 output possibilities (Healthy or not healthy), a single neuron with sigmoid activation is sufficient. A sigmoid function will output either 0 or 1.  

## Choose Loss function and Optimizer
```python
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
```
Since we have only two classes (Healthy and Not Healthy), we use the binary_crossentropy loss function.  
We use the Adam optimizer, because it is better than plain SGD (stochastic gradient descent).  
We want to focus on accuracy (percentage of correct guesses) as our metric.  

## Start the training process
```python
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
```
We will train our neural network on the training data for 10 iterations.  

## Evaluate with Test set
```python
test_loss, test_acc = model.evaluate(x_test, y_test)

print(test_acc)
```
We run our neural network on the test set. The test set consists of examples that the neural network has never seen before. We print the accuracy of the neural network on the test set.  

## Plot the training and validation loss
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

## Summary
We have managed to train a neural network to predict if the patient has heart disease or not based on his hospital data. The accuracy was more than 83%.  