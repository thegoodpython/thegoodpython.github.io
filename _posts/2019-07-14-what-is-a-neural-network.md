---
layout: post
title: "What is a neural network"
author: rishi
category: general
post-image: assets/images/neural_network.png
publish: true
---

## What is a Neural Network ?

> A Neural Network is a universal function approximator.  

What does that even mean ?  
Let's break it down.  
First, what is a function ?  
A function takes input X and gives an output Y.  
![Function]({{ site.baseurl }}/assets/images/function.png)
Therefore, a Neural Network is nothing but a function.  
It takes an input X and gives an output Y.  

"Universal function approximator" means that a Neural Network can *approximate* **any** function.  

## Why are Neural Networks useful ?  
Most problems in our world, can be broken down into functions.  
![Patient example]({{ site.baseurl }}/assets/images/real_example.png)
Suppose we are given an X-ray, and we want to predict if patient is healthy or not.  
In this case, Input X is the X-ray image, and the output is a prediction which can be "Healthy" or "Not Healthy".  
Another example: If input is the contents of an email (X), we want to predict if the email is "Spam" or "Not Spam" (Y).  
Since most problems in our world can be broken down into functions, Neural Networks can be used to approximate these functions, and solve our problems.  

## How does a Neural Network approximate a function ?  
This is the interesting part of Neural Networks. A Neural Network "learns" the function. Along the way, we will also learn some important terms :)  

Let's dive in.  

If a neural network is given many *correct examples* of X and Y, it will *learn* the relationship between X and Y. 

![Training data]({{ site.baseurl }}/assets/images/train_data.png)

> These "correct examples of X and Y are called **Training data**

![Neural Network]({{ site.baseurl }}/assets/images/neural_network.png)

A Neural Network will take in the first input example X1 and randomly guess an output Y1'  
Now, this *guess* might be wrong.  
The Neural Network compares the guess Y1' with the correct answer Y1. This comparison is done by an **Error function**

> Second important term to remember is Error function, also called **Loss function**.  

And then, the Neural Network will modify it's parameters to reduce the error. This modification is done by an **Optimizer**.  

> Third important term to remember is **Optimizer**

In conclusion,  
![Training]({{ site.baseurl }}/assets/images/training.png)  
We can think of training a Neural Network, like teaching a baby. If you show a baby, a photo of a car, and tell it that it is a car. A baby might make wrong guesses in the first few attempts. But after many attempts, the baby will learn to correctly recognize a car. Similarly, after showing many examples to a Neural Network, the Neural Network will learn to solve the problem.  

These "attempts" made by a baby are like the *iterations* of our neural network program. These iterations are called "Epochs". After every epoch, accuracy of Neural Network increases.  

> Another important term to remember is **Epochs**

## What is inside a neural network ?
Neural Networks are made up of many neurons. These neurons are arranged in layers.   
Neural Networks are required to learn complex functions to solve most real world problems. This complexity cannot be achieved by a single simple function. Therefore each neuron can be divided into 2 simple parts, A **linear function** and an **activation function**. (Don't worry if you don't understand this now. Just remember that it is there.) 
![Activation function]({{ site.baseurl }}/assets/images/activation.png)  

> Final important term to remember is **Activation function**  

# Terms to remember from this tutorial:
1. Training data  
2. Loss function
3. Optimizer
4. Epochs
5. Activation function

We will look at the indepth discussion of the mathematics behind these things, in a theoretical blogpost: [Link will be up later](/)  
But for now, let's program a neural network, by writing some Python code: [Click here]({{ site.baseurl }}{% post_url 2019-07-09-simplest-neural-network %})
