---
layout: post
title: Object Detection
author: rishi
categories: [tensorflow]
post-image: 
---

In this tutorial, we will learn how to use the Object Detection API by Tensorflow.  

## Installation process
The Object Detection API doesn't yet work on Tensorflow 2.0  
So we will install Tensorflow 1.14 to work with the Object Detection API  
It is recommended to create a new virtual environment to install all the required dependencies.  

```bash
python3 -m venv odvenv

source odvenv/bin/activate
```
This command creates a virtual environment named *odvenv* and activates it.
To deactivate the virtual environment, you just need to
```bash
deactivate
```