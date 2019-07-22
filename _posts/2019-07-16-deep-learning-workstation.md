---
layout: post
title: "Setting up a Deep Learning Workstation"
date: 2019-07-16
categories: python
hidden: true
---

In this tutorial, we will see how to install and setup Python and Tensorflow on our system.  

We will go through the setup for both Windows and Linux(Ubuntu) systems.  
But Linux system is highly recommended for any programming purpose.  

There are two ways of installing Python and Tensorflow on your system:
1. Method 1 (Will only work on modern CPUs)
2. Method 2 (Will work on all CPUs)

# Method 1

## Step 1
Download the latest 64 bit version of Python from [https://python.org](https://python.org)  
![Step 1](/assets/images/ModernCPU/step1.jpg)

## Step 2
Run the exe file that you just downloaded
![Step 2](/assets/images/ModernCPU/installationwizard1.jpg)
![Step 2](/assets/images/ModernCPU/installationwizard2.jpg)

## Step 3
Make sure Python (64 bit) is successfully installed.
![Step 3](/assets/images/ModernCPU/installationwizard3.jpg)

## Step 4
Download the 'Visual Studio Code' editor from [https://code.visualstudio.com](https://code.visualstudio.com)  

## Step 5
Run the exe file that you just downloaded
![Step 5](/assets/images/ModernCPU/vscode1.jpg)
![Step 5](/assets/images/ModernCPU/vscode2.jpg)

## Step 6
Make sure Visual Studio Code is successfully installed.
![Step 6](/assets/images/ModernCPU/vscode3.jpg)

## Step 7
Create a new folder on your system. We will save all our programs in this folder.

## Step 8
Open Visual Studio Code. Click File -> Open Folder
![Step 8](/assets/images/ModernCPU/vscode_setfolder.jpg)  
The name of my folder was "Python_Programs"

## Step 9
Create a new file. File -> New

## Step 10
Write a program and save it to your folder
![Step 10](/assets/images/ModernCPU/vscode_savefile.jpg)  
The name of my program was test1.py   

## Step 11
Open a new Terminal in Visual Studio Code
![Step 11](/assets/images/ModernCPU/vscode_terminal.jpg)

## Step 12
Run the following command (**This is how you will execute any Python program that you write**)  
![Step 12](/assets/images/ModernCPU/test_python.jpg)

## Step 13
Time to install our packages  
We need the following packages for deep learning: Numpy, Matplotlib, and Tensorflow.  
Python comes with a package manager called "pip".  
pip allows us to install, upgrade and uninstall any package with a simple pip command.  
Run the following command to get the latest version of pip.  
![Step 13](/assets/images/ModernCPU/command0.jpg)

## Step 14
Let's use pip to install Numpy, Matplotlib and Tensorflow  
![Step 14](/assets/images/ModernCPU/command1.jpg)  
![Step 14](/assets/images/ModernCPU/command2.jpg)  
Make sure you get success message   
![Step 14](/assets/images/ModernCPU/command2_success.jpg)

## Step 15
Now that we have installed all our packages, let's get started with writing our deep learning programs.
