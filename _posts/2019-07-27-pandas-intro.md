---
layout: post
title: "Quick introduction to Pandas"
author: rishi
categories: [pandas python]
post-image: assets/images/pandas.jpg
---

This tutorial will serve as a quick introduction to Pandas for handling CSV data.  
CSV data is the most common format for storing and distributing structured data.  
Many Machine Learning datasets are distributed as CSV datafiles.  
After going through this tutorial, you will able to handle CSV files as input for your neural networks.  

![Panda]({{ site.baseurl }}/assets/images/pandas.jpg)

# Install Pandas
```bash
pip install pandas
```

# Reading a CSV with Pandas
```python
import pandas as pd
df = pd.read_csv('heart.csv')

print(df)
```
The read_csv() function helps us to read the CSV data from a file and into a DataFrame in our program.  

# Printing first few rows of a Dataframe
```python
print(df.head())
```
The head() command returns the first 5 rows of the DataFrame.  

# Selecting a single column
```python
age_column = df['age']
```
If the name of our column is "age", we can select that column in the above way.  

# Selecting certain rows and certain columns
```python
age_and_sex = df.iloc[0:10, 0:2]
```
If we want to select only the first 10 rows and the first 2 columns, we can do it with the iloc command. The slicing values are separated by a comma (first the row slicing, and then the column slicing).  

# Getting only the unique values in a column
```python
cp_values = df['cp'].unique()
```
After selecting a column, if we apply the unique() command, then only the unique values are returned (without repetition).  

# Replacing string values with integer values
```python
mapping = {'male': 0, 'female': 1}
df = df.replace(mapping)
```
The replace() function replaces the values in the DataFrame according to the mapping.  

# Converting to numpy array
```python
age_and_sex = age_and_sex.to_numpy()
```
The to_numpy() command converts the DataFrame to a Numpy Array. We can use this Numpy array as input to out Neural Network.  

# Shuffling the rows
```python
df = df.sample(frac=1)
```
The sample() method returns some rows from the DataFrame in a random order. If frac = 0.5, then half the rows are returned in a random order. If frac = 1, all the rows are returned in a random order.