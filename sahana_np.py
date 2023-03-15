'''1. Create a 4X2 integer array using random.random and Prints its attributes; shape and dimensions'''
import numpy as np

arr1 = np.random.random([4,2]) 
print(arr1)

print("Array Shape: ", arr1.shape)
print("Array dimensions: ", arr1.ndim)


'''2. Create a 5X2 integer array from a range between 100 to 200 using arange() such that the difference 
between each element is 10'''

arr2 = np.arange(100, 200, 10).reshape(5,2)
print (arr2)


'''3. Create 3x3 array of floats from a nested list.'''

arr3 = np.array([[1.9, 7.6, 4.8],[ 6.6, 2.0, 5.7],[ 7.9, 8.6, 9.2]])
print(arr3)


'''4. Return array of odd rows and even columns from below numpy array
a = numpy.array([[3 ,6, 9, 12], [15 ,18, 21, 24], 
[27 ,30, 33, 36], [39 ,42, 45, 48], [51 ,54, 57, 60]])'''

a = np.array([[3 ,6, 9, 12], [15 ,18, 21, 24], 
[27 ,30, 33, 36], [39 ,42, 45, 48], [51 ,54, 57, 60]])
print(a)

arr4 = a[::2, 1::2]
print(arr4)


'''5. Create a 8X3 integer array from a range between 10 to 34 such that the difference between each 
element is 1 and then Split the array into four equal-sized sub-arrays.'''

arr5 = np.arange(10, 34, 1).reshape(8,3)
print (arr5)

sub_arr = np.split(arr5, 4) 
print(sub_arr)


'''6. What would be returned by this expression:
one_dim = np.arange(1,6)
one_dim + np.arange(5, 0, -1) ?'''

one_dim = np.arange(1,6)
print(one_dim)
one_dim + np.arange(5, 0, -1)

'''Ans.array([6, 6, 6, 6, 6])'''


'''7. Use any titanics_survival dataset from the datasets shared early today:-
Do the following:-
Find the size of the dataset. How many observations and attributes are there?
Find total survivals
Find total Males
Find count of Females
Find the count of survivals having age greater than 40
Use np.nan() to know if there are any null values?
if there are null values, replace thoses will np.fillna()'''

import numpy as np
import pandas as pd
 
data = pd.read_csv('D:/DLithe AIML/titanic.csv')
print(data)

rows, columns = data.shape
print(f"The dataset has {rows} rows and {columns} columns.")

print("Number of Observations: ", data.shape[0])
print("Number of Attributes: ", data.shape[1])


print("Total Survivals: ", data['survived'].sum())


print("Total Males: ", data[data['sex'] == 'male'].shape[0])


print("Total Females: ", data[data['sex'] == 'female'].shape[0])


print("Count of Survivals having age greater than 40: ", data[(data['age'] > 40) & (data['survived'] == 1)].shape[0])


print("Are there any Null Values: ", data.isnull().values.any())


data.fillna(np.nan, inplace=True)
print("Null Values After Replacing: ", data.isnull().values.any())


