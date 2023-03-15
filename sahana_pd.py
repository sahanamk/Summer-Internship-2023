'''1. Use pandas to package to read the file.'''

import pandas as pd
import numpy as np

data=pd.read_csv('D:/DLithe AIML/titanic.csv')

'''2. Upload the dataset as a dataframe in the variable named data'''

data_df=pd.DataFrame(data)
print(data)

'''3. Display the data types of the attributes of dataframe'''

print(data_df.dtypes)

'''4. Display the attribute names and save them in the a list (list data type)'''

print(data_df.columns)
attributes=list(data.columns)
print(attributes)

'''5. Write a function to check the index of the attribute named "age" and "survived" from the list'''

def attribute_index(attributes):
    age_index=data_df.columns.get_loc("age") if "age" in data_df.columns else None
    survival_index=data_df.columns.get_loc("survived") if "survived" in data_df.columns else None
    return age_index, survival_index
attribute_index(attributes)

'''6. Display the size of the whole dataset'''

print(data_df.size)

pass_above_60=data[data['age']>60]
print(pass_above_60)
pass_below_15=data[data['age']<=15]
print(pass_below_15)
age_15=(data['age']<=15).sum()
print(age_15)
pass_male_60=data[(data['age']>60)&(data['sex']=='male')]
print(pass_male_60) 

'''7. Display the summarized information of the dataframe'''

print(data_df.info())

'''8. Display the descriptive statistics of the dataframe. Write what have you analysed from this description
 as a comment inn the python script'''
 
print(data_df.describe())
#from the  stats of the dataframe we can analyse that the 'count' gives the number of non-null values for each column
#using the 'mean' 'std' we can get the mean and std deviation of each numerical column of the data frame
#'min and 'max' gives us the minimum and maximum value from each column and 25%, 50%, 75% will give percentiles of the values in the dataframe

'''9. Each column of the dataframe is referenced by its “Label”. Similar to numpy array we can use index based
referencing to reference elements in each column of the data frame. Using referencing print the first 10 
observations/values for 'age' Verify if there are NAN values in the dataframe?Count the nan in age 
attributeFill nan with mean of the age using numpy functionalitiesFill nan with mean of the age using pandas 
functionalities'''

for i in range(10):
     print(data_df.iloc[i]['age'])
print(data_df.isna().any())
print(data_df['age'].isna().sum())
data_df['age']=np.where(np.isnan(data_df['age']), np.nanmean(data_df['age']), data_df['age'])
data_df['age']=data_df['age'].fillna(data_df['age'].mean())

'''10. Dataframe object can take logical statements as inputs. Depending upon value of this logical index, it
will return the resulting dataframe object. Select all passenger with age greater than 60 and save in another
variable. Select all passengers with age less than or equal to 15. Select all number of passengers with Age 
less than or equal to 15 using conditional and logic operations find Passengers whose age is more than 60 and 
are male.'''

above_60=data_df[data_df['age']>60]
print(above_60)
below_15=data_df[data_df['age']<=15]
print(below_15)
age_15=(data_df['age']<=15).sum()
print(age_15)
male_60=data_df[(data_df['age']>60)&(data_df['sex']=='male')]
print(male_60)