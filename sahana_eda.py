import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''1. You can check for size, shape, info etc'''
data = pd.read_excel('D:/DLithe AIML/Bengaluru_House_Data.xlsx')

'''2. Check for missing values and fill them if required. elaborate as a comment on your action taken on 
missing values and the reason for taking that action'''
data.isnull().sum()

location = data['location'].mode()[0]
data['location'].fillna(location, inplace = True)
#The Location for the houses are an important aspect of the dataset and cannot be left blank
#The mode() function gives the locations with the highest occurence and mode().[0] gives the first location if there are more than one locations with the same number of occurence as the mode

size = data['size'].mode()[0]
data['size'].fillna(size,inplace = True)
#The size of the houses are an important aspect of the dataset and cannot be left blank
#The mode() function gives the size with the highest occurence and mode().[0] gives the first size if there are more than one sizes with the same number of occurence as the mode

'''3. Do univariant and multivariant analysis, drop the attributes if not required'''
data['price'].head

#univariant analysis
data['price'].mean()
 
data['price'].median()
 
data['price'].std() 

data['price'].describe()
data['price'].value_counts()

data.boxplot(column=['price'], grid=False, color='black')
data.hist(column=['price'], grid=False, edgecolor='black')
data[data['price'] < 350]['price'].plot.hist()

#multivariant analysis
plt.scatter(data['price'], data['size'])

#dropping an attribute
data.drop(['society'], axis=1, inplace=True)

'''4. Verify if there are any outliers, remove outliers'''
Q1 = data['price'].quantile(0.25)
Q3 = data['price'].quantile(0.75)
IQR = Q3 - Q1

# Identify the outliers
outliers = (data['price'] < (Q1 - 1.5 * IQR)) | (data['price'] > (Q3 + 1.5 * IQR))

# Remove the outliers from the DataFrame
data = data[~outliers]

'''5.Save the clean data along with its description as a new dataframe'''
clean_data = data

clean_data.to_csv('D:/DLithe AIML/clean_data.csv', index=False)

df=pd.read_csv('D:/DLithe AIML/clean_data.csv') 
df.head()
