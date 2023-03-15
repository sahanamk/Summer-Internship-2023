"""Use machine learning, specifically KNN to predict survival of passenger on ill-fated RMS Titanic ship.

Steps to follow:-
1. Pre-processing
    * Reading the data
     # get titanic_train & test csv files as a DataFrame, you need to create two separate dataframes - one for train, one for test

    * Cleaning the data
          - # drop unnecessary columns, ['PassengerId','Name','Ticket'] columns won't be useful in analysis and prediction
         -# Embarked, # only for titanic fill the two missing values with the most occurred value, which is "S".
         -# group by embarked, and get the mean for survived passengers for each value in Embarked. Also plot boxplot on embarked and survived

# Either to consider Embarked column in predictions, and remove "S" dummy variable (using pd.get_dummies), and leave "C" & "Q", since they seem to have a good rate for Survival.
OR
don't create dummy variables for Embarked column, just drop it, because logically, Embarked doesn't seem to be useful in prediction.

# Fare
# only for test_df, since there is a missing "Fare" values
# convert from float to int, use .astype()
# get fare for survived & didn't survive passengers

#Age
# get average and std for fare of survived/not survived passengers
# get average, std, and number of NaN values in titanic_df , test_df
# generate two random numbers  between (mean - std) & (mean + std)
# plot original Age values, drop all null values, and convert to int
# fill NaN values in Age column with random values generated in
titanic_df , test_df
# convert from Age in both dataframes from float to int
# plot new Age Values (histogram with bins = 70. Write in comment your observations)
# barplot for average survived passengers by age

# Cabin
# It has a lot of NaN values, so it won't cause a remarkable impact on prediction"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

titanic_train = pd.read_csv('D:/DLithe AIML/Titanic_Survival_train.csv')
titanic_test = pd.read_csv('D:/DLithe AIML/Titanic_Survival_test.csv')

titanic_train.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
titanic_test.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)


titanic_train['Embarked'].fillna(titanic_train['Embarked'].mode()[0], inplace=True)

embarked_survived_mean = titanic_train.groupby('Embarked')['Survived'].mean()

titanic_train.boxplot(column='Survived', by='Embarked')
plt.show()


embarked_train = pd.get_dummies(titanic_train['Embarked'])
embarked_test = pd.get_dummies(titanic_test['Embarked'])

embarked_train.drop('S', axis=1, inplace=True)
embarked_test.drop('S', axis=1, inplace=True)

titanic_train = pd.concat([titanic_train, embarked_train], axis=1)
titanic_test = pd.concat([titanic_test, embarked_test], axis=1)

titanic_train.drop('Embarked', axis=1, inplace=True)
titanic_test.drop('Embarked', axis=1, inplace=True)


titanic_test['Fare'].fillna(titanic_test.groupby('Pclass')['Fare'].transform('median'), inplace=True)

titanic_train['Fare'] = titanic_train['Fare'].astype(int)
titanic_test['Fare'] = titanic_test['Fare'].astype(int)

survived_median_fare = titanic_train[titanic_train['Survived']==1]['Fare'].median()
not_survived_median_fare = titanic_train[titanic_train['Survived']==0]['Fare'].median()

print("Median fare for survived passengers:", survived_median_fare)
print("Median fare for didn't survive passengers:", not_survived_median_fare)


survived_fare = titanic_train[titanic_train['Survived'] == 1]['Fare']
not_survived_fare = titanic_train[titanic_train['Survived'] == 0]['Fare']
print("Survived fare - Mean: {:.2f}, Std: {:.2f}".format(survived_fare.mean(), survived_fare.std()))
print("Not survived fare - Mean: {:.2f}, Std: {:.2f}".format(not_survived_fare.mean(), not_survived_fare.std()))

for df in [titanic_train, titanic_test]:
    name =[x for x in globals() if globals()[x] is df][0]
    print("\nDataset: ", name)
    print("Fare - Mean: {:.2f}, Std: {:.2f}, NaN values: {}".format(df['Fare'].mean(), df['Fare'].std(), df['Fare'].isnull().sum()))
    
    rand1 = np.random.randint(df['Age'].mean()-df['Age'].std(), df['Age'].mean()+df['Age'].std())
    rand2 = np.random.randint(df['Age'].mean()-df['Age'].std(), df['Age'].mean()+df['Age'].std())
    
    df['Age'].hist(bins=20)
    df['Age'].dropna().astype(int)
    df.loc[df['Age'].isnull(), 'Age'] = np.random.randint(df['Age'].mean()-df['Age'].std(), df['Age'].mean()+df['Age'].std())
    print("Age - Mean: {:.2f}, Std: {:.2f}, NaN values: {}".format(df['Age'].mean(), df['Age'].std(), df['Age'].isnull().sum()))
    df['Age'] = df['Age'].astype(int)
    df['Age'].hist(bins=20)
    df['Age'] = df['Age'].astype(int)


# plot new Age Values
titanic_train['Age'].hist(bins=70)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# barplot for average survived passengers by age
age_survived = titanic_train.groupby('Age')['Survived'].mean()
plt.bar(age_survived.index, age_survived)
plt.title("Average Survived Passengers by Age")
plt.xlabel("Age")
plt.ylabel("Survival Rate")
plt.show()



'''2. Feature Engineering: Analyzing the data to decide on "features" to be used for modeling

# Family
# Instead of having two columns Parch & SibSp,
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not. Create a function and apply it on age, sex to create a new feature 'person' use .apply(). you can drop Sex column since we created Person column

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
'''

def has_family(df):
    return (df['Parch'] + df['SibSp']) > 0

# apply the has_family function on train_df and test_df
titanic_train['HasFamily'] = titanic_train.apply(has_family, axis=1).astype(int)
titanic_test['HasFamily'] = titanic_test.apply(has_family, axis=1).astype(int)

# create Person column
def get_person(df):
    age, sex = df
    if age < 16:
        return 'child'
    else:
        return sex

titanic_train['Person'] = titanic_train[['Age', 'Sex']].apply(get_person, axis=1)
titanic_test['Person'] = titanic_test[['Age', 'Sex']].apply(get_person, axis=1)

titanic_train = titanic_train.drop(['Sex'], axis=1)
titanic_test = titanic_test.drop(['Sex'], axis=1)

person_dummies_train = pd.get_dummies(titanic_train['Person'])
person_dummies_test = pd.get_dummies(titanic_test['Person'])

titanic_train = pd.concat([titanic_train, person_dummies_train], axis=1)
titanic_test = pd.concat([titanic_test, person_dummies_test], axis=1)

titanic_train = titanic_train.drop(['male'], axis=1)
titanic_test = titanic_test.drop(['male'], axis=1)

pclass_dummies_train = pd.get_dummies(titanic_train['Pclass'])
pclass_dummies_test = pd.get_dummies(titanic_test['Pclass'])

titanic_train = pd.concat([titanic_train, pclass_dummies_train], axis=1)
titanic_test = pd.concat([titanic_test, pclass_dummies_test], axis=1)

titanic_train = titanic_train.drop([3], axis=1)
titanic_test = titanic_test.drop([3], axis=1)

'''3. Implementation of KNN
# define training and testing sets
#implement knn'''

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train = titanic_train[['Pclass', 'female', 'SibSp', 'C', 'Q', 'Fare']]
y_train = titanic_train['Survived']

X_test = titanic_test[['Pclass', 'female', 'SibSp', 'C', 'Q', 'Fare']]
y_test = titanic_test['Survived']

X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_, y_train_)

y_pred = knn.predict(X_val)

y_test_pred = knn.predict(X_test)


"""4. Analyzing results
calculate accuracy, confusion matrix and display confusion matrix
"""
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

train_predictions = knn.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

test_predictions = knn.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Training accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)

cm = confusion_matrix(y_test, test_predictions)

sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")


